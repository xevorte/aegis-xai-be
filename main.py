import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf

logger = logging.getLogger("aegis-be")

# Path absolut agar aman dijalankan dari direktori mana pun
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MULTI_DIR = MODELS_DIR / "multi"

# ----------------------------------------------------
# LIFESPAN: LOAD MODEL KETIKA SERVER DIMULAI
# (pengganti @app.on_event("startup") yang deprecated)
# ----------------------------------------------------
models: dict = {}


def _load_one(name: str, filename: str, loader):
    path = MODELS_DIR / filename
    if not path.exists():
        logger.warning("⚠️ File model tidak ditemukan: %s (key '%s' dilewati)", path, name)
        return
    try:
        models[name] = loader(str(path))
        logger.info("✅ Loaded %s", filename)
    except Exception as e:
        logger.warning("⚠️ Gagal memuat %s: %s", filename, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading model & scaler ke memori dari %s ...", MODELS_DIR)
    _load_one("scaler", "scaler.pkl", joblib.load)
    _load_one("rf", "model_rf.pkl", joblib.load)
    _load_one("xgb", "model_xgb.pkl", joblib.load)
    _load_one("cnn1d", "model_cnn1d.h5", tf.keras.models.load_model)
    # SHAP explainer bersifat opsional: prediksi tetap jalan tanpanya
    _load_one("explainer", "shap_explainer.pkl", joblib.load)
    missing = {"scaler", "rf", "xgb", "cnn1d"} - models.keys()
    if missing:
        logger.warning("⚠️ Model belum lengkap, yang hilang: %s", sorted(missing))
    else:
        logger.info("✅ Seluruh model prediksi berhasil dimuat!")
    _load_best_model_config()
    _load_multi_models()
    yield
    models.clear()


# Inisialisasi FastAPI
app = FastAPI(
    title="Aegis XAI - Health Screening API",
    description=(
        "Backend Machine Learning Multi-Model & Explainable AI (XAI). "
        "Skrining diabetes (3 model inti) + indikasi awal 4 kondisi lain "
        "(hipertensi, kolesterol tinggi, penyakit jantung, riwayat stroke)."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# Konfigurasi CORS agar bisa diakses oleh Next.js (http://localhost:3000)
# NOTE: ["*"] tidak boleh digabung dengan allow_credentials=True (ditolak browser),
# jadi daftarkan origin eksplisit. Tambahkan domain produksi saat deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.6:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# PYDANTIC SCHEMA INPUT DARI FRONTEND
# Batasan ge/le mencegah data mustahil masuk ke model.
# ----------------------------------------------------
class HealthScreeningInput(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="1 = Tekanan Darah Tinggi, 0 = Normal")
    HighChol: int = Field(..., ge=0, le=1, description="1 = Kolesterol Tinggi, 0 = Normal")
    CholCheck: int = Field(default=1, ge=0, le=1)
    BMI: float = Field(..., ge=10, le=60, description="Indeks Massa Tubuh (misal: 24.5)")
    Smoker: int = Field(..., ge=0, le=1, description="1 = Merokok, 0 = Tidak")
    Stroke: int = Field(default=0, ge=0, le=1)
    HeartDiseaseorAttack: int = Field(default=0, ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1, description="1 = Olahraga rutin, 0 = Tidak")
    Fruits: int = Field(default=1, ge=0, le=1, description="1 = Rutin makan buah")
    Veggies: int = Field(default=1, ge=0, le=1, description="1 = Rutin makan sayur")
    HvyAlcoholConsump: int = Field(default=0, ge=0, le=1)
    AnyHealthcare: int = Field(default=1, ge=0, le=1)
    NoDocbcCost: int = Field(default=0, ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5, description="Skala Kesehatan Umum (1 = Sangat Baik, 5 = Buruk)")
    MentHlth: int = Field(default=0, ge=0, le=30, description="Hari gangguan mental per bulan (0-30)")
    PhysHlth: int = Field(default=0, ge=0, le=30, description="Hari gangguan fisik per bulan (0-30)")
    DiffWalk: int = Field(default=0, ge=0, le=1, description="1 = Kesulitan berjalan")
    Sex: int = Field(..., ge=0, le=1, description="1 = Laki-laki, 0 = Perempuan")
    Age: int = Field(..., ge=1, le=13, description="Kategori Usia 1-13 (1: 18-24th, 9: 60-64th, dst)")
    Education: int = Field(default=4, ge=1, le=6)
    Income: int = Field(default=6, ge=1, le=8)

# Nama kolom persis sesuai dataset CDC BRFSS
FEATURE_NAMES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

REQUIRED_PREDICT_KEYS = {"scaler", "rf", "xgb", "cnn1d"}

# ----------------------------------------------------
# BEST MODEL: dipilih train.py via validation PR-AUC (metrics.json).
# Dipakai endpoint /api/predict-best untuk konsumsi FE utama.
# ----------------------------------------------------
DEFAULT_BEST_MODEL_KEY = "xgb"  # fallback sebelum training menghasilkan metrics.json
DEFAULT_BEST_MODEL_NAME = "Gradient Boosting (XGBoost)"
best_model_key: str = DEFAULT_BEST_MODEL_KEY
best_model_display_name: str = DEFAULT_BEST_MODEL_NAME
best_model_metrics: dict | None = None
METRIC_TO_DISPLAY = {
    "rf": "Random Forest",
    "xgb": "Gradient Boosting (XGBoost)",
    "cnn1d": "1D-CNN (Deep Learning)",
}
decision_threshold: float = 0.5

# ----------------------------------------------------
# MULTI-KONDISI (INDIKATIF)
# Semua output adalah evaluasi internal pada dataset survei BRFSS.
# Tidak ada output yang menyatakan validasi klinis.
# ----------------------------------------------------
CONDITION_META = {
    "diabetes": {
        "display_name": "Diabetes",
        "label_column": "Diabetes_binary",
        "validated": False,
        "evidence_level": "internal_evaluation",
    },
    "hypertension": {
        "display_name": "Hipertensi",
        "label_column": "HighBP",
        "validated": False,
        "evidence_level": "preliminary",
    },
    "high_cholesterol": {
        "display_name": "Kolesterol Tinggi",
        "label_column": "HighChol",
        "validated": False,
        "evidence_level": "preliminary",
    },
    "heart_disease": {
        "display_name": "Penyakit Jantung",
        "label_column": "HeartDiseaseorAttack",
        "validated": False,
        "evidence_level": "preliminary",
    },
    "stroke": {
        "display_name": "Riwayat Stroke",
        "label_column": "Stroke",
        "validated": False,
        "evidence_level": "preliminary",
    },
}
MULTI_CONDITIONS = [k for k in CONDITION_META if k != "diabetes"]

DISCLAIMER_BASE = (
    "Skor model berdasarkan pola data populasi BRFSS; bukan diagnosis medis "
    "dan belum merupakan validasi klinis."
)
DISCLAIMER_PRELIMINARY = (
    "Kondisi ini diperkirakan dari status/riwayat pada survei, bukan kejadian "
    "kesehatan di masa depan."
)
DISCLAIMER_INTERNAL = (
    "Performa yang ditampilkan berasal dari evaluasi internal dengan pemisahan "
    "data train, validation, dan test."
)
SCORE_NOTE = "Skor model 0-100; bukan probabilitas medis terkalibrasi."


def _load_best_model_config():
    """Baca best_model dari metrics.json; fallback ke XGBoost bila belum ada."""
    global best_model_key, best_model_display_name, best_model_metrics, decision_threshold
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        logger.warning("⚠️ metrics.json belum ada, best model fallback ke %s. Jalankan train.py untuk pemilihan otomatis.", DEFAULT_BEST_MODEL_NAME)
        return
    try:
        data = json.loads(metrics_path.read_text())
        protocol = data.get("protocol", {})
        if protocol.get("validation_fraction") != 0.15:
            logger.warning("Metrik lama terdeteksi; jalankan ulang train.py untuk evaluasi 70/15/15.")
            return
        entry = data.get("best_model")
        if entry and entry.get("key") in METRIC_TO_DISPLAY and entry.get("selection_metric") == "validation_pr_auc":
            best_model_key = entry["key"]
            best_model_display_name = entry.get("display_name", METRIC_TO_DISPLAY[best_model_key])
            best_model_metrics = {
                "validation_pr_auc": entry.get("validation_pr_auc"),
                "validation_recall": entry.get("validation_recall"),
                "test_pr_auc": entry.get("test_pr_auc"),
                "test_recall": entry.get("test_recall"),
                "test_f1_score": entry.get("test_f1_score"),
                "test_ece": entry.get("test_ece"),
                "decision_threshold": data.get("models", {}).get(entry["key"], {}).get("validation", {}).get("decision_threshold", 0.5),
            }
            decision_threshold = best_model_metrics["decision_threshold"]
            logger.info("🏆 Best model (validation PR-AUC): %s", best_model_display_name)
        else:
            logger.warning("⚠️ metrics.json tanpa field best_model valid, fallback ke %s.", DEFAULT_BEST_MODEL_NAME)
    except Exception as e:
        logger.warning("⚠️ Gagal membaca metrics.json (%s), fallback ke %s.", e, DEFAULT_BEST_MODEL_NAME)


# ----------------------------------------------------
# HELPER: PEMBENTUK KALIMAT REKOMENDASI XAI (INDONESIA)
# Mencakup seluruh 21 fitur agar top-SHAP apa pun ada narasinya.
# ----------------------------------------------------
_XAI_TEMPLATES = {
    'BMI': "BMI menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Disarankan mengatur pola makan dan menurunkan berat badan ke rentang ideal.",
    'HighBP': "Tekanan darah tinggi menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Lakukan pemeriksaan rutin dan kurangi asupan garam.",
    'HighChol': "Kolesterol tinggi menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Batasi makanan berlemak jenuh dan cek lipid berkala.",
    'CholCheck': "Riwayat pemeriksaan kolesterol menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Pertahankan jadwal cek kolesterol minimal setahun sekali.",
    'Smoker': "Kebiasaan merokok menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Pertimbangkan program berhenti merokok.",
    'Stroke': "Riwayat stroke menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Lakukan kontrol neurologi dan pantau tekanan darah.",
    'HeartDiseaseorAttack': "Riwayat penyakit jantung/serangan jantung menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Konsultasikan ke dokter untuk evaluasi lanjutan.",
    'PhysActivity': "Aktivitas fisik menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Tingkatkan olahraga sesuai kondisi dan anjuran tenaga kesehatan.",
    'Fruits': "Konsumsi buah menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Tambahkan buah dalam pola makan seimbang.",
    'Veggies': "Konsumsi sayur menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Tambahkan sayur dalam pola makan seimbang.",
    'HvyAlcoholConsump': "Konsumsi alkohol menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Kurangi frekuensi dan jumlah konsumsi alkohol.",
    'AnyHealthcare': "Akses layanan kesehatan menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Pastikan akses layanan kesehatan tetap tersedia.",
    'NoDocbcCost': "Kendala biaya berobat menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Manfaatkan layanan kesehatan primer yang terjangkau.",
    'GenHlth': "Kondisi kesehatan umum menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Konsultasikan ke dokter untuk pemeriksaan menyeluruh.",
    'MentHlth': "Gangguan kesehatan mental ({d} hari/bulan) menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Kelola stres dan cari dukungan profesional bila perlu.",
    'PhysHlth': "Gangguan fisik ({d} hari/bulan) menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Evaluasi penyebabnya dan jangan tunda pemeriksaan.",
    'DiffWalk': "Kesulitan berjalan menjadi salah satu kontribusi relatif terbesar pada penjelasan model ({p}%). Lakukan evaluasi mobilitas.",
    'Sex': "Jenis kelamin menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Fokus pada faktor kesehatan yang dapat diubah.",
    'Age': "Usia menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Skrining berkala dapat dipertimbangkan.",
    'Education': "Pendidikan menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Tingkatkan literasi kesehatan melalui sumber tepercaya.",
    'Income': "Pendapatan menjadi salah satu kontribusi relatif pada penjelasan model ({p}%). Manfaatkan layanan kesehatan primer yang terjangkau.",
}


def generate_xai_recommendations(feature_importance_list, input_row: dict | None = None):
    """
    Menerjemahkan nilai kontribusi fitur tertinggi dari SHAP
    menjadi kalimat narasi rekomendasi bahasa Indonesia.
    """
    recommendations = []

    # Ambil 3 fitur dengan kontribusi risiko positif tertinggi
    top_risk_factors = [item for item in feature_importance_list if item['shap_value'] > 0][:3]

    for item in top_risk_factors:
        feat = item['feature']
        tpl = _XAI_TEMPLATES.get(feat)
        if not tpl:
            continue
        raw_val = (input_row or {}).get(feat, "")
        recommendations.append(tpl.format(p=item['impact_percent'], d=raw_val))

    if not recommendations:
        recommendations.append("Profil gaya hidup Anda saat ini menunjukkan indikator yang relatif stabil dan berisiko rendah.")

    return recommendations


def _require_models():
    missing = sorted(REQUIRED_PREDICT_KEYS - models.keys())
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model belum lengkap di server, hilang: {', '.join(missing)}. Restart server dan cek folder models/."
        )


def _run_three_models(input_df: pd.DataFrame, input_scaled: np.ndarray) -> dict:
    """Inferensi tiga model inti dan mengembalikan skor model."""
    prob_rf = float(models["rf"].predict_proba(input_df)[0][1])
    prob_xgb = float(models["xgb"].predict_proba(input_df)[0][1])
    input_cnn = input_scaled.reshape((1, input_scaled.shape[1], 1))
    prob_cnn = float(models["cnn1d"].predict(input_cnn, verbose=0)[0][0])
    return {"rf": prob_rf, "xgb": prob_xgb, "cnn1d": prob_cnn}


def _predict_single(model_key: str, input_df: pd.DataFrame, input_scaled: np.ndarray) -> float:
    """Inferensi 1 model saja (dipakai /api/predict-best)."""
    if model_key == "cnn1d":
        input_cnn = input_scaled.reshape((1, input_scaled.shape[1], 1))
        return float(models[model_key].predict(input_cnn, verbose=0)[0][0])
    # RF/XGBoost menggunakan data mentah; pipeline model menangani SMOTENC.
    return float(models[model_key].predict_proba(input_df)[0][1])


def _require_best_model():
    _require_models()
    missing = sorted({"scaler", best_model_key} - models.keys())
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model terbaik ('{best_model_key}') belum termuat di server. Restart server dan cek folder models/."
        )


def _load_multi_models():
    """Muat model terbaik per kondisi indikatif + buat TreeExplainer in-memory.

    Hanya 1 model per kondisi yang disimpan train.py (RF atau XGB), sehingga
    ringan untuk disk dan startup. Explainer tidak dipersist ke disk.
    """
    models["multi"] = {}
    models["multi_explainers"] = {}
    for cond in MULTI_CONDITIONS:
        cond_dir = MULTI_DIR / cond
        model_path = cond_dir / "model.pkl"
        metrics_path = cond_dir / "metrics.json"
        if not model_path.exists() or not metrics_path.exists():
            logger.warning(
                "⚠️ Model multi-kondisi '%s' belum tersedia. Jalankan train.py.", cond
            )
            continue
        try:
            meta = json.loads(metrics_path.read_text())
            model = joblib.load(str(model_path))
            models["multi"][cond] = {
                "model": model,
                "features": meta["features"],
                "display_name": meta.get("display_name", cond),
                "best_model": meta.get("best_model", {}),
            }
            models["multi_explainers"][cond] = shap.TreeExplainer(model)
            logger.info(
                "✅ Multi-kondisi %s dimuat (%s)",
                cond,
                meta.get("best_model", {}).get("display_name", "?"),
            )
        except Exception as e:
            logger.warning("⚠️ Gagal memuat multi-kondisi %s: %s", cond, e)


def _require_multi_model(condition: str):
    entry = models.get("multi", {}).get(condition)
    if entry is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model untuk kondisi '{condition}' belum tersedia. "
                "Jalankan train.py lalu restart server."
            ),
        )
    return entry



def _compute_shap_generic(explainer, input_df: pd.DataFrame, feature_names: list[str]):
    """Hitung SHAP untuk explainer + daftar fitur tertentu.

    Mengembalikan feature_importance_sorted | None. Tidak pernah raise
    agar endpoint prediksi tetap memberi hasil walau XAI gagal.
    """
    if explainer is None:
        logger.warning("SHAP explainer tidak tersedia, XAI dilewati.")
        return None
    try:
        # API baru (shap >= 0.41): explainer(X) -> Explanation
        explanation = explainer(input_df)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            # (n_samples, n_features, n_outputs) -> kelas positif
            shap_array = values[0][:, -1]
        elif values.ndim == 2:
            shap_array = values[0]
        else:
            shap_array = values
        if shap_array.ndim > 1:
            shap_array = shap_array[:, -1]
    except Exception as e_new:
        logger.warning("SHAP API baru gagal (%s), coba API lama...", e_new)
        try:
            # API lama: TreeExplainer.shap_values(X)
            shap_vals = explainer.shap_values(input_df)
            if isinstance(shap_vals, list):
                shap_array = np.asarray(shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0])
            else:
                arr = np.asarray(shap_vals)
                shap_array = arr[0] if arr.ndim >= 2 else arr
        except Exception as e_old:
            logger.warning("SHAP API lama juga gagal (%s). XAI dilewati.", e_old)
            return None

    total_shap = float(np.sum(np.abs(shap_array))) + 1e-6
    feature_importance = []
    for feat_name, s_val in zip(feature_names, shap_array):
        impact_pct = round((abs(float(s_val)) / total_shap) * 100, 1)
        feature_importance.append({
            "feature": feat_name,
            "shap_value": float(s_val),
            "impact_percent": impact_pct,
        })
    feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return feature_importance


def _compute_shap(input_df: pd.DataFrame):
    """SHAP untuk model diabetes (explainer RF yang dipersist)."""
    return _compute_shap_generic(models.get("explainer"), input_df, FEATURE_NAMES)


def _shap_for_condition(condition: str, input_df: pd.DataFrame):
    """SHAP untuk model multi-kondisi (TreeExplainer in-memory)."""
    entry = models.get("multi", {}).get(condition)
    if entry is None:
        return None
    df_cond = input_df[entry["features"]]
    return _compute_shap_generic(
        models.get("multi_explainers", {}).get(condition), df_cond, entry["features"]
    )


def _build_summary(all_probs: list[float]):
    avg_risk_probability = float(np.mean(all_probs))
    final_diagnosis = "BERISIKO DIABETES" if avg_risk_probability >= decision_threshold else "TIDAK BERISIKO"
    return avg_risk_probability, final_diagnosis


# ----------------------------------------------------
# ENDPOINT UTAMA: PREDIKSI MULTI-MODEL & XAI
# ----------------------------------------------------
@app.get("/")
def initiation():
    return {"status": "success", "message": "Backend Aegis XAI berjalan normal"}

@app.post("/api/predict")
def predict_health_risk(payload: HealthScreeningInput):
    _require_models()

    # 1. Format Payload menjadi DataFrame (kolom sesuai urutan dataset)
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    # 2. Inferensi tiga model inti
    probs = _run_three_models(input_df, input_scaled)
    all_probs = [probs["rf"], probs["xgb"], probs["cnn1d"]]
    avg_risk_probability, final_diagnosis = _build_summary(all_probs)

    # 3. Kalkulasi SHAP Values (opsional, tidak menggagalkan diagnosis)
    feature_importance = _compute_shap(input_df)
    if feature_importance is None:
        feature_importance = []
        xai_sentences = ["Modul penjelasan (XAI) tidak tersedia saat ini, namun skor diagnosis tetap valid."]
    else:
        xai_sentences = generate_xai_recommendations(feature_importance, input_dict)

    # 4. Susun Respons JSON (kontrak untuk FE Next.js + PDF)
    return {
        "status": "success",
        "summary": {
            "final_diagnosis": final_diagnosis,
            "overall_risk_score_percent": round(avg_risk_probability * 100, 2),
            "score_note": SCORE_NOTE,
            "consensus_voting": f"{sum(1 for p in all_probs if p >= decision_threshold)} dari 3 model mendeteksi risiko."
        },
        "model_predictions": [
            {"model_name": "Random Forest", "risk_probability": round(probs["rf"] * 100, 2), "prediction": "Berisiko" if probs["rf"] >= decision_threshold else "Aman"},
            {"model_name": "Gradient Boosting (XGBoost)", "risk_probability": round(probs["xgb"] * 100, 2), "prediction": "Berisiko" if probs["xgb"] >= decision_threshold else "Aman"},
            {"model_name": "1D-CNN (Deep Learning)", "risk_probability": round(probs["cnn1d"] * 100, 2), "prediction": "Berisiko" if probs["cnn1d"] >= decision_threshold else "Aman"}
        ],
        "explainable_ai": {
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences
        }
    }


@app.post("/api/predict-lite")
def predict_lite(payload: HealthScreeningInput):
    """Endpoint ringan untuk slider What-If Analysis: 3 model TANPA SHAP."""
    _require_models()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)
    probs = _run_three_models(input_df, input_scaled)
    all_probs = [probs["rf"], probs["xgb"], probs["cnn1d"]]
    avg_risk, diagnosis = _build_summary(all_probs)
    return {
        "status": "success",
        "summary": {
            "final_diagnosis": diagnosis,
            "overall_risk_score_percent": round(avg_risk * 100, 2),
            "score_note": SCORE_NOTE,
            "consensus_voting": f"{sum(1 for p in all_probs if p >= decision_threshold)} dari 3 model mendeteksi risiko."
        },
        "model_predictions": [
            {"model_name": "Random Forest", "risk_probability": round(probs["rf"] * 100, 2)},
            {"model_name": "Gradient Boosting (XGBoost)", "risk_probability": round(probs["xgb"] * 100, 2)},
            {"model_name": "1D-CNN (Deep Learning)", "risk_probability": round(probs["cnn1d"] * 100, 2)},
        ],
    }


@app.get("/api/metrics")
def get_metrics():
    """Menyajikan metrik evaluasi tiga model inti dan ensemble untuk dashboard."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json belum tersedia. Jalankan train.py terbaru untuk menghasilkan evaluasi."
        )
    try:
        return {"status": "success", "metrics": json.loads(metrics_path.read_text())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca metrics.json: {e}")


# ----------------------------------------------------
# ENDPOINT MODEL TERBAIK: 1 MODEL (validation PR-AUC tertinggi) + SHAP
# Dipakai FE utama; /api/predict tetap untuk komparasi jurnal
# ----------------------------------------------------
@app.post("/api/predict-best")
def predict_best(payload: HealthScreeningInput):
    """Prediksi dengan satu model terbaik (dipilih train.py via validation PR-AUC).

    Cocok untuk halaman utama FE. SHAP dihitung dari explainer Random Forest
    (wakil analisis fitur), dinyatakan eksplisit di field xai_basis_model.
    """
    _require_best_model()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    prob = _predict_single(best_model_key, input_df, input_scaled)
    threshold = (best_model_metrics or {}).get("decision_threshold", decision_threshold)
    diagnosis = "BERISIKO DIABETES" if prob >= threshold else "TIDAK BERISIKO"

    feature_importance = _compute_shap(input_df)
    if feature_importance is None:
        feature_importance = []
        xai_sentences = ["Modul penjelasan (XAI) tidak tersedia saat ini, namun skor diagnosis tetap valid."]
    else:
        xai_sentences = generate_xai_recommendations(feature_importance, input_dict)

    response = {
        "status": "success",
        "best_model": {
            "key": best_model_key,
            "display_name": best_model_display_name,
            "risk_probability": round(prob * 100, 2),
            "prediction": "Berisiko" if prob >= threshold else "Aman",
            "decision_threshold": threshold,
        },
        "summary": {
            "final_diagnosis": diagnosis,
            "risk_score_percent": round(prob * 100, 2),
            "score_note": SCORE_NOTE,
        },
        "explainable_ai": {
            "xai_basis_model": METRIC_TO_DISPLAY.get("rf", "Random Forest"),
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences,
        },
    }
    if best_model_metrics:
        response["best_model"]["validation_pr_auc"] = best_model_metrics.get("validation_pr_auc")
        response["best_model"]["validation_recall"] = best_model_metrics.get("validation_recall")
        response["best_model"]["test_pr_auc"] = best_model_metrics.get("test_pr_auc")
        response["best_model"]["test_recall"] = best_model_metrics.get("test_recall")
        response["best_model"]["test_f1_score"] = best_model_metrics.get("test_f1_score")
        response["best_model"]["test_ece"] = best_model_metrics.get("test_ece")
        response["best_model"]["metrics_note"] = "Metrik dipilih dari validation dan dilaporkan pada test internal."
    return response


ACTIONABLE_COUNTERFACTUALS = {
    "BMI": lambda value: [round(max(18.5, value - step), 1) for step in (1, 2, 3, 5) if value - step >= 10],
    "Smoker": lambda value: [0] if value == 1 else [],
    "PhysActivity": lambda value: [1] if value == 0 else [],
    "Fruits": lambda value: [1] if value == 0 else [],
    "Veggies": lambda value: [1] if value == 0 else [],
    "HvyAlcoholConsump": lambda value: [0] if value == 1 else [],
    "GenHlth": lambda value: list(range(max(1, value - 2), value)),
    "MentHlth": lambda value: [max(0, value - step) for step in (5, 10, 15) if value > 0],
    "PhysHlth": lambda value: [max(0, value - step) for step in (5, 10, 15) if value > 0],
}


def _counterfactual_candidates(input_dict: dict) -> list[tuple[str, object]]:
    candidates = []
    for feature, generator in ACTIONABLE_COUNTERFACTUALS.items():
        for value in generator(input_dict[feature]):
            candidates.append((feature, value))
    return candidates


@app.post("/api/counterfactual")
def counterfactual(payload: HealthScreeningInput):
    """Cari perubahan minimal pada fitur yang dapat diubah untuk simulasi skor model."""
    _require_best_model()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)
    baseline = _predict_single(best_model_key, input_df, input_scaled)
    target = (best_model_metrics or {}).get("decision_threshold", decision_threshold)
    if baseline < target:
        return {
            "status": "success",
            "found": False,
            "reason": "Skor awal sudah berada di bawah threshold model.",
            "baseline_score": round(baseline * 100, 2),
            "target_score": round(target * 100, 2),
            "changes": [],
            "disclaimer": "Simulasi model, bukan rekomendasi medis atau hubungan sebab-akibat.",
        }

    options = []
    for feature, value in _counterfactual_candidates(input_dict):
        candidate = dict(input_dict)
        candidate[feature] = value
        candidate_df = pd.DataFrame([candidate])[FEATURE_NAMES]
        candidate_scaled = models["scaler"].transform(candidate_df)
        score = _predict_single(best_model_key, candidate_df, candidate_scaled)
        options.append((score, feature, input_dict[feature], value))
    options.sort(key=lambda item: (item[0], item[1]))
    successful = [item for item in options if item[0] < target]
    if not successful:
        return {
            "status": "success",
            "found": False,
            "reason": "Tidak ditemukan perubahan satu fitur yang melewati threshold pada ruang simulasi terbatas.",
            "baseline_score": round(baseline * 100, 2),
            "target_score": round(target * 100, 2),
            "changes": [],
            "disclaimer": "Simulasi model, bukan rekomendasi medis atau hubungan sebab-akibat.",
        }
    score, feature, old_value, new_value = successful[0]
    return {
        "status": "success",
        "found": True,
        "baseline_score": round(baseline * 100, 2),
        "counterfactual_score": round(score * 100, 2),
        "target_score": round(target * 100, 2),
        "changes": [{"feature": feature, "from": old_value, "to": new_value}],
        "disclaimer": "Simulasi model, bukan rekomendasi medis atau hubungan sebab-akibat.",
    }


def _diabetes_condition_block(input_dict: dict, input_df: pd.DataFrame, input_scaled: np.ndarray) -> dict:
    """Blok kondisi diabetes: tiga model inti + ensemble + SHAP."""
    probs = _run_three_models(input_df, input_scaled)
    all_probs = [probs["rf"], probs["xgb"], probs["cnn1d"]]
    avg_risk, diagnosis = _build_summary(all_probs)

    feature_importance = _compute_shap(input_df)
    if feature_importance is None:
        feature_importance = []
        xai_sentences = ["Modul penjelasan (XAI) tidak tersedia saat ini, namun skor diagnosis tetap valid."]
    else:
        xai_sentences = generate_xai_recommendations(feature_importance, input_dict)

    return {
        "condition": "diabetes",
        "display_name": "Diabetes",
        "diagnosis": diagnosis,
        "risk_percent": round(avg_risk * 100, 2),
        "score_note": SCORE_NOTE,
        "best_model": {
            "key": best_model_key,
            "display_name": best_model_display_name,
            "validation_pr_auc": (best_model_metrics or {}).get("validation_pr_auc"),
            "validation_recall": (best_model_metrics or {}).get("validation_recall"),
            "test_pr_auc": (best_model_metrics or {}).get("test_pr_auc"),
            "test_recall": (best_model_metrics or {}).get("test_recall"),
            "test_f1_score": (best_model_metrics or {}).get("test_f1_score"),
            "test_ece": (best_model_metrics or {}).get("test_ece"),
            "metrics_note": "Metrik dipilih dari validation dan dilaporkan pada test internal.",
        },
        "consensus": f"{sum(1 for p in all_probs if p >= decision_threshold)} dari 3 model mendeteksi risiko.",
        "models": [
            {"model_name": "Random Forest", "risk_probability": round(probs["rf"] * 100, 2)},
            {"model_name": "Gradient Boosting (XGBoost)", "risk_probability": round(probs["xgb"] * 100, 2)},
            {"model_name": "1D-CNN (Deep Learning)", "risk_probability": round(probs["cnn1d"] * 100, 2)},
        ],
        "evidence_level": "internal_evaluation",
        "validated": False,
        "disclaimer": f"{DISCLAIMER_INTERNAL} {DISCLAIMER_BASE}",
        "explainable_ai": {
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences,
        },
    }


def _multi_condition_block(condition: str, input_dict: dict, input_df: pd.DataFrame) -> dict:
    """Blok kondisi indikatif: 1 model terbaik + SHAP + disclaimer eksplisit."""
    meta = CONDITION_META[condition]
    entry = _require_multi_model(condition)
    best = entry.get("best_model", {})

    df_cond = input_df[entry["features"]]
    prob = float(entry["model"].predict_proba(df_cond)[0][1])
    diagnosis = "INDIKASI RISIKO" if prob >= 0.5 else "RISIKO RENDAH"

    feature_importance = _shap_for_condition(condition, input_df)
    if feature_importance is None:
        feature_importance = []
        xai_sentences = ["Modul penjelasan (XAI) tidak tersedia saat ini, namun skor indikasi tetap valid."]
    else:
        xai_sentences = generate_xai_recommendations(feature_importance, input_dict)

    return {
        "condition": condition,
        "display_name": meta["display_name"],
        "diagnosis": diagnosis,
        "risk_percent": round(prob * 100, 2),
        "score_note": SCORE_NOTE,
        "best_model": {
            "key": best.get("key"),
            "display_name": best.get("display_name", "?"),
            "validation_pr_auc": best.get("validation_pr_auc", best.get("pr_auc")),
            "validation_recall": best.get("validation_recall", best.get("recall")),
            "test_pr_auc": best.get("test_pr_auc", best.get("pr_auc")),
            "test_recall": best.get("test_recall", best.get("recall")),
            "test_ece": best.get("test_ece", best.get("ece")),
            "metrics_note": "Metrik kondisi tambahan berasal dari artefak preliminary; bukan eksperimen inti diabetes.",
        },
        "evidence_level": meta["evidence_level"],
        "validated": meta["validated"],
        "disclaimer": f"{DISCLAIMER_PRELIMINARY} {DISCLAIMER_INTERNAL} {DISCLAIMER_BASE}",
        "explainable_ai": {
            "xai_basis_model": best.get("display_name", "?"),
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences,
        },
    }


# ----------------------------------------------------
# ENDPOINT MULTI-KONDISI: semua kondisi sekaligus / per kondisi
# Kondisi non-diabetes bersifat indikatif (preliminary), di luar eksperimen inti.
# ----------------------------------------------------
@app.post("/api/predict-all")
def predict_all(payload: HealthScreeningInput):
    """Satu request -> blok risiko untuk 5 kondisi."""
    _require_models()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    conditions = {}
    for cond in CONDITION_META:
        if cond == "diabetes":
            conditions[cond] = _diabetes_condition_block(input_dict, input_df, input_scaled)
            continue
        if cond not in models.get("multi", {}):
            conditions[cond] = {
                "condition": cond,
                "display_name": CONDITION_META[cond]["display_name"],
                "available": False,
                "error": "Model belum tersedia; jalankan train.py lalu restart server.",
        "evidence_level": "internal_evaluation",
        "validated": False,
        "disclaimer": f"{DISCLAIMER_INTERNAL} {DISCLAIMER_BASE}",
            }
            continue
        try:
            conditions[cond] = _multi_condition_block(cond, input_dict, input_df)
        except Exception as e:
            logger.warning("⚠️ Gagal prediksi kondisi %s: %s", cond, e)
            conditions[cond] = {
                "condition": cond,
                "display_name": CONDITION_META[cond]["display_name"],
                "available": False,
                "error": "Gagal menghitung indikasi kondisi ini.",
                "evidence_level": "preliminary",
                "validated": False,
                "disclaimer": f"{DISCLAIMER_PRELIMINARY} {DISCLAIMER_INTERNAL} {DISCLAIMER_BASE}",
            }

    return {"status": "success", "input_echo": input_dict, "conditions": conditions}


@app.post("/api/predict/{condition}")
def predict_one_condition(condition: str, payload: HealthScreeningInput):
    """Prediksi satu kondisi: diabetes, hypertension, high_cholesterol, heart_disease, stroke."""
    cond = condition.strip().lower()
    if cond not in CONDITION_META:
        raise HTTPException(
            status_code=404,
            detail=f"Kondisi '{condition}' tidak dikenal. Pilihan: {', '.join(CONDITION_META)}.",
        )
    _require_models()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    if cond == "diabetes":
        block = _diabetes_condition_block(input_dict, input_df, input_scaled)
    else:
        block = _multi_condition_block(cond, input_dict, input_df)
    return {"status": "success", "input_echo": input_dict, **block}



@app.get("/health")
def health_check():
    loaded = sorted(k for k in models if k in REQUIRED_PREDICT_KEYS or k == "explainer")
    multi_status = {}
    for cond in MULTI_CONDITIONS:
        entry = models.get("multi", {}).get(cond)
        multi_status[cond] = {
            "loaded": entry is not None,
            "display_name": CONDITION_META[cond]["display_name"],
            "best_model": (entry or {}).get("best_model", {}).get("display_name"),
        }
    return {
        "status": "ok",
        "message": "Backend FastAPI ML Aegis XAI berjalan dengan normal.",
        "version": app.version,
        "loaded_models": loaded,
        "best_model": {
            "key": best_model_key,
            "display_name": best_model_display_name,
            "validation_pr_auc": (best_model_metrics or {}).get("validation_pr_auc"),
            "validation_recall": (best_model_metrics or {}).get("validation_recall"),
            "test_pr_auc": (best_model_metrics or {}).get("test_pr_auc"),
            "test_recall": (best_model_metrics or {}).get("test_recall"),
            "test_f1_score": (best_model_metrics or {}).get("test_f1_score"),
            "test_ece": (best_model_metrics or {}).get("test_ece"),
        },
        "multi_conditions": multi_status,
    }
