import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf

logger = logging.getLogger("aegis-be")

# Path absolut agar aman dijalankan dari direktori mana pun
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

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
    _load_one("svm", "model_svm.pkl", joblib.load)
    _load_one("dt", "model_dt.pkl", joblib.load)
    _load_one("rf", "model_rf.pkl", joblib.load)
    _load_one("xgb", "model_xgb.pkl", joblib.load)
    _load_one("cnn1d", "model_cnn1d.h5", tf.keras.models.load_model)
    # SHAP explainer bersifat opsional: prediksi tetap jalan tanpanya
    _load_one("explainer", "shap_explainer.pkl", joblib.load)
    missing = {"scaler", "svm", "dt", "rf", "xgb", "cnn1d"} - models.keys()
    if missing:
        logger.warning("⚠️ Model belum lengkap, yang hilang: %s", sorted(missing))
    else:
        logger.info("✅ Seluruh model prediksi berhasil dimuat!")
    _load_best_model_config()
    yield
    models.clear()


# Inisialisasi FastAPI
app = FastAPI(
    title="Aegis XAI - Health Screening API",
    description="Backend Machine Learning Multi-Model & Explainable AI (XAI)",
    version="1.0.0",
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

REQUIRED_PREDICT_KEYS = {"scaler", "svm", "dt", "rf", "xgb", "cnn1d"}

# ----------------------------------------------------
# BEST MODEL: dipilih train.py via F1-Score tertinggi (metrics.json).
# Dipakai endpoint /api/predict-best untuk konsumsi FE utama.
# ----------------------------------------------------
DEFAULT_BEST_MODEL_KEY = "xgb"  # fallback sebelum training menghasilkan metrics.json
DEFAULT_BEST_MODEL_NAME = "Gradient Boosting (XGBoost)"
best_model_key: str = DEFAULT_BEST_MODEL_KEY
best_model_display_name: str = DEFAULT_BEST_MODEL_NAME
best_model_metrics: dict | None = None
METRIC_TO_DISPLAY = {
    "svm": "Support Vector Machine (SVM)",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "Gradient Boosting (XGBoost)",
    "cnn1d": "1D-CNN (Deep Learning)",
}


def _load_best_model_config():
    """Baca best_model dari metrics.json; fallback ke XGBoost bila belum ada."""
    global best_model_key, best_model_display_name, best_model_metrics
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        logger.warning("⚠️ metrics.json belum ada, best model fallback ke %s. Jalankan train.py untuk pemilihan otomatis.", DEFAULT_BEST_MODEL_NAME)
        return
    try:
        data = json.loads(metrics_path.read_text())
        entry = data.get("best_model")
        if entry and entry.get("key") in METRIC_TO_DISPLAY:
            best_model_key = entry["key"]
            best_model_display_name = entry.get("display_name", METRIC_TO_DISPLAY[best_model_key])
            best_model_metrics = {
                "f1_score": entry.get("f1_score"),
                "accuracy": entry.get("accuracy"),
            }
            logger.info("🏆 Best model (F1): %s", best_model_display_name)
        else:
            logger.warning("⚠️ metrics.json tanpa field best_model valid, fallback ke %s.", DEFAULT_BEST_MODEL_NAME)
    except Exception as e:
        logger.warning("⚠️ Gagal membaca metrics.json (%s), fallback ke %s.", e, DEFAULT_BEST_MODEL_NAME)


# ----------------------------------------------------
# HELPER: PEMBENTUK KALIMAT REKOMENDASI XAI (INDONESIA)
# Mencakup seluruh 21 fitur agar top-SHAP apa pun ada narasinya.
# ----------------------------------------------------
_XAI_TEMPLATES = {
    'BMI': "Indeks Massa Tubuh (BMI) Anda berkontribusi sebesar +{p}% terhadap kenaikan risiko. Disarankan mengatur pola makan dan menurunkan berat badan ke rentang ideal.",
    'HighBP': "Indikator Tekanan Darah Tinggi meningkatkan risiko sebesar +{p}%. Lakukan pemeriksaan rutin dan kurangi asupan garam.",
    'HighChol': "Tingkat Kolesterol Tinggi menambah beban risiko sebesar +{p}%. Batasi makanan berlemak jenuh dan cek lipid berkala.",
    'CholCheck': "Riwayat pemeriksaan kolesterol berkontribusi +{p}%. Pertahankan jadwal cek kolesterol minimal setahun sekali.",
    'Smoker': "Kebiasaan merokok berkontribusi sebesar +{p}% terhadap risiko. Pertimbangkan program berhenti merokok.",
    'Stroke': "Riwayat stroke menyumbang risiko sebesar +{p}%. Lakukan kontrol neurologi dan pantau tekanan darah ketat.",
    'HeartDiseaseorAttack': "Riwayat penyakit jantung/serangan jantung menyumbang +{p}% risiko. Konsultasikan ke kardiolog untuk evaluasi lanjutan.",
    'PhysActivity': "Kurangnya aktivitas fisik menambah risiko sebesar +{p}%. Tingkatkan olahraga minimal 150 menit per minggu.",
    'Fruits': "Rendahnya konsumsi buah berkontribusi +{p}% terhadap risiko. Tambahkan minimal 2 porsi buah per hari.",
    'Veggies': "Rendahnya konsumsi sayur berkontribusi +{p}% terhadap risiko. Tambahkan sayur setiap kali makan.",
    'HvyAlcoholConsump': "Konsumsi alkohol berat menyumbang +{p}% risiko. Kurangi frekuensi dan jumlah konsumsi alkohol.",
    'AnyHealthcare': "Keterbatasan akses layanan kesehatan berkontribusi +{p}%. Pastikan kepemilikan jaminan/asuransi kesehatan aktif.",
    'NoDocbcCost': "Kendala biaya berobat menyumbang +{p}% risiko. Manfaatkan layanan puskesmas/BPJS untuk skrining rutin.",
    'GenHlth': "Kondisi kesehatan umum yang menurun menyumbang risiko sebesar +{p}%. Konsultasikan ke dokter untuk pemeriksaan menyeluruh.",
    'MentHlth': "Gangguan kesehatan mental ({d} hari/bulan) berkontribusi +{p}%. Kelola stres, tidur cukup, dan cari dukungan profesional bila perlu.",
    'PhysHlth': "Gangguan fisik ({d} hari/bulan) berkontribusi +{p}%. Evaluasi penyebab fisik dan jangan tunda pemeriksaan.",
    'DiffWalk': "Kesulitan berjalan menyumbang +{p}% risiko. Lakukan fisioterapi dan evaluasi mobilitas.",
    'Sex': "Faktor demografis jenis kelamin berkontribusi +{p}% pada skor model. Fokus pada faktor gaya hidup yang bisa diubah.",
    'Age': "Faktor usia berkontribusi +{p}% terhadap risiko. Skrining berkala makin penting seiring bertambahnya usia.",
    'Education': "Faktor edukasi berkontribusi +{p}% pada skor model. Tingkatkan literasi kesehatan lewat edukasi gizi dan gaya hidup.",
    'Income': "Faktor pendapatan berkontribusi +{p}% pada skor model. Manfaatkan layanan kesehatan primer yang terjangkau.",
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


def _run_five_models(input_df: pd.DataFrame, input_scaled: np.ndarray) -> dict:
    """Inferensi 5 algoritma. Mengembalikan probabilitas tiap model."""
    prob_svm = float(models["svm"].predict_proba(input_scaled)[0][1])
    prob_dt = float(models["dt"].predict_proba(input_df)[0][1])
    prob_rf = float(models["rf"].predict_proba(input_df)[0][1])
    prob_xgb = float(models["xgb"].predict_proba(input_df)[0][1])
    input_cnn = input_scaled.reshape((1, input_scaled.shape[1], 1))
    prob_cnn = float(models["cnn1d"].predict(input_cnn, verbose=0)[0][0])
    return {
        "svm": prob_svm, "dt": prob_dt, "rf": prob_rf,
        "xgb": prob_xgb, "cnn": prob_cnn,
    }


def _predict_single(model_key: str, input_df: pd.DataFrame, input_scaled: np.ndarray) -> float:
    """Inferensi 1 model saja (dipakai /api/predict-best)."""
    if model_key in ("svm",):
        return float(models[model_key].predict_proba(input_scaled)[0][1])
    if model_key == "cnn1d":
        input_cnn = input_scaled.reshape((1, input_scaled.shape[1], 1))
        return float(models[model_key].predict(input_cnn, verbose=0)[0][0])
    # dt, rf, xgb: pakai data mentah (tanpa scaling)
    return float(models[model_key].predict_proba(input_df)[0][1])


def _require_best_model():
    _require_models()
    missing = sorted({"scaler", best_model_key} - models.keys())
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model terbaik ('{best_model_key}') belum termuat di server. Restart server dan cek folder models/."
        )


def _compute_shap(input_df: pd.DataFrame):
    """Hitung SHAP dengan API baru, fallback ke API lama, lalu ke kosong.

    Mengembalikan (feature_importance_sorted | None). Tidak pernah raise
    agar endpoint /api/predict tetap memberi diagnosis walau XAI gagal.
    """
    explainer = models.get("explainer")
    if explainer is None:
        logger.warning("SHAP explainer tidak dimuat, XAI dilewati.")
        return None
    try:
        # API baru (shap >= 0.41): explainer(X) -> Explanation
        explanation = explainer(input_df)
        values = np.asarray(explanation.values)
        shap_array = values[0] if values.ndim >= 2 else values
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
    for feat_name, s_val in zip(FEATURE_NAMES, shap_array):
        impact_pct = round((abs(float(s_val)) / total_shap) * 100, 1)
        feature_importance.append({
            "feature": feat_name,
            "shap_value": float(s_val),
            "impact_percent": impact_pct
        })
    feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return feature_importance


def _build_summary(all_probs: list[float]):
    avg_risk_probability = float(np.mean(all_probs))
    final_diagnosis = "BERISIKO DIABETES" if avg_risk_probability >= 0.5 else "TIDAK BERISIKO"
    return avg_risk_probability, final_diagnosis


# ----------------------------------------------------
# ENDPOINT UTAMA: PREDIKSI MULTI-MODEL & XAI
# ----------------------------------------------------
@app.post("/api/predict")
def predict_health_risk(payload: HealthScreeningInput):
    _require_models()

    # 1. Format Payload menjadi DataFrame (kolom sesuai urutan dataset)
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    # 2. Inferensi 5 Algoritma
    probs = _run_five_models(input_df, input_scaled)
    all_probs = [probs["svm"], probs["dt"], probs["rf"], probs["xgb"], probs["cnn"]]
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
            "consensus_voting": f"{sum(1 for p in all_probs if p >= 0.5)} dari 5 model mendeteksi risiko."
        },
        "model_predictions": [
            {"model_name": "Support Vector Machine (SVM)", "risk_probability": round(probs["svm"] * 100, 2), "prediction": "Berisiko" if probs["svm"] >= 0.5 else "Aman"},
            {"model_name": "Decision Tree", "risk_probability": round(probs["dt"] * 100, 2), "prediction": "Berisiko" if probs["dt"] >= 0.5 else "Aman"},
            {"model_name": "Random Forest", "risk_probability": round(probs["rf"] * 100, 2), "prediction": "Berisiko" if probs["rf"] >= 0.5 else "Aman"},
            {"model_name": "Gradient Boosting (XGBoost)", "risk_probability": round(probs["xgb"] * 100, 2), "prediction": "Berisiko" if probs["xgb"] >= 0.5 else "Aman"},
            {"model_name": "1D-CNN (Deep Learning)", "risk_probability": round(probs["cnn"] * 100, 2), "prediction": "Berisiko" if probs["cnn"] >= 0.5 else "Aman"}
        ],
        "explainable_ai": {
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences
        }
    }


@app.post("/api/predict-lite")
def predict_lite(payload: HealthScreeningInput):
    """Endpoint ringan untuk slider What-If Analysis: 5 model TANPA SHAP (<100ms)."""
    _require_models()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)
    probs = _run_five_models(input_df, input_scaled)
    all_probs = [probs["svm"], probs["dt"], probs["rf"], probs["xgb"], probs["cnn"]]
    avg_risk, diagnosis = _build_summary(all_probs)
    return {
        "status": "success",
        "summary": {
            "final_diagnosis": diagnosis,
            "overall_risk_score_percent": round(avg_risk * 100, 2),
            "consensus_voting": f"{sum(1 for p in all_probs if p >= 0.5)} dari 5 model mendeteksi risiko."
        },
        "model_predictions": [
            {"model_name": "Support Vector Machine (SVM)", "risk_probability": round(probs["svm"] * 100, 2)},
            {"model_name": "Decision Tree", "risk_probability": round(probs["dt"] * 100, 2)},
            {"model_name": "Random Forest", "risk_probability": round(probs["rf"] * 100, 2)},
            {"model_name": "Gradient Boosting (XGBoost)", "risk_probability": round(probs["xgb"] * 100, 2)},
            {"model_name": "1D-CNN (Deep Learning)", "risk_probability": round(probs["cnn"] * 100, 2)},
        ],
    }


@app.get("/api/metrics")
def get_metrics():
    """Menyajikan metrik evaluasi 5 model untuk dashboard FE + tabel jurnal."""
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
# ENDPOINT MODEL TERBAIK: 1 MODEL (F1 tertinggi) + SHAP
# Dipakai FE utama; /api/predict tetap untuk komparasi jurnal
# ----------------------------------------------------
@app.post("/api/predict-best")
def predict_best(payload: HealthScreeningInput):
    """Prediksi dengan satu model terbaik (dipilih train.py via F1-Score).

    Cocok untuk halaman utama FE. SHAP dihitung dari explainer Random Forest
    (wakil analisis fitur), dinyatakan eksplisit di field xai_basis_model.
    """
    _require_best_model()
    input_dict = payload.model_dump()
    input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
    input_scaled = models["scaler"].transform(input_df)

    prob = _predict_single(best_model_key, input_df, input_scaled)
    diagnosis = "BERISIKO DIABETES" if prob >= 0.5 else "TIDAK BERISIKO"

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
            "prediction": "Berisiko" if prob >= 0.5 else "Aman",
        },
        "summary": {
            "final_diagnosis": diagnosis,
            "risk_score_percent": round(prob * 100, 2),
        },
        "explainable_ai": {
            "xai_basis_model": METRIC_TO_DISPLAY.get("rf", "Random Forest"),
            "top_contributing_features": feature_importance[:5],
            "narrative_recommendations": xai_sentences,
        },
    }
    if best_model_metrics:
        response["best_model"]["f1_score"] = best_model_metrics.get("f1_score")
        response["best_model"]["accuracy"] = best_model_metrics.get("accuracy")
    return response


@app.get("/health")
def health_check():
    loaded = sorted(models.keys())
    return {
        "status": "ok",
        "message": "Backend FastAPI ML Aegis XAI berjalan dengan normal.",
        "loaded_models": loaded,
        "best_model": {
            "key": best_model_key,
            "display_name": best_model_display_name,
            "f1_score": (best_model_metrics or {}).get("f1_score"),
        },
    }
