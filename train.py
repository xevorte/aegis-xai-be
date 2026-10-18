import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import shap

np.random.seed(42)

# 1. Buat folder penyimpan model
os.makedirs("models", exist_ok=True)

# 2. Load Dataset (Ganti path sesuai file CSV Kaggle kamu)
print("📥 Memuat dataset...")
# Dataset: Diabetes Health Indicators Dataset (253,680 data)
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"].astype(int)  # CSV float -> int utk bincount & biner

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scaling Data (Sangat penting untuk SVM, KNN, & 1D-CNN)
print("⚙️ Melakukan Preprocessing & Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

metrics = {}

def _score_block(name, y_true, y_pred):
    """Hitung accuracy/precision/recall/F1 + confusion matrix untuk 1 model."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "confusion_labels": ["TN", "FP", "FN", "TP"],
    }
    print(f"📊 {name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

# ----------------------------------------------------
# A. Model 1: Support Vector Machine (SVM)
# ----------------------------------------------------
print("🤖 Training Model 1: Support Vector Machine (SVM)...")

# Sub-sampling 15.000 data khusus untuk SVM agar training cepat (< 10 detik)
idx_sub = np.random.choice(X_train_scaled.shape[0], size=15000, replace=False)
X_train_svm = X_train_scaled[idx_sub]
y_train_svm = y_train.iloc[idx_sub]

svm_model = SVC(
    probability=True,
    kernel='rbf',
    C=1.0,
    max_iter=3000,
    verbose=False,
    random_state=42,
    class_weight='balanced',  # handling imbalance minimal untuk jurnal
)
svm_model.fit(X_train_svm, y_train_svm)
joblib.dump(svm_model, "models/model_svm.pkl")
_score_block("svm", y_test, svm_model.predict(X_test_scaled))

# ----------------------------------------------------
# B. Model 2: Decision Tree
# ----------------------------------------------------
print("🤖 Training Model 2: Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "models/model_dt.pkl")
_score_block("decision_tree", y_test, dt_model.predict(X_test))

# ----------------------------------------------------
# C. Model 3: Random Forest
# ----------------------------------------------------
print("🤖 Training Model 3: Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "models/model_rf.pkl")
_score_block("random_forest", y_test, rf_model.predict(X_test))

# ----------------------------------------------------
# D. Model 4: Gradient Boosting (XGBoost)
# ----------------------------------------------------
print("🤖 Training Model 4: XGBoost...")
# scale_pos_weight menangani imbalance: neg/pos
neg, pos = np.bincount(y_train.to_numpy().astype(int))
scale_pos_weight = float(neg / max(pos, 1))
xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='logloss', n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "models/model_xgb.pkl")
_score_block("xgboost", y_test, xgb_model.predict(X_test))

# ----------------------------------------------------
# E. Model 5: 1D Convolutional Neural Network (1D-CNN)
# ----------------------------------------------------
print("🤖 Training Model 5: 1D-CNN (Data Tabular)...")
# Reshape data tabular dari (sampel, fitur) -> (sampel, fitur, 1)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# class_weight untuk imbalance pada deep learning
neg_t, pos_t = np.bincount(y_train.to_numpy().astype(int))
cnn_class_weight = {0: 1.0, 1: float(neg_t / max(pos_t, 1))}

cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=256, validation_split=0.1, verbose=1, class_weight=cnn_class_weight)
cnn_model.save("models/model_cnn1d.h5")
y_pred_cnn = (cnn_model.predict(X_test_cnn, verbose=0).ravel() >= 0.5).astype(int)
_score_block("cnn1d", y_test, y_pred_cnn)

# ----------------------------------------------------
# G. TENTUKAN MODEL TERBAIK (untuk endpoint /api/predict-best)
# Dipilih berdasarkan F1-Score tertinggi (paling adil utk data imbalance)
# ----------------------------------------------------
_METRIC_TO_RUNTIME_KEY = {
    "svm": "svm",
    "decision_tree": "dt",
    "random_forest": "rf",
    "xgboost": "xgb",
    "cnn1d": "cnn1d",
}
_METRIC_TO_DISPLAY = {
    "svm": "Support Vector Machine (SVM)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "Gradient Boosting (XGBoost)",
    "cnn1d": "1D-CNN (Deep Learning)",
}
best_metric_name = max(metrics, key=lambda k: metrics[k]["f1_score"])
best_meta = metrics[best_metric_name]
best_model_entry = {
    "metric_name": best_metric_name,
    "key": _METRIC_TO_RUNTIME_KEY[best_metric_name],
    "display_name": _METRIC_TO_DISPLAY[best_metric_name],
    "f1_score": best_meta["f1_score"],
    "accuracy": best_meta["accuracy"],
    "selection_metric": "f1_score",
}
print(f"🏆 Model terbaik (F1): {best_model_entry['display_name']} (F1={best_meta['f1_score']})")

# Simpan metrik untuk dashboard FE (GET /api/metrics) + tabel jurnal
with open("models/metrics.json", "w") as f:
    json.dump({
        "models": metrics,
        "imbalance_handling": "class_weight balanced / scale_pos_weight",
        "best_model": best_model_entry,
    }, f, indent=2)
print("💾 metrics.json tersimpan.")

# ----------------------------------------------------
# F. Inisialisasi & Simpan SHAP Explainer (XAI)
# ----------------------------------------------------
print("💡 Membangun SHAP Explainer untuk Explainable AI...")
# Menggunakan TreeExplainer pada Random Forest sebagai wakil analisis fitur
explainer = shap.TreeExplainer(rf_model)
joblib.dump(explainer, "models/shap_explainer.pkl")

print("✅ SELESEAI! Seluruh model dan komponen XAI telah disimpan di folder /models.")
