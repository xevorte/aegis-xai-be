"""Train and evaluate the Aegis diabetes screening models.

The protocol keeps a locked 70/15/15 train/validation/test split. SMOTENC is
applied only inside the training folds through an imbalanced-learn pipeline.
The validation split selects thresholds and the final model; the test split is
read only for the final report.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
CATEGORICAL_INDICES = [
    index for index, name in enumerate(FEATURE_NAMES)
    if name not in {"BMI", "MentHlth", "PhysHlth"}
]
SMOTE_SAMPLING_STRATEGY = 0.5
CV_SPLITS = 5


def expected_calibration_error(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Return expected calibration error using equal-width probability bins."""
    y_true_array = np.asarray(y_true, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability_array >= lower) & (
            probability_array < upper if index < bins - 1 else probability_array <= upper
        )
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(y_true_array[mask].mean()) - float(probability_array[mask].mean())
        )
    return round(ece, 4)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a validation threshold by maximising F1 without reading test data."""
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return round(float(thresholds[int(np.argmax(scores))]), 2)


def score_block(name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    try:
        roc_auc = round(float(roc_auc_score(y_true, probabilities)), 4)
        pr_auc = round(float(average_precision_score(y_true, probabilities)), 4)
    except ValueError:
        roc_auc = None
        pr_auc = None
    block = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "ece": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "confusion_labels": ["TN", "FP", "FN", "TP"],
        "decision_threshold": threshold,
    }
    print(
        f"📊 {name}: PR-AUC={block['pr_auc']} ROC-AUC={block['roc_auc']} "
        f"recall={block['recall']} F1={block['f1_score']} ECE={block['ece']}"
    )
    return block


def smote_pipeline(model):
    return ImbPipeline(
        steps=[
            (
                "smote",
                SMOTENC(
                    categorical_features=CATEGORICAL_INDICES,
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("model", model),
        ]
    )


def tune_model(name: str, estimator, parameter_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Tune one tabular model with SMOTENC inside every CV training fold."""
    search = GridSearchCV(
        estimator=smote_pipeline(estimator),
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    print(f"🔎 Tuning {name} dengan Stratified {CV_SPLITS}-Fold CV + SMOTENC...")
    search.fit(X_train, y_train)
    print(f"✅ {name}: best PR-AUC CV={search.best_score_:.4f}, params={search.best_params_}")
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_pr_auc": round(float(search.best_score_), 4),
        "cv_folds": CV_SPLITS,
        "sampling_strategy": SMOTE_SAMPLING_STRATEGY,
    }


def build_cnn(input_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_features, 1)),
            Conv1D(filters=32, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=2, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


print("📥 Memuat dataset...")
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_NAMES].copy()
y = df["Diabetes_binary"].astype(int)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout, y_holdout, test_size=0.50, random_state=RANDOM_STATE, stratify=y_holdout
)
print(f"📐 Split: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

rf_model, rf_cv = tune_model(
    "Random Forest",
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight=None),
    {"model__n_estimators": [100], "model__max_depth": [10, 14]},
    X_train,
    y_train,
)
xgb_model, xgb_cv = tune_model(
    "XGBoost",
    XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1, tree_method="hist"),
    {
        "model__n_estimators": [100],
        "model__max_depth": [4, 6],
        "model__learning_rate": [0.1],
    },
    X_train,
    y_train,
)

print("🎯 Kalibrasi probabilitas RF dan XGBoost...")
rf_calibrated = CalibratedClassifierCV(rf_model, method="sigmoid", cv=3, n_jobs=-1)
xgb_calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3, n_jobs=-1)
rf_calibrated.fit(X_train, y_train)
xgb_calibrated.fit(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("🤖 Training 1D-CNN dengan SMOTENC pada data training...")
smote_for_cnn = SMOTENC(
    categorical_features=CATEGORICAL_INDICES,
    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
    random_state=RANDOM_STATE,
)
X_cnn_train, y_cnn_train = smote_for_cnn.fit_resample(X_train, y_train)
X_train_cnn = scaler.transform(X_cnn_train).reshape((-1, len(FEATURE_NAMES), 1))
X_val_cnn = scaler.transform(X_val).reshape((-1, len(FEATURE_NAMES), 1))
X_test_cnn = scaler.transform(X_test).reshape((-1, len(FEATURE_NAMES), 1))
cnn_model = build_cnn(len(FEATURE_NAMES))
cnn_model.fit(
    X_train_cnn,
    y_cnn_train,
    validation_data=(X_val_cnn, y_val),
    epochs=10,
    batch_size=256,
    callbacks=[EarlyStopping(monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True)],
    verbose=1,
)
cnn_model.save(MODELS_DIR / "model_cnn1d.h5")

probabilities = {
    "rf": (rf_calibrated.predict_proba(X_val)[:, 1], rf_calibrated.predict_proba(X_test)[:, 1]),
    "xgb": (xgb_calibrated.predict_proba(X_val)[:, 1], xgb_calibrated.predict_proba(X_test)[:, 1]),
    "cnn1d": (
        cnn_model.predict(X_val_cnn, verbose=0).ravel(),
        cnn_model.predict(X_test_cnn, verbose=0).ravel(),
    ),
}
metrics = {}
for key, (val_probability, test_probability) in probabilities.items():
    threshold = choose_threshold(y_val, val_probability)
    metrics[key] = {
        "validation": score_block(f"{key}:validation", y_val, val_probability, threshold),
        "test": score_block(f"{key}:test", y_test, test_probability, threshold),
    }
metrics["rf"]["cv"] = rf_cv
metrics["xgb"]["cv"] = xgb_cv

ensemble_val = np.column_stack([probabilities[key][0] for key in ("rf", "xgb", "cnn1d")]).mean(axis=1)
ensemble_test = np.column_stack([probabilities[key][1] for key in ("rf", "xgb", "cnn1d")]).mean(axis=1)
ensemble_threshold = choose_threshold(y_val, ensemble_val)
metrics["ensemble_average"] = {
    "validation": score_block("ensemble_average:validation", y_val, ensemble_val, ensemble_threshold),
    "test": score_block("ensemble_average:test", y_test, ensemble_test, ensemble_threshold),
}

best_model_key = max(("rf", "xgb", "cnn1d"), key=lambda key: metrics[key]["validation"]["pr_auc"] or -1.0)
best_display_name = {
    "rf": "Random Forest",
    "xgb": "Gradient Boosting (XGBoost)",
    "cnn1d": "1D-CNN (Deep Learning)",
}[best_model_key]
best = metrics[best_model_key]

joblib.dump(rf_calibrated, MODELS_DIR / "model_rf.pkl")
joblib.dump(xgb_calibrated, MODELS_DIR / "model_xgb.pkl")
rf_tree = rf_model.named_steps["model"]
joblib.dump(shap.TreeExplainer(rf_tree), MODELS_DIR / "shap_explainer.pkl")

metrics_payload = {
    "protocol": {
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "cv": f"Stratified {CV_SPLITS}-Fold",
        "smote": "SMOTENC inside each training fold; sampling_strategy=0.5",
        "selection_metric": "validation_pr_auc",
        "threshold_selection": "validation_f1_maximisation",
        "calibration": "sigmoid CalibratedClassifierCV for RF and XGBoost",
    },
    "models": metrics,
    "best_model": {
        "metric_name": best_model_key,
        "key": best_model_key,
        "display_name": best_display_name,
        "validation_pr_auc": best["validation"]["pr_auc"],
        "validation_recall": best["validation"]["recall"],
        "test_pr_auc": best["test"]["pr_auc"],
        "test_recall": best["test"]["recall"],
        "test_f1_score": best["test"]["f1_score"],
        "test_ece": best["test"]["ece"],
        "selection_metric": "validation_pr_auc",
    },
    "scope": "Core diabetes experiment: Random Forest, XGBoost, and 1D-CNN.",
}
with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics_payload, handle, indent=2)

print(f"🏆 Best model (validation PR-AUC): {best_display_name}")
print("✅ Training selesai. Model inti, metrik, kalibrasi, dan SHAP tersimpan di models/.")
