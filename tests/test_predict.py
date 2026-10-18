"""Smoke test API: 1 payload valid -> 200, 1 payload invalid (BMI) -> 422."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 28.5, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 2, "PhysHlth": 1, "DiffWalk": 0, "Sex": 1,
    "Age": 9, "Education": 4, "Income": 6,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_lite_validation_ok_shape():
    # Tanpa model termuat akan 500 (ekspektasi di CI tanpa artefak),
    # tapi validasi Pydantic harus lolos dulu (bukan 422).
    r = client.post("/api/predict-lite", json=VALID_PAYLOAD)
    assert r.status_code in (200, 500)


def test_predict_invalid_bmi_rejected():
    bad = dict(VALID_PAYLOAD, BMI=-5)
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422


def test_predict_invalid_age_rejected():
    bad = dict(VALID_PAYLOAD, Age=99)
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422
