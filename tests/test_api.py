import numpy as np
from fastapi.testclient import TestClient

from api import main


class FixedJamModel:
    def predict(self, features):
        assert "motor_load_ratio" in features.columns
        assert "belt_wear_pressure_index" in features.columns
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.15, 0.85]])


def _request_body():
    return {
        "features": {
            "motor_current_a": 18.0,
            "rated_motor_current_a": 20.0,
            "vibration_mm_s": 7.5,
            "belt_age_years": 8.0,
            "parcels_per_minute": 110.0,
            "dust_accumulation_index": 72.0,
            "days_since_maintenance": 95.0,
            "humidity_pct": 68.0,
        }
    }


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(main, "model", FixedJamModel())
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_jam_risk_and_probability(monkeypatch):
    monkeypatch.setattr(main, "model", FixedJamModel())
    monkeypatch.setattr(main, "metadata", {"model_name": "test-model"})
    response = TestClient(main.app).post("/predict", json=_request_body())
    assert response.status_code == 200
    assert response.json() == {
        "prediction": "Jam Risk",
        "jam_probability": 0.85,
        "model": "test-model",
    }


def test_predict_rejects_invalid_sensor_input():
    body = _request_body()
    body["features"]["humidity_pct"] = 101
    response = TestClient(main.app).post("/predict", json=body)
    assert response.status_code == 422
