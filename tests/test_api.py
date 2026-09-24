"""Tests for the FastAPI service."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


@pytest.fixture(autouse=True)
def no_log_writes(monkeypatch):
    monkeypatch.setattr("api.main.log_prediction", lambda result: None)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_fraud_and_legit_examples():
    samples = json.loads((ROOT / "models" / "sample_inputs.json").read_text())
    fraud = client.post("/predict", json={"features": samples["fraud_example"]}).json()
    legit = client.post("/predict", json={"features": samples["legit_example"]}).json()
    assert fraud["is_fraud"] is True
    assert legit["is_fraud"] is False
    assert 0.0 <= legit["fraud_probability"] <= 1.0


def test_missing_field_returns_400(valid_row):
    del valid_row["V5"]
    r = client.post("/predict", json={"features": valid_row})
    assert r.status_code == 400


def test_non_numeric_value_returns_422(valid_row):
    valid_row["V2"] = "abc"
    r = client.post("/predict", json={"features": valid_row})
    assert r.status_code == 422
