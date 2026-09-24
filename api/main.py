"""FastAPI service for fraud prediction. Reuses the saved pipeline and threshold."""
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitoring import log_prediction, read_log, summarize  # noqa: E402
from src.predict import load_artifact, predict_one  # noqa: E402

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")
_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = load_artifact()
    return _artifact


class Transaction(BaseModel):
    features: Dict[str, float]  # Time, V1..V28, Amount


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    a = get_artifact()
    return {
        "model": a["model_name"],
        "threshold": a["threshold"],
        "trained_at": a["trained_at"],
        "test_metrics": a["test_metrics"],
        "features": a["features"],
    }


@app.post("/predict")
def predict(tx: Transaction):
    try:
        result = predict_one(get_artifact(), tx.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    log_prediction(result)
    return result


@app.get("/stats")
def stats():
    return summarize(read_log())
