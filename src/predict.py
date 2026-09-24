"""Prediction logic shared by the app and the tests."""
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES  # noqa: E402

ARTIFACT_PATH = ROOT / "models" / "fraud_pipeline.joblib"


def load_artifact(path=ARTIFACT_PATH):
    return joblib.load(path)


def validate_input(values):
    """Return a clean {feature: float} dict, or raise ValueError with a clear message."""
    missing = [f for f in FEATURES if f not in values or values[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing values for: {missing}")
    row = {}
    for f in FEATURES:
        try:
            v = float(values[f])
        except (TypeError, ValueError):
            raise ValueError(f"'{f}' must be a number, got {values[f]!r}")
        if not math.isfinite(v):
            raise ValueError(f"'{f}' must be a finite number")
        row[f] = v
    if row["Amount"] < 0:
        raise ValueError("Amount cannot be negative")
    if row["Time"] < 0:
        raise ValueError("Time cannot be negative")
    return row


def predict_one(artifact, values):
    """Predict one transaction using the saved pipeline and saved threshold."""
    row = validate_input(values)
    X = pd.DataFrame([row], columns=FEATURES)
    proba = float(artifact["pipeline"].predict_proba(X)[0, 1])
    threshold = float(artifact["threshold"])
    is_fraud = proba >= threshold
    return {
        "prediction": "FRAUDULENT TRANSACTION" if is_fraud else "LEGITIMATE TRANSACTION",
        "is_fraud": bool(is_fraud),
        "fraud_probability": proba,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import json

    art = load_artifact()
    samples = json.loads((ROOT / "models" / "sample_inputs.json").read_text())
    for label, values in samples.items():
        r = predict_one(art, values)
        print(f"{label:14}: {r['prediction']} | fraud probability {r['fraud_probability'] * 100:.1f}% "
              f"| threshold {r['threshold']}")
