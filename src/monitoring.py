"""Basic monitoring: log every prediction to a CSV and summarize it."""
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "predictions.csv"
COLUMNS = ["timestamp", "prediction", "fraud_probability", "threshold"]


def log_prediction(result, path=LOG_PATH):
    """Append one prediction result to the log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            result["prediction"],
            round(result["fraud_probability"], 6),
            result["threshold"],
        ])


def read_log(path=LOG_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def summarize(df):
    """Return counts and averages for the monitoring dashboard."""
    total = len(df)
    fraud = int((df["prediction"] == "FRAUDULENT TRANSACTION").sum()) if total else 0
    return {
        "total": total,
        "fraud": fraud,
        "legit": total - fraud,
        "fraud_rate": (fraud / total) if total else 0.0,
        "avg_probability": float(df["fraud_probability"].mean()) if total else 0.0,
    }


def clear_log(path=LOG_PATH):
    path = Path(path)
    if path.exists():
        path.unlink()
