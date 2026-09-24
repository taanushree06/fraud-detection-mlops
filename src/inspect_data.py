"""Inspect the raw dataset and save a summary."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_PATH = ROOT / "reports" / "data_summary.json"
TARGET = "Class"


def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Run: python src\\download_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        print(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")
        sys.exit(1)

    features = [c for c in df.columns if c != TARGET]
    counts = df[TARGET].value_counts()
    n_legit = int(counts.get(0, 0))
    n_fraud = int(counts.get(1, 0))

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET,
        "feature_columns": features,
        "legitimate_count": n_legit,
        "fraud_count": n_fraud,
        "fraud_percentage": round(100 * n_fraud / len(df), 4),
        "imbalance_ratio": round(n_legit / n_fraud, 1) if n_fraud else None,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    print("=" * 55)
    print("DATASET INSPECTION")
    print("=" * 55)
    print(f"Rows            : {summary['rows']:,}")
    print(f"Columns         : {summary['columns']}")
    print(f"Target column   : {TARGET}")
    print(f"Features ({len(features)})   : {features}")
    print("-" * 55)
    print(f"Legitimate (0)  : {n_legit:,}")
    print(f"Fraud (1)       : {n_fraud:,}")
    print(f"Fraud %         : {summary['fraud_percentage']}%")
    print(f"Imbalance ratio : about {summary['imbalance_ratio']} legit per 1 fraud")
    print("-" * 55)
    print(f"Missing values  : {summary['missing_values']}")
    print(f"Duplicate rows  : {summary['duplicate_rows']}")
    print("=" * 55)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
