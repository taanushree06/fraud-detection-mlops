"""Reusable data validation. Raises an error if critical checks fail."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_PATH = ROOT / "reports" / "validation_report.json"

TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_COLUMNS = FEATURES + [TARGET]


class DataValidationError(Exception):
    """Raised when a critical validation check fails."""


def validate(df, raise_on_error=True):
    errors, warnings = [], []

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return _finish(errors, warnings, {}, raise_on_error)

    # 2. Data types (everything must be numeric)
    non_numeric = [c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")
        return _finish(errors, warnings, {}, raise_on_error)

    # 3. Missing values
    n_missing = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    if n_missing > 0:
        errors.append(f"Found {n_missing} missing values")

    # 4. Invalid values
    values = df[REQUIRED_COLUMNS].to_numpy(dtype=float)
    n_inf = int(np.isinf(values).sum())
    if n_inf > 0:
        errors.append(f"Found {n_inf} infinite values")
    if (df["Amount"] < 0).any():
        errors.append("Negative values found in Amount")
    if (df["Time"] < 0).any():
        errors.append("Negative values found in Time")

    # 5. Target values
    bad_targets = sorted(set(df[TARGET].dropna().unique()) - {0, 1})
    if bad_targets:
        errors.append(f"Target has unexpected values: {bad_targets}")

    # 6. Class distribution
    counts = df[TARGET].value_counts()
    n_legit, n_fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    if n_legit == 0 or n_fraud == 0:
        errors.append("Both classes (0 and 1) must be present")
    fraud_pct = round(100 * n_fraud / len(df), 4) if len(df) else 0

    # 7. Duplicates (warning only)
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        warnings.append(f"{n_dup} duplicate rows (will be dropped before splitting)")

    stats = {
        "rows": int(len(df)),
        "legitimate": n_legit,
        "fraud": n_fraud,
        "fraud_percentage": fraud_pct,
        "duplicates": n_dup,
    }
    return _finish(errors, warnings, stats, raise_on_error)


def _finish(errors, warnings, stats, raise_on_error):
    report = {"passed": len(errors) == 0, "errors": errors, "warnings": warnings, "stats": stats}
    if errors and raise_on_error:
        raise DataValidationError("; ".join(errors))
    return report


def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Run: python src\\download_data.py")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    report = validate(df, raise_on_error=False)

    print("=" * 50)
    print("DATA VALIDATION:", "PASSED" if report["passed"] else "FAILED")
    print("=" * 50)
    for k, v in report["stats"].items():
        print(f"{k:18}: {v}")
    for w in report["warnings"]:
        print("WARNING:", w)
    for e in report["errors"]:
        print("ERROR  :", e)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
