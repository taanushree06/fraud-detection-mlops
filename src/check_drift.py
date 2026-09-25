"""Simple data drift check: compare new data's feature averages to training data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES  # noqa: E402
from src.preprocessing import get_splits, load_data  # noqa: E402

THRESHOLD_STD = 0.5  # flag a feature if its mean shifts by more than half a training std


def check_drift(reference_df, new_df, threshold_std=THRESHOLD_STD):
    ref_mean, ref_std = reference_df[FEATURES].mean(), reference_df[FEATURES].std()
    new_mean = new_df[FEATURES].mean()
    shift = (new_mean - ref_mean).abs() / ref_std.replace(0, 1)
    flagged = shift[shift > threshold_std].sort_values(ascending=False)
    return flagged, shift


def main():
    df = load_data()
    X_train, _, X_test, *_ = get_splits(df)
    flagged, shift = check_drift(X_train, X_test)

    print("=" * 50)
    print("DATA DRIFT CHECK (test set vs training set)")
    print("=" * 50)
    if len(flagged) == 0:
        print("No significant drift detected.")
    else:
        print(f"{len(flagged)} feature(s) shifted by more than {THRESHOLD_STD} std:")
        print(flagged.round(3).to_string())
    print("-" * 50)
    print("Largest shifts overall:")
    print(shift.sort_values(ascending=False).head(5).round(3).to_string())


if __name__ == "__main__":
    main()
