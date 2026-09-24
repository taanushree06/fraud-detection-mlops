"""Preprocessing: load data, clean, split, and build the scaler pipeline."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES, TARGET, validate  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
RANDOM_STATE = 42
SCALE_COLS = ["Time", "Amount"]
PASS_COLS = [c for c in FEATURES if c not in SCALE_COLS]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_preprocessor():
    """Scale Time and Amount; keep V1-V28 unchanged."""
    pre = ColumnTransformer(
        [("scale", StandardScaler(), SCALE_COLS), ("keep", "passthrough", PASS_COLS)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def get_splits(df, drop_duplicates=True):
    """Return stratified 60/20/20 train, validation, test splits."""
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    df = load_data()
    validate(df)  # stops here if the data is bad
    print("Validation passed.")

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df)
    print("-" * 55)
    for name, X, y in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"{name:11}: {len(X):>7,} rows | fraud = {int(y.sum()):>3} ({100 * y.mean():.3f}%)")

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)  # fit on TRAIN only
    Xt_val = pre.transform(X_val)          # only transform
    print("-" * 55)
    print("Output shape (train):", Xt_train.shape)
    print("Amount after scaling, train: mean=%.3f std=%.3f" % (Xt_train["Amount"].mean(), Xt_train["Amount"].std()))
    print("Amount after scaling, val  : mean=%.3f std=%.3f" % (Xt_val["Amount"].mean(), Xt_val["Amount"].std()))
    print("Same columns in train and val:", list(Xt_train.columns) == list(Xt_val.columns))


if __name__ == "__main__":
    main()
