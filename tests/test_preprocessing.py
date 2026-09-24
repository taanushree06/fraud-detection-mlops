"""Tests for the preprocessing pipeline and the data splits."""
import numpy as np

from src.data_validation import FEATURES
from src.preprocessing import SCALE_COLS, build_preprocessor, get_splits


def test_output_shape(sample_df):
    X = sample_df[FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert out.shape == (len(X), 30)


def test_output_has_same_feature_names(sample_df):
    out = build_preprocessor().fit_transform(sample_df[FEATURES])
    assert set(out.columns) == set(FEATURES)


def test_time_and_amount_are_scaled(sample_df):
    out = build_preprocessor().fit_transform(sample_df[FEATURES])
    for col in SCALE_COLS:
        assert abs(out[col].mean()) < 1e-6
        assert abs(out[col].std(ddof=0) - 1.0) < 1e-6


def test_v_features_are_unchanged(sample_df):
    X = sample_df[FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert np.allclose(out["V14"].to_numpy(), X["V14"].to_numpy())


def test_transform_uses_training_statistics_only(sample_df):
    train = sample_df[FEATURES].iloc[:300]
    new = sample_df[FEATURES].iloc[300:310]
    pre = build_preprocessor().fit(train)
    scaler = pre.named_transformers_["scale"]
    i = SCALE_COLS.index("Amount")
    expected = (new["Amount"].iloc[0] - scaler.mean_[i]) / scaler.scale_[i]
    out = pre.transform(new)
    assert np.isclose(out["Amount"].iloc[0], expected)


def test_splits_do_not_overlap_and_keep_both_classes(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(sample_df)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    for y in (y_train, y_val, y_test):
        assert set(y.unique()) == {0, 1}
    assert len(X_train) + len(X_val) + len(X_test) == len(sample_df.drop_duplicates())
