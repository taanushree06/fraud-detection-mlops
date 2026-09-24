"""Tests for the dataset and the data-validation step."""
import numpy as np
import pytest

from src.data_validation import REQUIRED_COLUMNS, TARGET, DataValidationError, validate


def test_dataset_loads(sample_df):
    assert len(sample_df) > 0


def test_expected_columns_exist(sample_df):
    for col in REQUIRED_COLUMNS:
        assert col in sample_df.columns, f"missing column {col}"


def test_has_30_features_and_target(sample_df):
    assert len(REQUIRED_COLUMNS) == 31
    assert TARGET == "Class"


def test_validation_passes_on_good_data(sample_df):
    report = validate(sample_df)
    assert report["passed"] is True
    assert report["stats"]["fraud"] > 0


def test_validation_fails_on_missing_column(sample_df):
    with pytest.raises(DataValidationError):
        validate(sample_df.drop(columns=["V1"]))


def test_validation_fails_on_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, "V1"] = np.nan
    with pytest.raises(DataValidationError):
        validate(df)


def test_validation_fails_on_negative_amount(sample_df):
    df = sample_df.copy()
    df.loc[0, "Amount"] = -5.0
    with pytest.raises(DataValidationError):
        validate(df)


def test_validation_fails_on_bad_target(sample_df):
    df = sample_df.copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError):
        validate(df)


def test_validation_fails_when_one_class_missing(sample_df):
    only_legit = sample_df[sample_df["Class"] == 0]
    with pytest.raises(DataValidationError):
        validate(only_legit)


def test_validation_can_report_without_raising(sample_df):
    report = validate(sample_df.drop(columns=["V1"]), raise_on_error=False)
    assert report["passed"] is False
    assert report["errors"]
