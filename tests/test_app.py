"""Tests for input validation in the app and for the monitoring log."""
import pytest

from src.monitoring import log_prediction, read_log, summarize
from src.predict import predict_one, validate_input


def test_valid_input_produces_a_prediction(artifact, valid_row):
    result = predict_one(artifact, valid_row)
    assert "prediction" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_missing_field_is_rejected(artifact, valid_row):
    del valid_row["V5"]
    with pytest.raises(ValueError, match="Missing"):
        predict_one(artifact, valid_row)


def test_empty_value_is_rejected(valid_row):
    valid_row["V3"] = ""
    with pytest.raises(ValueError):
        validate_input(valid_row)


def test_non_numeric_value_is_rejected(valid_row):
    valid_row["V2"] = "abc"
    with pytest.raises(ValueError, match="number"):
        validate_input(valid_row)


def test_nan_and_infinite_values_are_rejected(valid_row):
    valid_row["V7"] = float("nan")
    with pytest.raises(ValueError):
        validate_input(valid_row)
    valid_row["V7"] = float("inf")
    with pytest.raises(ValueError):
        validate_input(valid_row)


def test_negative_amount_is_rejected(valid_row):
    valid_row["Amount"] = -10
    with pytest.raises(ValueError, match="Amount"):
        validate_input(valid_row)


def test_negative_time_is_rejected(valid_row):
    valid_row["Time"] = -1
    with pytest.raises(ValueError, match="Time"):
        validate_input(valid_row)


def test_monitoring_log_and_summary(tmp_path):
    path = tmp_path / "predictions.csv"
    log_prediction({"prediction": "FRAUDULENT TRANSACTION", "fraud_probability": 0.9, "threshold": 0.3}, path)
    log_prediction({"prediction": "LEGITIMATE TRANSACTION", "fraud_probability": 0.1, "threshold": 0.3}, path)
    log_prediction({"prediction": "FRAUDULENT TRANSACTION", "fraud_probability": 0.7, "threshold": 0.3}, path)
    stats = summarize(read_log(path))
    assert stats["total"] == 3
    assert stats["fraud"] == 2
    assert stats["legit"] == 1
    assert abs(stats["avg_probability"] - (0.9 + 0.1 + 0.7) / 3) < 1e-6


def test_empty_log_gives_zero_counts(tmp_path):
    stats = summarize(read_log(tmp_path / "does_not_exist.csv"))
    assert stats["total"] == 0
    assert stats["fraud"] == 0
