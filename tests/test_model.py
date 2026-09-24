"""Tests for the saved model artifact and predictions."""
import json
from pathlib import Path

from src.data_validation import FEATURES
from src.predict import predict_one

ROOT = Path(__file__).resolve().parents[1]


def test_artifact_loads_with_all_parts(artifact):
    for key in ["pipeline", "threshold", "model_name", "features"]:
        assert key in artifact


def test_threshold_is_valid(artifact):
    assert 0.0 < artifact["threshold"] < 1.0


def test_pipeline_includes_preprocessing(artifact):
    assert "preprocess" in artifact["pipeline"].named_steps


def test_probabilities_are_between_0_and_1(artifact, sample_df):
    proba = artifact["pipeline"].predict_proba(sample_df[FEATURES])
    assert proba.shape == (len(sample_df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_one_returns_expected_fields(artifact, valid_row):
    result = predict_one(artifact, valid_row)
    assert result["prediction"] in ("FRAUDULENT TRANSACTION", "LEGITIMATE TRANSACTION")
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["threshold"] == artifact["threshold"]


def test_decision_follows_threshold(artifact, sample_df):
    for i in range(25):
        row = sample_df.iloc[i][FEATURES].astype(float).to_dict()
        r = predict_one(artifact, row)
        assert r["is_fraud"] == (r["fraud_probability"] >= artifact["threshold"])


def test_saved_examples_are_classified_correctly(artifact):
    samples = json.loads((ROOT / "models" / "sample_inputs.json").read_text())
    assert predict_one(artifact, samples["fraud_example"])["is_fraud"] is True
    assert predict_one(artifact, samples["legit_example"])["is_fraud"] is False


def test_sanity_on_sample_data(artifact, sample_df):
    proba = artifact["pipeline"].predict_proba(sample_df[FEATURES])[:, 1]
    flagged = proba >= artifact["threshold"]
    is_fraud = sample_df["Class"].to_numpy() == 1
    recall = flagged[is_fraud].mean()
    false_alarm_rate = flagged[~is_fraud].mean()
    assert recall > 0.6
    assert false_alarm_rate < 0.05
