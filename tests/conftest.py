"""Shared fixtures for all tests."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_validation import FEATURES  # noqa: E402
from src.predict import load_artifact  # noqa: E402


@pytest.fixture(scope="session")
def sample_df():
    return pd.read_csv(ROOT / "tests" / "sample_data.csv")


@pytest.fixture(scope="session")
def artifact():
    return load_artifact()


@pytest.fixture
def valid_row(sample_df):
    return sample_df.iloc[0][FEATURES].astype(float).to_dict()
