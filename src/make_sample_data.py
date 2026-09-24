"""Create a small sample dataset (500 rows) for tests and CI."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "creditcard.csv"
OUT = ROOT / "tests" / "sample_data.csv"

df = pd.read_csv(SRC)
fraud = df[df["Class"] == 1].sample(40, random_state=42)
legit = df[df["Class"] == 0].sample(460, random_state=42)
sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv(OUT, index=False)
print(f"Saved {OUT} ({len(sample)} rows, {int(sample['Class'].sum())} fraud)")
