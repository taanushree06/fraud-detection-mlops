"""Write requirements.txt (app + tests) and requirements-train.txt (adds training tools)."""
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
core = ["pandas", "numpy", "scikit-learn", "xgboost", "joblib", "streamlit", "pytest"]
train = ["matplotlib", "mlflow"]

(ROOT / "requirements.txt").write_text("\n".join(f"{p}=={version(p)}" for p in core) + "\n")
(ROOT / "requirements-train.txt").write_text(
    "-r requirements.txt\n" + "\n".join(f"{p}=={version(p)}" for p in train) + "\n"
)
print((ROOT / "requirements.txt").read_text())
print((ROOT / "requirements-train.txt").read_text())
