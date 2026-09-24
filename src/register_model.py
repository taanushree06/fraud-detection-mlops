"""Phase 6+7: build the final artifact, log it to MLflow, register the model."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.preprocessing import get_splits, load_data  # noqa: E402

CAND_DIR = ROOT / "models" / "candidates"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ARTIFACT_PATH = MODELS / "fraud_pipeline.joblib"
EXPERIMENT = "credit-card-fraud-detection"
REGISTERED_NAME = "fraud-detection-model"
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def log_to_mlflow(pipe, name, threshold, selected_by, val_m, test_m):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"FINAL - {name}") as run:
        mlflow.set_tag("stage", "final")
        mlflow.log_param("model_name", name)
        mlflow.log_param("imbalance_method", "class_weight")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("threshold_selection", selected_by)
        for k in KEYS:
            mlflow.log_metric(f"val_{k}", val_m[k])
            mlflow.log_metric(f"test_{k}", test_m[k])
        mlflow.log_metric("test_frauds_caught", test_m["tp"])
        mlflow.log_metric("test_false_alarms", test_m["fp"])
        mlflow.log_metric("test_frauds_missed", test_m["fn"])

        for f in ["best_model.json", "model_comparison.csv", "threshold_analysis.csv",
                  "threshold_curve.png", "pr_curves.png", "confusion_matrices.png"]:
            if (REPORTS / f).exists():
                mlflow.log_artifact(str(REPORTS / f), artifact_path="reports")
        mlflow.log_artifact(str(MODELS / "threshold.json"))
        mlflow.log_artifact(str(ARTIFACT_PATH))

        try:
            mlflow.sklearn.log_model(pipe, name="model", registered_model_name=REGISTERED_NAME, serialization_format="cloudpickle")
        except TypeError:  # older MLflow versions
            mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=REGISTERED_NAME, serialization_format="cloudpickle")
        run_id = run.info.run_id

    # Mark the newest registered version as "production"
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(REGISTERED_NAME, "production", str(latest))
    return run_id, latest


def main():
    best = json.loads((REPORTS / "best_model.json").read_text())
    thr_info = json.loads((MODELS / "threshold.json").read_text())
    name, slug = best["best_model"], best["slug"]
    threshold = float(thr_info["threshold"])
    pipe = joblib.load(CAND_DIR / f"{slug}.joblib")

    df = load_data()
    _, X_val, X_test, _, y_val, y_test = get_splits(df)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]
    val_m = compute_metrics(y_val, val_proba, threshold)
    test_m = compute_metrics(y_test, test_proba, threshold)

    # 1. One file with preprocessing + model + threshold
    artifact = {
        "pipeline": pipe,  # scaler + model
        "threshold": threshold,
        "model_name": name,
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_metrics": {k: round(test_m[k], 4) for k in KEYS},
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Model: {name} | Threshold: {threshold}")

    # 2. Two real example transactions from the TEST set for the app demo
    fraud_idx = [i for i in X_test.index[y_test == 1] if test_proba[X_test.index.get_loc(i)] >= threshold]
    legit_idx = [i for i in X_test.index[y_test == 0] if test_proba[X_test.index.get_loc(i)] < 0.05]
    fraud_idx.sort(key=lambda i: test_proba[X_test.index.get_loc(i)])
    samples = {
        "fraud_example": X_test.loc[fraud_idx[len(fraud_idx) // 2]].astype(float).to_dict(),
        "legit_example": X_test.loc[legit_idx[0]].astype(float).to_dict(),
    }
    (MODELS / "sample_inputs.json").write_text(json.dumps(samples, indent=2))
    print("Saved models\\sample_inputs.json (one fraud, one legitimate transaction)")

    # 3. MLflow tracking + registry
    try:
        run_id, version = log_to_mlflow(pipe, name, threshold, thr_info["selected_by"], val_m, test_m)
        print(f"MLflow: run {run_id[:8]} logged | registered '{REGISTERED_NAME}' version {version} "
              f"(alias: production)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: MLflow step failed: {exc}")
        print("The saved model file is still fine. Paste this message and I will fix it.")


if __name__ == "__main__":
    main()

