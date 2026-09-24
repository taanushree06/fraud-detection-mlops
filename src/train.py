"""Train Logistic Regression, Random Forest and XGBoost; compare on validation data."""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import validate  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.preprocessing import RANDOM_STATE, build_preprocessor, get_splits, load_data  # noqa: E402

CAND_DIR = ROOT / "models" / "candidates"
REPORTS = ROOT / "reports"
EXPERIMENT = "credit-card-fraud-detection"
EVAL_THRESHOLD = 0.5  # default threshold for this comparison; optimized in Phase 5


def get_models(y_train):
    """Each entry: name -> (estimator, parameters to log). Imbalance handled by class weights."""
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = round(neg / pos, 1)
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {"max_iter": 1000, "class_weight": "balanced"},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 2,
             "class_weight": "balanced_subsample"},
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1, scale_pos_weight=spw,
                eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "scale_pos_weight": spw},
        ),
    }


def log_to_mlflow(name, params, metrics, train_seconds):
    """Log one model run. Training continues even if MLflow has a problem."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("imbalance_method", "class_weight")
            mlflow.log_param("eval_threshold", EVAL_THRESHOLD)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("train_seconds", round(train_seconds, 1))
            for k in ["precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]:
                mlflow.log_metric(k, metrics[k])
    except Exception as exc:  # noqa: BLE001
        print(f"   WARNING: MLflow logging skipped for {name}: {exc}")


def main():
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load_data()
    validate(df)  # pipeline stops here if data is bad
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df)
    print(f"Train: {len(X_train):,} rows ({int(y_train.sum())} fraud) | "
          f"Validation: {len(X_val):,} rows ({int(y_val.sum())} fraud)")
    print("Imbalance method: class weights (no resampling, so no leakage)\n")

    results, probas = {}, {}
    for name, (estimator, params) in get_models(y_train).items():
        print(f"Training {name} ...")
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        start = time.time()
        pipe.fit(X_train, y_train)  # scaler is fitted on TRAIN only
        seconds = time.time() - start

        proba = pipe.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, proba, EVAL_THRESHOLD)
        results[name], probas[name] = metrics, proba

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, CAND_DIR / f"{slug}.joblib")
        log_to_mlflow(name, params, metrics, seconds)
        print(f"   done in {seconds:.0f}s | PR-AUC={metrics['pr_auc']:.4f} recall={metrics['recall']:.3f}\n")

    # Comparison table
    table = pd.DataFrame(results).T
    cols = ["precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn"]
    table = table[cols]
    table[["precision", "recall", "f1", "roc_auc", "pr_auc"]] = table[
        ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    ].round(4)
    table[["tp", "fp", "fn"]] = table[["tp", "fp", "fn"]].astype(int)
    table.index.name = "model"
    table.to_csv(REPORTS / "model_comparison.csv")

    print("=" * 78)
    print(f"MODEL COMPARISON (validation set, threshold = {EVAL_THRESHOLD})")
    print("=" * 78)
    print(table.to_string())
    print("=" * 78)
    print("tp = frauds caught | fp = false alarms | fn = frauds missed")

    best = table["pr_auc"].astype(float).idxmax()
    (REPORTS / "best_model.json").write_text(json.dumps({
        "best_model": best,
        "slug": best.lower().replace(" ", "_"),
        "selected_by": "PR-AUC on validation set",
    }, indent=2))
    print(f"\nBest model by PR-AUC: {best}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        p, r, _ = precision_recall_curve(y_val, proba)
        ax.plot(r, p, label=f"{name} (PR-AUC={results[name]['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "pr_curves.png", dpi=120)
    plt.close(fig)
    print("Saved reports\\model_comparison.csv, confusion_matrices.png, pr_curves.png")
    print("Saved models to models\\candidates\\")


if __name__ == "__main__":
    main()
