"""Threshold optimization: choose on VALIDATION data, confirm on TEST data."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import compute_metrics  # noqa: E402
from src.preprocessing import get_splits, load_data  # noqa: E402

CAND_DIR = ROOT / "models" / "candidates"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
BETA = 2.0  # recall counts twice as much as precision
SHOW = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
GRID = [round(i / 100, 2) for i in range(5, 96, 5)]  # 0.05 ... 0.95


def fbeta(p, r, beta=BETA):
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def main():
    best = json.loads((REPORTS / "best_model.json").read_text())
    name, slug = best["best_model"], best["slug"]
    pipe = joblib.load(CAND_DIR / f"{slug}.joblib")
    print(f"Model: {name}")

    df = load_data()
    _, X_val, X_test, _, y_val, y_test = get_splits(df)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    # Evaluate many thresholds on VALIDATION data
    rows = []
    for t in GRID:
        m = compute_metrics(y_val, val_proba, t)
        rows.append({
            "threshold": t, "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "f2": fbeta(m["precision"], m["recall"]),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        })
    table = pd.DataFrame(rows).set_index("threshold")
    table.round(4).to_csv(REPORTS / "threshold_analysis.csv")

    print("\nTHRESHOLD ANALYSIS (validation set)")
    print("-" * 66)
    print(table.loc[SHOW].round(4).to_string())
    print("-" * 66)

    # Best F2; if tied, take the higher threshold (fewer false alarms)
    best_f2 = table["f2"].max()
    chosen = float(table[table["f2"] == best_f2].index.max())
    v = table.loc[chosen]
    print(f"\nChosen threshold: {chosen}  (best F2 = {best_f2:.4f} on validation)")
    print(f"Validation at {chosen}: precision={v['precision']:.3f} recall={v['recall']:.3f} "
          f"caught={int(v['tp'])} false alarms={int(v['fp'])} missed={int(v['fn'])}")

    # Final check on TEST data (never used for training or for choosing the threshold)
    t_default = compute_metrics(y_test, test_proba, 0.5)
    t_chosen = compute_metrics(y_test, test_proba, chosen)
    print("\nFINAL CHECK ON TEST SET")
    print("-" * 66)
    for label, m in [("threshold 0.5", t_default), (f"threshold {chosen}", t_chosen)]:
        print(f"{label:15}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
              f"| caught={m['tp']} false alarms={m['fp']} missed={m['fn']}")
    print(f"ROC-AUC={t_chosen['roc_auc']:.4f}  PR-AUC={t_chosen['pr_auc']:.4f}")
    print("-" * 66)

    # Save the final threshold for the app
    MODELS.mkdir(exist_ok=True)
    (MODELS / "threshold.json").write_text(json.dumps({
        "threshold": chosen,
        "model": name,
        "selected_by": f"max F{BETA:g} on validation set",
        "validation": {k: round(float(v[k]), 4) for k in ["precision", "recall", "f1", "f2"]},
        "test": {k: round(t_chosen[k], 4) for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]},
    }, indent=2))
    print("Saved models\\threshold.json")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, style in [("precision", "-o"), ("recall", "-o"), ("f1", "-o"), ("f2", "--")]:
        ax.plot(table.index, table[col], style, markersize=3, label=col.upper() if col != "f2" else "F2")
    ax.axvline(chosen, color="red", linestyle=":", label=f"chosen = {chosen}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{name}: metrics vs threshold (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "threshold_curve.png", dpi=120)
    plt.close(fig)
    print("Saved reports\\threshold_analysis.csv and threshold_curve.png")


if __name__ == "__main__":
    main()
