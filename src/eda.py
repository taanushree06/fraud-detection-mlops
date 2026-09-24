"""EDA: print key stats and save plots to reports/."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files, no popup windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES, TARGET  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUT = ROOT / "reports"
GREEN, RED = "#4c9f70", "#d9534f"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print("Saved", OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    legit, fraud = df[df[TARGET] == 0], df[df[TARGET] == 1]

    # 1. Class distribution
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=[GREEN, RED])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_ylabel("Transactions")
    save(fig, "class_distribution.png")

    # 2. Amount distribution (log scale so small and large amounts are visible)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.log1p(df["Amount"].max()), 40)
    ax.hist(np.log1p(legit["Amount"]), bins=bins, alpha=0.6, density=True, label="Legitimate", color=GREEN)
    ax.hist(np.log1p(fraud["Amount"]), bins=bins, alpha=0.6, density=True, label="Fraud", color=RED)
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount: fraud vs legitimate")
    ax.legend()
    save(fig, "amount_distribution.png")

    # 3. Top features linked to fraud
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    top = corr.abs().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, f in zip(axes.ravel(), top):
        lo = min(legit[f].quantile(0.01), fraud[f].quantile(0.01))
        hi = max(legit[f].quantile(0.99), fraud[f].quantile(0.99))
        b = np.linspace(lo, hi, 40)
        ax.hist(legit[f], bins=b, alpha=0.6, density=True, color=GREEN, label="Legitimate")
        ax.hist(fraud[f], bins=b, alpha=0.6, density=True, color=RED, label="Fraud")
        ax.set_title(f"{f} (corr with fraud = {corr[f]:.2f})")
    axes[0, 0].legend()
    save(fig, "feature_distributions.png")

    # 4. Correlation with target
    s = corr.sort_values()
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(s.index, s.values, color=[RED if v > 0 else "#5b8def" for v in s.values])
    ax.set_title("Correlation of each feature with Class (fraud)")
    save(fig, "correlation_with_class.png")

    # Printed summary
    vcorr = df[[f"V{i}" for i in range(1, 29)]].corr().abs().to_numpy().copy()
    np.fill_diagonal(vcorr, 0)
    print("-" * 55)
    print(f"Amount  legit: mean={legit['Amount'].mean():.2f}  median={legit['Amount'].median():.2f}")
    print(f"Amount  fraud: mean={fraud['Amount'].mean():.2f}  median={fraud['Amount'].median():.2f}")
    print("Top 6 features linked to fraud:", top)
    print(f"Max correlation between any two V features: {vcorr.max():.3f} (PCA makes them independent)")


if __name__ == "__main__":
    main()
