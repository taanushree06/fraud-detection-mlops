"""Download the Credit Card Fraud dataset into data/raw/creditcard.csv."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "creditcard.csv"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already exists: {DEST}")
        return
    print("Downloading dataset (~150 MB), please wait...")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        if DEST.exists():
            DEST.unlink()
        print(f"Download failed: {exc}")
        print("Manual option: download creditcard.csv from")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"and put it in: {DEST}")
        sys.exit(1)
    print(f"Saved to {DEST}")


if __name__ == "__main__":
    main()
