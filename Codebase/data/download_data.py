"""Download the UCI SMS Spam Collection dataset.

Saves cleaned CSV to data/sms_data.csv
"""
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "SMSSpamCollection"
OUT_PATH = DATA_DIR / "sms_data.csv"
ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download():
    if OUT_PATH.exists():
        print(f"[INFO] Dataset already at {OUT_PATH}")
        return

    if not ZIP_PATH.exists():
        print("[INFO] Downloading SMS dataset from UCI ML repo...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("[INFO] Download complete.")

    print("[INFO] Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    df = pd.read_csv(
        RAW_PATH, sep="\t", header=None, names=["label", "text"], encoding="latin-1"
    )
    df.to_csv(OUT_PATH, index=False)
    print(f"[INFO] Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    download()
