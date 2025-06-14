#!/usr/bin/env python3
"""Download Kaggle dataset for the project.

Usage:
  python scripts/download_data.py

Requires ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.
The dataset is downloaded to ``data/raw/`` and the archive is unzipped.
If the destination directory already contains the CSV and a matching
``.sha256`` file, the script exits without re-downloading.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import hashlib

try:
    from src.dataprep import CSV_PATH
except ImportError:
    print(
        "Run `pip install -e .` from the repository root so Python can find "
        "the `src` package"
    )
    sys.exit(1)

DATASET = "architsharma01/loan-approval-prediction-dataset"
DEST_DIR = Path("data/raw")
CSV_NAME = Path(CSV_PATH).name


def main() -> None:
    """Authenticate with Kaggle and download the dataset."""
    csv_path = DEST_DIR / CSV_NAME
    sha_path = DEST_DIR / f"{CSV_NAME}.sha256"
    if csv_path.exists() and sha_path.exists():
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if sha_path.read_text().strip() == digest:
            print(f"Dataset already present at {csv_path}. Skipping download.")
            return

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        sys.exit("Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")

    # ``KaggleApi`` reads credentials from env vars. Ensure they are set
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(DEST_DIR), unzip=True)
    if csv_path.exists():
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        sha_path.write_text(sha)
    print("Download complete.")


if __name__ == "__main__":
    main()
