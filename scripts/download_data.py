#!/usr/bin/env python3
"""Download Kaggle dataset for the project.

Usage:
  python scripts/download_data.py

Requires ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.
The dataset is downloaded to ``data/raw/`` and the archive is unzipped.
If the destination directory already contains files, the script exits
without downloading again.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET = 'architsharma01/loan-approval-prediction-dataset'
DEST_DIR = Path('data/raw')


def main() -> None:
  """Authenticate with Kaggle and download the dataset."""
  if DEST_DIR.exists() and any(DEST_DIR.iterdir()):
    print(f'Dataset already present in {DEST_DIR}. Skipping download.')
    return

  username = os.getenv('KAGGLE_USERNAME')
  key = os.getenv('KAGGLE_KEY')
  if not username or not key:
    sys.exit('Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.')

  # ``KaggleApi`` reads credentials from env vars. Ensure they are set
  os.environ['KAGGLE_USERNAME'] = username
  os.environ['KAGGLE_KEY'] = key

  DEST_DIR.mkdir(parents=True, exist_ok=True)

  from kaggle.api.kaggle_api_extended import KaggleApi
  api = KaggleApi()
  api.authenticate()
  api.dataset_download_files(DATASET, path=str(DEST_DIR), unzip=True)
  print('Download complete.')


if __name__ == '__main__':
  main()
