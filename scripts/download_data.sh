#!/usr/bin/env bash
set -e
# Requires `pip install kaggle` and a valid kaggle.json API key file.
DATA_DIR="data/raw"
mkdir -p "${DATA_DIR}"
kaggle competitions download -c ieee-fraud-detection -p "${DATA_DIR}"
unzip -o "${DATA_DIR}/ieee-fraud-detection.zip" -d "${DATA_DIR}"
rm "${DATA_DIR}/ieee-fraud-detection.zip"
echo "CSV files extracted to ${DATA_DIR}"
