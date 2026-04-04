#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "FUSION PIPELINE: extract_predictions -> train_ensemble"
echo "============================================"

cd "$ROOT_DIR"

echo "[1/2] Generating fusion inputs..."
python "$ROOT_DIR/extract_predictions.py"

echo "[2/2] Training ensemble..."
python "$ROOT_DIR/train_ensemble.py"

echo "============================================"
echo "FUSION PIPELINE COMPLETE"
echo "============================================"