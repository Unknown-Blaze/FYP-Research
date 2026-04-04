#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "ROOT PIPELINE: Fusion + Ensemble Training"
echo "============================================"

cd "$ROOT_DIR"

bash "$ROOT_DIR/extract_and_ensemble.sh"

echo ""
echo "============================================"
echo "PIPELINE COMPLETE!"
echo "============================================"
