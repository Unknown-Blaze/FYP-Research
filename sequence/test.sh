#!/bin/bash
# Runs the v5 training script using a config file.
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the config file to use
CONFIG_FILE="$ROOT_DIR/config.yaml"

echo "Starting testing with config: $CONFIG_FILE"
echo "Log will be saved to the 'output_dir' specified in the config."

cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" train_finetune.py --config "$CONFIG_FILE" --evaluate_only

echo "Testing finished."