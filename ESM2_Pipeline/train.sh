#!/bin/bash
# Runs the v5 training script using a config file.
set -e

# Define the config file to use
CONFIG_FILE="config.yaml"

echo "Starting training with config: $CONFIG_FILE"
echo "Log will be saved to the 'output_dir' specified in the config."

CUDA_VISIBLE_DEVICES=2 python train_finetune.py --config $CONFIG_FILE

echo "Training finished."