#!/bin/bash
# This script runs the full ESM-2 fine-tuning process.
# It ensures the script stops if any command fails.
set -e

# --- Create Log Directory ---
mkdir -p logs

# --- Define Log File ---
LOG_FILE="logs/train_$(date +'%Y-%m-%d_%H-%M-%S').log"

# --- Run the Command ---
# - Uses 'python -u' for unbuffered output.
# - Uses '2>&1' to redirect stderr (warnings/errors) to stdout.
# - Uses 'tee' to copy merged stdout/stderr to screen and log file.
echo "Starting training... Logs (including warnings) will be saved to $LOG_FILE"
CUDA_VISIBLE_DEVICES=0 python train_finetune.py \
    --data_file iedb_dataset.csv \
    --output_dir ./checkpoints \
    --model_name "facebook/esm2_t33_650M_UR50D" \
    --epochs 50 \
    --batch_size 32 \
    --accum_steps 1 \
    --lr 1e-5 \
    --seed 42 \
    --test_size 0.1 \
    --valid_size 0.1 \
    --convert_to_pic50 \
    --normalize \
    --use_amp 2>&1 | tee $LOG_FILE

echo "Training finished. Logs saved to $LOG_FILE"