#!/bin/bash

# MVSF-AB Training Script
# Usage: ./run_training.sh

echo "🚀 Starting MVSF-AB Training"
echo "============================"

# Activate virtual environment
source mvsf_env/bin/activate

# Check GPU availability
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# Run training with optimized parameters
echo ""
echo "🏋️ Starting training with Cross-Validation..."

python train.py \
    --train datasets/pairs_sabdab.csv \
    --test datasets/pairs_benchmark.csv \
    --seq-path datasets/seq_natural.fasta \
    --feature-path datasets/seq_natural_embedding.csv \
    --num-epochs 30 \
    --batch-size 16 \
    --weight-decay 0.00001 \
    --lr 0.00001 \
    --kfolds 10 \
    --weight-module1 1 \
    --weight-module2 1 \
    --device 0 \
    --save-prefix models/mvsf_model \
    --outfile training_log.txt \
    --seed 42

echo "✅ Training completed!"
echo "📄 Check training_log.txt for detailed logs"
echo "📁 Models saved with prefix: models/mvsf_model"
