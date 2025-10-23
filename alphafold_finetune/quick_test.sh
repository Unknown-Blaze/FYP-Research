#!/bin/bash
# Quick test run - train for just 100 steps to verify improvements

# Activate conda environment if not already active
if [[ "$CONDA_DEFAULT_ENV" != "alphafold_finetune_38" ]]; then
    echo "Activating alphafold_finetune_38 environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate alphafold_finetune_38
fi

echo "================================================"
echo "  QUICK VERIFICATION RUN (100 steps)"
echo "================================================"
echo ""
echo "This will:"
echo "  1. Train for 100 steps only"
echo "  2. Run sanity check at step 50"
echo "  3. Show if model is collapsing or learning"
echo "  4. Complete in ~5-10 minutes"
echo ""
read -p "Press Enter to start..."

cd /home3/rudrapra001/FYP-Research/alphafold_finetune

# Create test output directory
TEST_DIR="enhanced_models_quick_test"
mkdir -p $TEST_DIR
mkdir -p training_logs/$TEST_DIR

# Run for minimal steps (using only 1 GPU for quick test)
CUDA_VISIBLE_DEVICES=1 python run_finetuning_ic50.py \
    --train_dataset datasets_alphafold_finetune/pmhc_finetune/final_train_70pct.csv \
    --valid_dataset datasets_alphafold_finetune/pmhc_finetune/final_valid_10pct.csv \
    --data_dir /home3/rudrapra001/FYP-Research/alphafold_finetune \
    --output_dir $TEST_DIR \
    --outprefix enhanced_test \
    --save_steps 50 \
    --valid_steps 50 \
    --num_epochs 1 \
    --apply_every 32 \
    --only_fit_binder true \
    --lr_coef 2.0 \
    --regression_loss_weight 1.0 \
    --grad_norm_clip 5.0 \
    --patience_limit 50 \
    --quick_valid true \
    --quick_valid_size 50 \
    2>&1 | tee training_logs/$TEST_DIR/quick_test.log

echo ""
echo "================================================"
echo "  QUICK TEST COMPLETE"
echo "================================================"
echo ""
echo "📊 Check the results above for:"
echo "  ✅ Prediction std > 0.5 (good variance)"
echo "  ✅ Prediction range expanding over time"
echo "  ❌ Collapse warning at step 50"
echo "  ❌ Predictions stuck at ~6.5"
echo ""
echo "If sanity check PASSED, run full training with:"
echo "  bash train.sh"
echo ""
