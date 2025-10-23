#!/bin/bash

# Enhanced Training Command for Optimized pIC50 Regression
# This script includes all the improvements for better model performance

echo "🧬 Starting Enhanced AlphaFold Fine-tuning for pIC50 Prediction"
echo "=============================================================="

# Check and activate conda environment
if [[ "$CONDA_DEFAULT_ENV" != "alphafold_finetune_38" ]]; then
    echo "📦 Activating conda environment: alphafold_finetune_38"
    eval "$(conda shell.bash hook)"
    conda activate alphafold_finetune_38
    
    if [[ "$CONDA_DEFAULT_ENV" != "alphafold_finetune_38" ]]; then
        echo "❌ Failed to activate alphafold_finetune_38 environment!"
        echo "   Please run: conda activate alphafold_finetune_38"
        exit 1
    fi
    echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
else
    echo "✅ Already in correct environment: $CONDA_DEFAULT_ENV"
fi
echo ""

# Configuration variables
RUN_ID="run1"
GPU_MEMORY_GB=12
CHECKPOINT_DIR=""
RESUME_FROM_CHECKPOINT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        --gpu_memory)
            GPU_MEMORY_GB="$2"
            shift 2
            ;;
        --resume)
            CHECKPOINT_DIR="$2"
            RESUME_FROM_CHECKPOINT=true
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --run_id <name>        Set run identifier (default: run1)"
            echo "  --gpu_memory <GB>      Set GPU memory limit in GB (default: 12)"
            echo "  --resume <checkpoint>  Resume from checkpoint directory"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set up GPU memory limiting
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.$(echo "scale=0; $GPU_MEMORY_GB * 100 / 24" | bc)"
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

echo "🖥️  Configuration:"
echo "   Run ID: $RUN_ID"
echo "   GPU Memory Limit: ${GPU_MEMORY_GB}GB"
echo "   Memory Fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    echo "   Resuming from: $CHECKPOINT_DIR"
fi
echo ""

# Output directory setup
OUTPUT_DIR="./enhanced_models_${RUN_ID}"
mkdir -p "$OUTPUT_DIR"

# Build training command
# Enhanced training command with correct and existing flags only
# CRITICAL FIX: Train only the binder head first to avoid model collapse
TRAINING_CMD="python run_finetuning_ic50.py \
    --data_dir /home3/rudrapra001/FYP-Research/alphafold_finetune \
    --train_dataset datasets_alphafold_finetune/pmhc_finetune/final_train_70pct.csv \
    --valid_dataset datasets_alphafold_finetune/pmhc_finetune/final_valid_10pct.csv \
    --output_dir $OUTPUT_DIR \
    --model_name model_2_ptm \
    --num_epochs 20 \
    --batch_size 1 \
    --save_steps 100 \
    --regression_loss_weight 15.0 \
    --normalize_features=true \
    --dropout_rate 0.2 \
    --lr_coef 0.1 \
    --grad_norm_clip 0.5 \
    --only_fit_binder=true" # Add checkpoint resumption if specified
if [ "$RESUME_FROM_CHECKPOINT" = true ]; then
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "❌ Error: Checkpoint directory $CHECKPOINT_DIR not found"
        exit 1
    fi
    TRAINING_CMD="$TRAINING_CMD --resume_from_checkpoint $CHECKPOINT_DIR"
fi

echo "🚀 Starting Enhanced Training..."

# Set up comprehensive logging
LOG_DIR="./training_logs"
LOG_FILE="${LOG_DIR}/train_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "📋 Training log file: ${LOG_FILE}"
echo "📋 Monitor progress with: tail -f ${LOG_FILE}"
echo ""

echo "Command: $TRAINING_CMD" | tee "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run the training with comprehensive logging (both terminal and log file)
echo "⏳ Starting compilation and training - this may take 5-10 minutes initially..." | tee -a "${LOG_FILE}"
echo "🔧 JAX is compiling the enhanced model architecture..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

if ! eval $TRAINING_CMD 2>&1 | tee -a "${LOG_FILE}"; then
    echo "" | tee -a "${LOG_FILE}"
    echo "❌ Training failed or was interrupted!" | tee -a "${LOG_FILE}"
    echo "💾 You can resume from the latest checkpoint using:" | tee -a "${LOG_FILE}"
    echo "   $0 --resume $OUTPUT_DIR --run_id $RUN_ID" | tee -a "${LOG_FILE}"
    echo "📋 Check full log for details: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "" | tee -a "${LOG_FILE}"
echo "✅ Enhanced training completed successfully!" | tee -a "${LOG_FILE}"
echo "📊 Key improvements include:" | tee -a "${LOG_FILE}"
echo "  🧠 Multi-layer neural network with residual connections" | tee -a "${LOG_FILE}"
echo "  🎯 Attention mechanism for feature importance" | tee -a "${LOG_FILE}"
echo "  🔬 Comprehensive structural and sequence features" | tee -a "${LOG_FILE}"
echo "  📉 Advanced loss functions (Huber + Ranking + Confidence)" | tee -a "${LOG_FILE}"
echo "  ⚡ AdamW optimizer with cosine annealing" | tee -a "${LOG_FILE}"
echo "  🛑 Early stopping with patience" | tee -a "${LOG_FILE}"
echo "  📈 Enhanced validation metrics and monitoring" | tee -a "${LOG_FILE}"
echo "  🎓 Curriculum learning and adaptive regularization" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "📋 Full training log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Save final training summary
SUMMARY_FILE="$OUTPUT_DIR/training_summary_${RUN_ID}.txt"
echo "Training Summary for $RUN_ID" > "$SUMMARY_FILE"
echo "Completed at: $(date)" >> "$SUMMARY_FILE"
echo "GPU Memory Limit: ${GPU_MEMORY_GB}GB" >> "$SUMMARY_FILE"
echo "Dataset: 70% train (final_train_70pct.csv), 10% valid (final_valid_10pct.csv)" >> "$SUMMARY_FILE"
echo "Enhanced features: Multi-layer network, attention, comprehensive features" >> "$SUMMARY_FILE"
echo "Checkpoints saved in: $OUTPUT_DIR" >> "$SUMMARY_FILE"

echo "💾 Training summary saved to: $SUMMARY_FILE"

# Optional: Run prediction on test set with auto-timeout 
echo ""
echo "Would you like to run predictions on the test set? (y/n)"
echo "⏰ Auto-running predictions in 60 seconds if no response..."

# Use timeout to wait for user input (60 seconds)
response=""
if read -t 60 -r response; then
    echo "User responded: $response"
else
    echo ""
    echo "⏰ No response detected - auto-running predictions on CUDA device 2..."
    response="y"
fi

# Handle user response
if [[ "$response" =~ ^[Nn]$ ]]; then
    echo "⏭️  Skipping predictions as requested."
elif [[ "$response" =~ ^[Yy]$ ]] || [ -z "$response" ]; then
    echo "🔮 Running enhanced predictions..."
    
    TEST_OUTPUT_DIR="./enhanced_predictions_${RUN_ID}"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Find the latest/best checkpoint
    BEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "*_params.pkl" -type f | head -1)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        echo "❌ No checkpoint found in $OUTPUT_DIR"
        exit 1
    fi
    
    echo "📂 Using checkpoint: $BEST_CHECKPOINT"
    
    # Auto-run on CUDA device 2 as requested
    echo "🎯 Running predictions on CUDA device 2..."
    CUDA_VISIBLE_DEVICES=2 python run_prediction_ic50.py \
        --targets datasets_alphafold_finetune/pmhc_finetune/final_test_20pct.csv \
        --data_dir /home3/rudrapra001/FYP-Research/alphafold_finetune \
        --model_names model_2_ptm \
        --model_params_files "$BEST_CHECKPOINT" \
        --output_dir "$TEST_OUTPUT_DIR" \
        --outfile_prefix enhanced_predictions_${RUN_ID} \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ Enhanced predictions completed!"
        echo "📁 Results saved in: $TEST_OUTPUT_DIR"
    else
        echo "❌ Prediction failed - check the logs above"
    fi
fi

echo ""
echo "🎉 Training session '$RUN_ID' finished at $(date)"
echo "📁 Model checkpoints: $OUTPUT_DIR"
echo "📋 Training summary: $SUMMARY_FILE"
echo ""
echo "💡 To resume training later:"
echo "   $0 --resume $OUTPUT_DIR --run_id ${RUN_ID}_continued"
echo ""


# CUDA_VISIBLE_DEVICES=2 bash enhanced_training.sh --run_id improved_architecture --gpu_memory 12