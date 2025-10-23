#!/bin/bash

# Quick verification script for enhanced training setup

echo "🔍 Enhanced Training Setup Verification"
echo "======================================"

# Check Python environment
echo "1. Checking Python environment..."
if python -c "import jax, haiku, optax, pandas, numpy; print('✅ All required packages available')" 2>/dev/null; then
    echo "   Python packages: OK"
else
    echo "   ❌ Missing required Python packages"
    echo "   Please ensure JAX, Haiku, Optax, Pandas, and NumPy are installed"
fi

# Check GPU
echo ""
echo "2. Checking GPU availability..."
if nvidia-smi > /dev/null 2>&1; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "   Total Memory: ${GPU_MEM}MB"
    
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "   Free Memory: ${FREE_MEM}MB"
    
    if [ "$FREE_MEM" -gt 10000 ]; then
        echo "   ✅ Sufficient GPU memory available"
    else
        echo "   ⚠️  Limited GPU memory - may need to reduce batch size"
    fi
else
    echo "   ❌ No GPU detected or nvidia-smi not available"
fi

# Check datasets
echo ""
echo "3. Checking datasets..."
TRAIN_FILE="datasets_alphafold_finetune/pmhc_finetune/final_train_70pct.csv"
VALID_FILE="datasets_alphafold_finetune/pmhc_finetune/final_valid_10pct.csv"
TEST_FILE="datasets_alphafold_finetune/pmhc_finetune/final_test_20pct.csv"

for file in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "   ✅ $file: $((lines-1)) samples"
    else
        echo "   ❌ Missing: $file"
    fi
done

# Check training script
echo ""
echo "4. Checking training scripts..."
if [ -f "run_finetuning_ic50.py" ]; then
    echo "   ✅ Training script: run_finetuning_ic50.py"
else
    echo "   ❌ Missing: run_finetuning_ic50.py"
fi

if [ -f "run_prediction_ic50.py" ]; then
    echo "   ✅ Prediction script: run_prediction_ic50.py"
else
    echo "   ❌ Missing: run_prediction_ic50.py"
fi

if [ -f "enhanced_training.sh" ] && [ -x "enhanced_training.sh" ]; then
    echo "   ✅ Enhanced training script: executable"
else
    echo "   ❌ Enhanced training script: not executable"
fi

# Disk space check
echo ""
echo "5. Checking disk space..."
DISK_FREE=$(df -h . | tail -1 | awk '{print $4}')
echo "   Available space: $DISK_FREE"

echo ""
echo "🚀 Verification complete!"
echo ""
echo "💡 Usage examples:"
echo "   Basic training:     ./enhanced_training.sh"
echo "   Custom run ID:      ./enhanced_training.sh --run_id my_experiment"
echo "   Limit GPU memory:   ./enhanced_training.sh --gpu_memory 10"
echo "   Resume training:    ./enhanced_training.sh --resume ./enhanced_models_run1"
echo ""
echo "📚 For help:          ./enhanced_training.sh --help"