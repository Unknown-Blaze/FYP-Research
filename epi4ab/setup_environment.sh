#!/bin/bash

# Epi4Ab Environment Setup Script
# This script ensures proper conda initialization and environment activation

echo "🔧 Setting up Epi4Ab environment..."

# Initialize conda
if ! command -v conda &> /dev/null; then
    echo "📦 Initializing conda..."
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Activate environment
echo "🚀 Activating epi4ab environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate epi4ab

# Verify environment
echo "✅ Environment setup complete"
echo "Current environment: $CONDA_DEFAULT_ENV"

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "
import torch
import torch_geometric
import pandas
import numpy
print('✅ PyTorch:', torch.__version__)
print('✅ PyTorch Geometric:', torch_geometric.__version__)
print('✅ Pandas:', pandas.__version__)
print('✅ NumPy:', numpy.__version__)
print('✅ All dependencies verified')
"

echo ""
echo "🎯 Environment is ready!"
echo "You can now use: ./master_run.sh [command]"
