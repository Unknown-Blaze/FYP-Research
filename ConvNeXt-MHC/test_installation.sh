#!/bin/bash
# Test ConvNeXt-MHC installation and environment

echo "=== ConvNeXt-MHC Installation Test ==="
echo ""

# Check conda environment
echo "1. Testing conda environment..."
if conda env list | grep -q "convnext-mhc"; then
    echo "   ✓ Environment 'convnext-mhc' exists"
    
    # Activate and check packages
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate convnext-mhc
    
    echo ""
    echo "2. Checking Python packages..."
    python -c "import tensorflow as tf; print(f'   ✓ TensorFlow {tf.__version__}')" 2>/dev/null || echo "   ❌ TensorFlow not found"
    python -c "import numpy as np; print(f'   ✓ NumPy {np.__version__}')" 2>/dev/null || echo "   ❌ NumPy not found"
    python -c "import pandas as pd; print(f'   ✓ Pandas {pd.__version__}')" 2>/dev/null || echo "   ❌ Pandas not found"
else
    echo "   ❌ Environment not found. Run: conda env create -f environment.yml"
    exit 1
fi

# Check netMHCpan
echo ""
echo "3. Testing netMHCpan..."
NETMHC_PATH="/home3/rudrapra001/FYP-Research/ConvNeXt-MHC/netMHCpan-4.1"
if [ -f "$NETMHC_PATH/netMHCpan" ]; then
    echo "   ✓ netMHCpan found"
    
    # Test execution
    cd "$NETMHC_PATH"
    if ./netMHCpan -h > /dev/null 2>&1; then
        echo "   ✓ netMHCpan executable"
    else
        echo "   ❌ netMHCpan not executable or dependencies missing"
    fi
    
    # Check tmp directory
    if [ -d "$NETMHC_PATH/tmp" ]; then
        echo "   ✓ tmp directory exists"
    else
        echo "   ❌ tmp directory missing (create with: mkdir $NETMHC_PATH/tmp)"
    fi
else
    echo "   ❌ netMHCpan not found at $NETMHC_PATH"
fi

# Check model weights
echo ""
echo "4. Checking model weights..."
WEIGHT_DIR="/home3/rudrapra001/FYP-Research/ConvNeXt-MHC/ConvNeXt-MHC/save_model_weight"
if [ -f "$WEIGHT_DIR/ConvNeXt-MHC_BA.h5" ]; then
    BA_SIZE=$(du -h "$WEIGHT_DIR/ConvNeXt-MHC_BA.h5" | cut -f1)
    echo "   ✓ BA model found ($BA_SIZE)"
else
    echo "   ❌ BA model not found"
fi

if [ -f "$WEIGHT_DIR/ConvNeXt-MHC_AP.h5" ]; then
    AP_SIZE=$(du -h "$WEIGHT_DIR/ConvNeXt-MHC_AP.h5" | cut -f1)
    echo "   ✓ AP model found ($AP_SIZE)"
else
    echo "   ❌ AP model not found"
fi

# Check encoding resources
echo ""
echo "5. Checking encoding resources..."
TRANS_DIR="/home3/rudrapra001/FYP-Research/ConvNeXt-MHC/ConvNeXt-MHC/Trans2Input9"
if [ -f "$TRANS_DIR/onehot_new.csv" ]; then
    echo "   ✓ onehot_new.csv found"
else
    echo "   ❌ onehot_new.csv not found"
fi

if [ -f "$TRANS_DIR/pseudo_encoding.csv" ]; then
    echo "   ✓ pseudo_encoding.csv found"
else
    echo "   ❌ pseudo_encoding.csv not found"
fi

if [ -f "$TRANS_DIR/pseudo/HLA_pseudoSeqs.fasta" ]; then
    echo "   ✓ HLA_pseudoSeqs.fasta found"
else
    echo "   ❌ HLA_pseudoSeqs.fasta not found"
fi

if [ -f "$TRANS_DIR/pseudo/site.txt" ]; then
    echo "   ✓ site.txt found"
else
    echo "   ❌ site.txt not found"
fi

# Test data
echo ""
echo "6. Checking test data..."
DATA_DIR="/home3/rudrapra001/FYP-Research/ConvNeXt-MHC/ConvNeXt-MHC/data_set"
if [ -f "$DATA_DIR/example.csv" ]; then
    NUM_LINES=$(wc -l < "$DATA_DIR/example.csv")
    echo "   ✓ example.csv found ($NUM_LINES lines)"
else
    echo "   ❌ example.csv not found"
fi

# Quick Python test
echo ""
echo "7. Testing Python imports..."
cd /home3/rudrapra001/FYP-Research/ConvNeXt-MHC/ConvNeXt-MHC
python << EOF
try:
    from model import convnext_10_10
    print("   ✓ model.py imports successfully")
except Exception as e:
    print(f"   ❌ model.py import error: {e}")

try:
    from utils import input_matrix_generation
    print("   ✓ utils.py imports successfully")
except Exception as e:
    print(f"   ❌ utils.py import error: {e}")
EOF

# Summary
echo ""
echo "=== Test Complete ==="
echo ""
echo "If all checks passed (✓), you're ready to run predictions!"
echo "If any failed (❌), check SETUP_GUIDE.md for troubleshooting."
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate convnext-mhc"
echo "  2. Prepare data CSV with columns: allele, peptide, 9mer"
echo "  3. Run: python run_model.py"
