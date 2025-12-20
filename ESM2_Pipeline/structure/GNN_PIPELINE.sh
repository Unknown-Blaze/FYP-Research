#!/bin/bash
# ============================================================================
# COMPLETE GNN PIPELINE
# PDB → Graphs → GNN Training
# ============================================================================

set -e

echo ""
echo "============================================================"
echo "  COMPLETE STRUCTURE → GRAPH → GNN PIPELINE"
echo "============================================================"
echo ""

STRUCTURE_DIR="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure"
cd "$STRUCTURE_DIR"

# Step 1: Check if structures are ready
echo "[Step 1/3] Checking PDB structures..."
echo "------------------------------------------------------------"

MHC_COUNT=$(find data/generated_pdbs/mhc -name "*_unrelaxed_rank_001_*.pdb" -type f 2>/dev/null | wc -l)
PEP_COUNT=$(find data/generated_pdbs/peptide -name "pep_*_unrelaxed_rank_001_*.pdb" -type f 2>/dev/null | wc -l)

echo "  MHC structures: $MHC_COUNT / 58"
echo "  Peptide structures: $PEP_COUNT / 24483"
echo ""

if [ "$MHC_COUNT" -lt 58 ] || [ "$PEP_COUNT" -lt 24483 ]; then
    echo "⚠ Not all structures generated yet!"
    echo "  Continue generating structures in separate terminal:"
    echo "    export MHC_MODE=accurate RECYCLES=1 PEP_ENGINE=colabfold"
    echo "    ./GENERATE_STRUCTURES.sh"
    echo ""
    echo "  You can run graph conversion and training later with:"
    echo "    ./GNN_PIPELINE.sh"
    exit 0
fi

echo "✓ All structures ready!"
echo ""

# Step 2: Convert PDB to graphs
echo "[Step 2/3] Converting PDB files to graphs..."
echo "------------------------------------------------------------"

COLABFOLD_PY="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure/localcolabfold/colabfold-conda/bin/python"
COLABFOLD_LIB="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure/localcolabfold/colabfold-conda/lib"
export LD_LIBRARY_PATH="$COLABFOLD_LIB:${LD_LIBRARY_PATH:-}"

# Check for torch-geometric
$COLABFOLD_PY - <<'PYCHECK'
try:
    import torch_geometric
    print("torch_geometric OK")
except Exception as e:
    print(f"torch_geometric missing: {e}")
    print("Install with: pip install torch-geometric pytorch-scatter pytorch-sparse")
    import sys
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "❌ torch-geometric not installed"
    echo "   Install with: pip install torch-geometric pytorch-scatter pytorch-sparse"
    exit 1
fi

$COLABFOLD_PY pdb_to_graphs.py \
    --mhc-pdb-dir data/generated_pdbs/mhc \
    --peptide-pdb-dir data/generated_pdbs/peptide \
    --output-dir data/graphs \
    --distance-threshold 5.0

echo ""

# Step 3: Train GNN
echo "[Step 3/3] Training Graph Neural Network..."
echo "------------------------------------------------------------"
echo "Using --resume to continue from checkpoint if available"

$COLABFOLD_PY train_gnn_model.py --resume

echo ""
echo "============================================================"
echo "  ✓ PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Model weights:    best_gnn_model.pth"
echo "  Training log:     gnn_training_log_*.txt"
echo "  Results JSON:     gnn_training_results_*.json"
echo ""
