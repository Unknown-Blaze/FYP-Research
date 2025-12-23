#!/bin/bash
# ============================================================================
# COMPLETE STRUCTURE GENERATION PIPELINE (Enhanced for GNNs)
# Generates MHC and Peptide structures with AMBER relaxation
# ============================================================================

set -e

echo ""
echo "============================================================"
echo "  MHC & PEPTIDE STRUCTURE GENERATION (GNN-Ready)"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Base Directories
STRUCTURE_DIR="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure"
OUTPUT_DIR="$STRUCTURE_DIR/data/generated_pdbs"

# ColabFold Paths
COLABFOLD_BASE="$STRUCTURE_DIR/localcolabfold/colabfold-conda"
COLABFOLD_BIN="$COLABFOLD_BASE/bin/colabfold_batch"
PYTHON_BIN="$COLABFOLD_BASE/bin/python"

# Ensure bundled libstdc++ is used (prevents GLIBCXX errors)
export LD_LIBRARY_PATH="$COLABFOLD_BASE/lib:${LD_LIBRARY_PATH:-}"

cd "$STRUCTURE_DIR"

# ------------------------------------------------------------------
# RUN MODES & SETTINGS
# ------------------------------------------------------------------

# Defaults (Can be overridden by environment variables)
# MHC: Needs high accuracy for side-chains (binding specificity)
MHC_RECYCLES=${MHC_RECYCLES:-3}      # 3 is standard for high quality
MHC_MSA=${MHC_MSA:-mmseqs2_uniref_env} # Full MSA for MHCs

# Peptide: Monomers are flexible/disordered; keep it fast
PEP_MSA=${PEP_MSA:-single_sequence}  # MSAs don't help short peptides much
PEP_RECYCLES=${PEP_RECYCLES:-1}      # 1 is enough for a simple helix/strand prediction

# ------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# ------------------------------------------------------------------

if [ ! -f "$COLABFOLD_BIN" ]; then
    echo "❌ LocalColabFold binary not found at:"
    echo "   $COLABFOLD_BIN"
    exit 1
fi

echo "✓ LocalColabFold found"

# Check for GPU (NVIDIA SMI) to safely use --use-gpu-relax
USE_GPU_RELAX=""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected (enabling GPU-accelerated relaxation)"
    USE_GPU_RELAX="--use-gpu-relax"
else
    echo "⚠ No GPU detected. Relaxation will be slow (CPU mode)."
fi
echo ""

# ------------------------------------------------------------------
# STEP 1: PREPARE PAIRED INPUT
# ------------------------------------------------------------------
# Input format for ColabFold complex: >ID\nSEQUENCE1:SEQUENCE2
echo "[Step 1] Preparing Complex Fasta..."

# Assuming you have a python script that creates 'complexes.fasta'
# where sequences look like: >ID_001\nMHC_SEQ:PEP_SEQ
if [ ! -f "complexes.fasta" ]; then
    echo "❌ Please generate complexes.fasta with 'MHC:PEP' format first."
    exit 1
fi

# ------------------------------------------------------------------
# STEP 2: GENERATE COMPLEX STRUCTURES
# ------------------------------------------------------------------
echo "[Step 2] Generating Bound Complexes..."

mkdir -p "$OUTPUT_DIR/complexes"

# ColabFold automatically handles the interaction when ':' is used
$COLABFOLD_BIN \
    --num-models 1 \
    --num-recycle 3 \
    --msa-mode "mmseqs2_uniref_env" \
    --amber \
    $USE_GPU_RELAX \
    complexes.fasta \
    "$OUTPUT_DIR/complexes"

echo "✓ Complexes generated. PDBs will contain Chain A (MHC) and Chain B (Peptide)."

echo ""
echo "============================================================"
echo "  ✓ PIPELINE COMPLETE"
echo "============================================================"
echo "Data ready for GNN conversion."