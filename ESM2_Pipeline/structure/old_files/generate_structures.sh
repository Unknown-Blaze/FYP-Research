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
# STEP 1: PREPARE FASTA FILES
# ------------------------------------------------------------------
echo "[Step 1/3] Preparing unique FASTA files..."

if [ ! -f "mhc_unique.fasta" ] || [ ! -f "peptide_unique.fasta" ]; then
    echo "  Running prepare_esmfold_inputs.py..."
    # Ensure we use the ColabFold python or system python
    "$PYTHON_BIN" prepare_esmfold_inputs.py
else
    MHC_COUNT=$(grep -c '^>' mhc_unique.fasta)
    PEP_COUNT=$(grep -c '^>' peptide_unique.fasta)
    echo "  ✓ FASTA files exist"
    echo "    - MHC sequences: $MHC_COUNT"
    echo "    - Peptide sequences: $PEP_COUNT"
fi
echo ""

# ------------------------------------------------------------------
# STEP 2: GENERATE MHC STRUCTURES
# ------------------------------------------------------------------
echo "[Step 2/3] Generating MHC Structures..."
echo "Settings: Recycles=$MHC_RECYCLES | MSA=$MHC_MSA | AMBER=ON"
echo "------------------------------------------------------------"

mkdir -p "$OUTPUT_DIR/mhc"

# Check if we need to run
EXISTING_MHC=$(ls "$OUTPUT_DIR/mhc"/*rank_001*.pdb 2>/dev/null | wc -l || echo 0)
TOTAL_MHC=$(grep -c '^>' mhc_unique.fasta)

if [ "$EXISTING_MHC" -ge "$TOTAL_MHC" ]; then
    echo "  ✓ All MHC structures already generated."
else
    echo "  Generating remaining MHC structures..."
    
    # CRITICAL GNN FLAGS:
    # --amber: Relaxes atoms to prevent overlaps (exploding gradients in GNN)
    # --num-recycle 3: Ensures side-chains pack correctly
    
    $COLABFOLD_BIN \
        --num-models 1 \
        --num-recycle "$MHC_RECYCLES" \
        --msa-mode "$MHC_MSA" \
        --amber \
        $USE_GPU_RELAX \
        mhc_unique.fasta \
        "$OUTPUT_DIR/mhc"
fi
echo ""

# ------------------------------------------------------------------
# STEP 3: GENERATE PEPTIDE STRUCTURES
# ------------------------------------------------------------------
echo "[Step 3/3] Generating Peptide Structures..."
echo "Settings: Recycles=$PEP_RECYCLES | MSA=$PEP_MSA | AMBER=ON"
echo "------------------------------------------------------------"

mkdir -p "$OUTPUT_DIR/peptide"

# Check if we need to run
EXISTING_PEP=$(ls "$OUTPUT_DIR/peptide"/*rank_001*.pdb 2>/dev/null | wc -l || echo 0)
TOTAL_PEP=$(grep -c '^>' peptide_unique.fasta)

if [ "$EXISTING_PEP" -ge "$TOTAL_PEP" ]; then
    echo "  ✓ All peptide structures already generated."
else
    echo "  Generating remaining peptide structures..."
    
    # Even for peptides, we use AMBER to ensure bond lengths are physical
    # We use single_sequence to speed this up massively
    
    $COLABFOLD_BIN \
        --num-models 1 \
        --num-recycle "$PEP_RECYCLES" \
        --msa-mode "$PEP_MSA" \
        --amber \
        $USE_GPU_RELAX \
        peptide_unique.fasta \
        "$OUTPUT_DIR/peptide"
fi

echo ""
echo "============================================================"
echo "  ✓ PIPELINE COMPLETE"
echo "============================================================"
echo "Data ready for GNN conversion."