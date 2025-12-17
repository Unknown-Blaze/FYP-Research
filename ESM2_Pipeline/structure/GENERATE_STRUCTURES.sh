#!/bin/bash
# ============================================================================
# COMPLETE STRUCTURE GENERATION PIPELINE
# Generates BOTH MHC and Peptide structures using LocalColabFold
# ============================================================================

set -e

echo ""
echo "============================================================"
echo "  MHC & PEPTIDE MONOMER STRUCTURE GENERATION"
echo "============================================================"
echo ""

# Paths
STRUCTURE_DIR="/home3/rudrapra001/FYP-Research/ESM2_Pipeline/structure"
OUTPUT_DIR="$STRUCTURE_DIR/data/generated_pdbs"
COLABFOLD_BIN="$STRUCTURE_DIR/localcolabfold/colabfold-conda/bin/colabfold_batch"
COLABFOLD_PY="$STRUCTURE_DIR/localcolabfold/colabfold-conda/bin/python"
COLABFOLD_LIB="$STRUCTURE_DIR/localcolabfold/colabfold-conda/lib"
PYTHON_BIN="$COLABFOLD_PY"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# Ensure bundled libstdc++ is used to avoid GLIBCXX errors
export LD_LIBRARY_PATH="$COLABFOLD_LIB:${LD_LIBRARY_PATH:-}"

cd "$STRUCTURE_DIR"

# Modes (override via env vars)
# MHC: ACCURACY first (MSA) by default, Peptides: SPEED first (ColabFold single_sequence)
# Options:
#   MHC_MODE:   accurate|fast           (default: accurate)
#   PEP_ENGINE: colabfold|esmfold       (default: colabfold)
#   PEP_MSA:    mmseqs2|single_sequence (default: single_sequence for speed)
#   RECYCLES:   integer recycles count  (default: 1 for speed, use 0 for ultra-fast peptides)
MHC_MODE=${MHC_MODE:-accurate}
PEP_ENGINE=${PEP_ENGINE:-colabfold}
PEP_MSA=${PEP_MSA:-single_sequence}
RECYCLES=${RECYCLES:-1}

# Check if LocalColabFold is installed
if [ ! -f "$COLABFOLD_BIN" ]; then
    echo "❌ LocalColabFold not found!"
    echo "   Expected: $COLABFOLD_BIN"
    echo ""
    echo "Install with:"
    echo "  bash install_colabbatch_linux.sh"
    exit 1
fi

echo "✓ LocalColabFold found"
echo ""

# Step 1: Prepare unique sequences (if not done)
echo "[Step 1/3] Preparing unique FASTA files..."
echo "------------------------------------------------------------"

if [ ! -f "mhc_unique.fasta" ] || [ ! -f "peptide_unique.fasta" ]; then
    echo "  Running prepare_esmfold_inputs.py..."
    CUDA_VISIBLE_DEVICES=0 python prepare_esmfold_inputs.py
else
    MHC_COUNT=$(grep -c '^>' mhc_unique.fasta)
    PEP_COUNT=$(grep -c '^>' peptide_unique.fasta)
    echo "  ✓ FASTA files already exist"
    echo "    - MHC sequences: $MHC_COUNT"
    echo "    - Peptide sequences: $PEP_COUNT"
fi

echo ""

# Step 2: Generate MHC structures
echo "[Step 2/3] Generating MHC Monomer Structures..."
echo "------------------------------------------------------------"

mkdir -p "$OUTPUT_DIR/mhc"

if ls "$OUTPUT_DIR/mhc"/*rank_001*.pdb 1> /dev/null 2>&1; then
    EXISTING_MHC=$(ls "$OUTPUT_DIR/mhc"/*rank_001*.pdb 2>/dev/null | wc -l)
    TOTAL_MHC=$(grep -c '^>' mhc_unique.fasta)
    
    echo "  Found $EXISTING_MHC / $TOTAL_MHC MHC structures"
    
    if [ "$EXISTING_MHC" -eq "$TOTAL_MHC" ]; then
        echo "  ✓ All MHC structures already generated!"
        echo "  Skipping MHC generation..."
    else
        echo "  ⚠ Incomplete. Generating remaining structures..."
        if [ "$MHC_MODE" = "accurate" ]; then
            echo "  Using ACCURATE mode: MSA + recycles=$RECYCLES, no relax"
            $COLABFOLD_BIN \
                --num-models 1 \
                --num-recycle "$RECYCLES" \
                mhc_unique.fasta \
                "$OUTPUT_DIR/mhc"
        else
            echo "  Using FAST mode: single_sequence + recycles=$RECYCLES, no relax"
            $COLABFOLD_BIN \
                --msa-mode single_sequence \
                --num-models 1 \
                --num-recycle "$RECYCLES" \
                mhc_unique.fasta \
                "$OUTPUT_DIR/mhc"
        fi
    fi
else
    echo "  Starting MHC structure generation (monomers)..."
    echo ""
    if [ "$MHC_MODE" = "accurate" ]; then
        echo "  Using ACCURATE mode: MSA + recycles=$RECYCLES, no relax"
        $COLABFOLD_BIN \
            --num-models 1 \
            --num-recycle "$RECYCLES" \
            mhc_unique.fasta \
            "$OUTPUT_DIR/mhc"
    else
        echo "  Using FAST mode: single_sequence + recycles=$RECYCLES, no relax"
        $COLABFOLD_BIN \
            --msa-mode single_sequence \
            --num-models 1 \
            --num-recycle "$RECYCLES" \
            mhc_unique.fasta \
            "$OUTPUT_DIR/mhc"
    fi
    echo "  ✓ MHC generation submitted!"
fi

echo ""

# Step 3: Generate Peptide structures
echo "[Step 3/3] Generating Peptide Monomer Structures..."
echo "------------------------------------------------------------"

mkdir -p "$OUTPUT_DIR/peptide"

TOTAL_PEPTIDES=$(grep -c '^>' peptide_unique.fasta)

if ls "$OUTPUT_DIR/peptide"/*rank_001*.pdb 1> /dev/null 2>&1; then
    EXISTING_PEP=$(ls "$OUTPUT_DIR/peptide"/*rank_001*.pdb 2>/dev/null | wc -l)
    echo "  Found $EXISTING_PEP / $TOTAL_PEPTIDES peptide structures"
    
    if [ "$EXISTING_PEP" -eq "$TOTAL_PEPTIDES" ]; then
        echo "  ✓ All peptide structures already generated!"
        echo "  Skipping peptide generation..."
    else
        echo "  ⚠ Incomplete ($EXISTING_PEP/$TOTAL_PEPTIDES)"
        if [ "$PEP_ENGINE" = "esmfold" ]; then
            echo "  Using ESMFold for peptides (fast monomer folding)"
            CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" generate_peptides_esmfold.py --input peptide_unique.fasta --out "$OUTPUT_DIR/peptide"
        else
            echo "  Using ColabFold for peptides: msa-mode=$PEP_MSA recycles=$RECYCLES"
            if [ "$PEP_MSA" = "single_sequence" ]; then
                $COLABFOLD_BIN \
                    --msa-mode single_sequence \
                    --num-models 1 \
                    --num-recycle "$RECYCLES" \
                    peptide_unique.fasta \
                    "$OUTPUT_DIR/peptide"
            else
                $COLABFOLD_BIN \
                    --num-models 1 \
                    --num-recycle "$RECYCLES" \
                    peptide_unique.fasta \
                    "$OUTPUT_DIR/peptide"
            fi
        fi
    fi
else
    echo "  Peptide structures not found yet. Total: $TOTAL_PEPTIDES"
    if [ "$PEP_ENGINE" = "esmfold" ]; then
        echo "  Using ESMFold for peptides (fast monomer folding)"
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" generate_peptides_esmfold.py --input peptide_unique.fasta --out "$OUTPUT_DIR/peptide"
    else
        echo "  Using ColabFold for peptides: msa-mode=$PEP_MSA recycles=$RECYCLES"
        if [ "$PEP_MSA" = "single_sequence" ]; then
            $COLABFOLD_BIN \
                --msa-mode single_sequence \
                --num-models 1 \
                --num-recycle "$RECYCLES" \
                peptide_unique.fasta \
                "$OUTPUT_DIR/peptide"
        else
            $COLABFOLD_BIN \
                --num-models 1 \
                --num-recycle "$RECYCLES" \
                peptide_unique.fasta \
                "$OUTPUT_DIR/peptide"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "  ✓ STRUCTURE GENERATION COMPLETE!"
echo "============================================================"
echo ""
echo "MHC structures:      $OUTPUT_DIR/mhc/"
echo "Peptide structures:  $OUTPUT_DIR/peptide/"
echo ""
echo "Next step: Train the model"
echo "  python train_structure_model.py"
echo ""
