#!/bin/bash
# ==============================================
# FULL INTEGRATED PIPELINE: Sequence + Structure
# ==============================================

set -e

echo "============================================"
echo "FYP INTEGRATED MHC-PEPTIDE BINDING PIPELINE"
echo "============================================"

PROJECT_DIR="$PWD"
CUDA_DEVICE=0

# Step 1: Sequence Pipeline (Already Done)
echo ""
echo "[1/4] Sequence Pipeline (Already Trained)"
echo "✓ ESMplusplus model trained"
echo "✓ Warm-start cross-attention converged"
echo "✓ Current best RMSE: 0.73-0.75"

# Step 2: Generate Structures
echo ""
echo "[2/4] Generating 3D Structures with AlphaFold..."
mkdir -p structure/data/{structures,graphs,embeddings}
cd structure
python run_alphafold.py
echo "✓ Structure generation complete"

# Step 3: Build Graphs
echo ""
echo "[3/4] Building Graph Representations..."
python build_graphs.py
echo "✓ Graph building complete"

# Step 4: Train Integrated Model
echo ""
echo "[4/4] Training Integrated Sequence + Structure Model..."
cd ../multimodal_pipeline
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python integrated_model.py \
    --config ../config_multimodal.yaml \
    --pilot  # Remove for full training

echo ""
echo "============================================"
echo "PIPELINE COMPLETE!"
echo "============================================"
echo "Results saved to: ./multimodal_pipeline/checkpoints/"
echo "Compare to sequence baseline: RMSE 0.73-0.75"
echo "Expected improvement: +5-10% (RMSE 0.68-0.71)"
