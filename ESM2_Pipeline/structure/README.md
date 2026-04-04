# Structure-Based MHC-Peptide Binding Prediction Pipeline (Physics-Enhanced GNN)

## Overview
This pipeline generates physics-enhanced graphs representing 3D protein structures of MHC-Peptide complexes via template threading and trains a Graph Neural Network (GNN) to predict binding affinity (IC50).

## Requirements
- **Python Environment**: Ensure you are using your `.venv` or conda environment with PyTorch and PyTorch Geometric installed (`pip install torch-geometric torch-scatter torch-sparse`).
- **Templates**: PDB format templates for MHC structures.

## Complete Workflow

### Step 1: Download Templates
Download the required base PDB templates from the RCSB database.
```bash
python download_templates.py
```
*Output: Saves 1HHI, 1A1M, 3VCL, 1XR8 templates into `templates/`.*

### Step 2: Generate Graphs (Template Threading)
Map the dataset sequences to the 3D coordinates based on the templates, and extract physicochemical properties into `.pt` graph files.
```bash
python graph_generator.py
```
*Output: Generates `.pt` files containing node and edge features inside `data/graphs/ensemble/`.*

*(Optional) Clean up corrupted or old format graphs:*
```bash
python clean_data.py
```

### Step 3: Train the GNN
Train the physics-enhanced multi-template GNN using the generated graphs. This script has an auto-restart watchdog to handle OOM/timeout crashes on A100s.
```bash
./train_gnn_model.sh
```
*Alternatively, run training directly: `python train_gnn_model.py`*

**Results:** Saves model checkpoints in `checkpoints/` and logs in `logs/`.

### Step 4: Sequence-Side Merge (Ensemble)
Once the structural model finishes training, you can combine its structured intelligence with your sequence models (like ESM2) using the multimodal integration script.

From the root pipeline directory:
```bash
cd ..
./extract_and_ensemble.sh
# or 
python train_ensemble.py
```

## Monitoring Progress
- Check graph generation: `ls -1 data/graphs/ensemble/mhc | wc -l`
- Check GNN training outputs via the active log file: `tail -f training_monitor.log` or by querying the latest training log in `logs/`.
