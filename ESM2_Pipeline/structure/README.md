# Structure-Based MHC-Peptide Binding Prediction Pipeline

## Overview
This pipeline generates 3D protein structures using LocalColabFold (AlphaFold2) and trains a binding affinity prediction model using structural features.

## Requirements
- **GPU**: CUDA-enabled GPU (recommended for fast generation)
- **LocalColabFold**: Already installed at `./localcolabfold/`
- **Python Environment**: `esm` conda environment with PyTorch

## Files in This Directory

### Essential Files (DO NOT DELETE)
- `prepare_esmfold_inputs.py` - Extracts unique MHC and peptide sequences from dataset
- `GENERATE_STRUCTURES.sh` - Main script to generate all PDB structures
- `train_structure_model.py` - Training script for structure-based model
- `mhc_unique.fasta` - 58 unique MHC sequences
- `peptide_unique.fasta` - 24,483 unique peptide sequences
- `row_to_pdb_map.json` - Maps dataset rows to PDB file IDs
- `localcolabfold/` - LocalColabFold installation directory
- `data/generated_pdbs/` - Output directory for PDB files

### Generated Files (Can be regenerated)
- `data/generated_pdbs/mhc/` - MHC structure files
- `data/generated_pdbs/peptide/` - Peptide structure files
- `best_structure_model.pth` - Trained model weights

## Complete Workflow

### Step 1: Prepare Input Files
```bash
# Extract unique sequences from dataset (if not already done)
python prepare_esmfold_inputs.py
```

**Output:**
- `mhc_unique.fasta` - 58 MHC sequences
- `peptide_unique.fasta` - 24,483 peptide sequences
- `row_to_pdb_map.json` - Mapping file

### Step 2: Generate Structures
```bash
# Make script executable
chmod +x GENERATE_STRUCTURES.sh

# Run structure generation
./GENERATE_STRUCTURES.sh
```

**What This Does:**
1. Checks if LocalColabFold is installed
2. Generates MHC structures (~10 hours for 58 sequences)
3. Prompts for peptide generation (200-400 hours for all 24,483)

**Time Estimates:**
- **MHC structures**: ~10 hours (58 sequences × 10 min/sequence)
- **Peptide structures**: 200-400 hours (24,483 sequences × 0.5-1 min/sequence)

**Tips for Long Runs:**
```bash
# Run in background with nohup
nohup ./GENERATE_STRUCTURES.sh > generation.log 2>&1 &

# Or use screen/tmux
screen -S structure_gen
./GENERATE_STRUCTURES.sh
# Ctrl+A, D to detach

# Monitor progress
watch -n 60 'ls data/generated_pdbs/mhc/*rank_001*.pdb | wc -l'
watch -n 60 'ls data/generated_pdbs/peptide/*rank_001*.pdb | wc -l'
```

### Step 3: Generate Subset (Optional for Testing)
If you want to test with a subset first:

```bash
# Generate only first 100 peptides
head -200 peptide_unique.fasta > peptide_subset.fasta

localcolabfold/colabfold-conda/bin/colabfold_batch \
    --num-models 1 \
    --num-recycle 3 \
    peptide_subset.fasta \
    data/generated_pdbs/peptide
```

### Step 4: Train the Model
Once structures are generated:

```bash
# Activate environment
conda activate esm

# Train model
python train_structure_model.py
```

**What This Does:**
1. Loads dataset and PDB mapping
2. Extracts structural features from PDB files:
   - Number of residues
   - Radius of gyration
   - Maximum distance
   - Contact density (8Å cutoff)
3. Trains a neural network to predict IC50 from structure features
4. Saves best model to `best_structure_model.pth`

## Structure Generation Details

### ColabFold Parameters
- `--num-models 1`: Generate 1 model per sequence (faster)
- `--num-recycle 3`: 3 recycling iterations (good quality/speed tradeoff)
- `--use-gpu-relax`: Use GPU for structure relaxation (faster)

### Output Files
For each sequence, ColabFold generates:
- `*_unrelaxed_rank_001_*.pdb` - Best predicted structure
- `*_unrelaxed_rank_002_*.pdb` - Second best (if using multiple models)
- `*_relaxed_rank_001_*.pdb` - Relaxed structure (if relaxation enabled)
- `*_scores_*.json` - Prediction confidence scores
- `*_pae_*.json` - Predicted aligned error

**We use only `rank_001` files** (best predictions) for feature extraction.

## Monitoring Progress

### Check Generation Status
```bash
# Count MHC structures
ls data/generated_pdbs/mhc/*rank_001*.pdb | wc -l

# Count peptide structures
ls data/generated_pdbs/peptide/*rank_001*.pdb | wc -l

# Check disk usage
du -sh data/generated_pdbs/
```

### Expected Disk Usage
- **MHC**: ~500 MB (58 structures × ~8 MB each)
- **Peptides**: ~200 GB (24,483 structures × ~8 MB each)

## Troubleshooting

### LocalColabFold Not Found
```bash
# Reinstall LocalColabFold
bash install_colabbatch_linux.sh
```

### Out of GPU Memory
```bash
# Reduce batch size by generating fewer sequences at once
# Split FASTA file into smaller chunks
split -l 1000 peptide_unique.fasta peptide_chunk_
```

### Slow Generation
- **Check GPU usage**: `nvidia-smi`
- **Reduce recycles**: Use `--num-recycle 1` instead of 3
- **Disable relaxation**: Remove `--use-gpu-relax` flag

### Missing Dependencies for Training
```bash
conda activate esm
pip install scipy scikit-learn
```

## Expected Results

### Baseline (Sequence-Only Model)
- **Test RMSE**: 0.73-0.75
- **Test R²**: 0.55-0.60

### Structure-Based Model Goal
- **Target RMSE**: 0.65-0.70 (5-10% improvement)
- **Target R²**: 0.60-0.65

### Why Structure Helps
- **Shape complementarity**: Radius of gyration, max distance
- **Surface properties**: Contact density
- **Binding pocket geometry**: Structural features capture binding interface

## Quick Commands

```bash
# 1. Prepare sequences
python prepare_esmfold_inputs.py

# 2. Generate MHC structures only
./GENERATE_STRUCTURES.sh
# (Select to skip peptides when prompted)

# 3. Generate peptide subset for testing
head -200 peptide_unique.fasta > test_peptides.fasta
localcolabfold/colabfold-conda/bin/colabfold_batch \
    --num-models 1 --num-recycle 3 \
    test_peptides.fasta data/generated_pdbs/peptide

# 4. Train model
python train_structure_model.py
```

## Contact
For issues or questions about the structure pipeline, check:
- LocalColabFold docs: https://github.com/YoshitakaMo/localcolabfold
- AlphaFold paper: https://www.nature.com/articles/s41586-021-03819-2
