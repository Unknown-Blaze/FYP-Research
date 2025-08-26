#!/bin/bash

echo "Generating model.pt file for inference..."

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate epi4ab

# Backup original parameters
cp parameters.txt parameters_backup.txt

# Use model generation parameters
cp parameters_model_gen.txt parameters.txt

# Ultimate warning suppression
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_OFFLINE=1

echo "Training with train_all=yes to generate model.pt..."
python3 epi_graph_silent.py

# Restore original parameters
cp parameters_backup.txt parameters.txt
rm parameters_backup.txt

echo "Model generation complete. Check output folder for model.pt file."
