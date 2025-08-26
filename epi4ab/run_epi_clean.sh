#!/bin/bash

echo "Running Epi4Ab with ALL warnings suppressed..."

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate epi4ab

# Ultimate warning suppression bash script
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRANSFORMERS_VERBOSITY=error
export HF_DATASETS_OFFLINE=1

# Run the silent Python script that suppresses all warnings
python3 epi_graph_silent.py
