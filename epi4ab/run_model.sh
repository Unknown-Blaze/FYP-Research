#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

# --- Your existing script variables ---
MODEL_NAME="2025-08-19_GNNResNet_13" # Make sure this is the correct model name
USE_REGION=false
USE_RELAXED=false
PLOT_NETWORK=false
INPUT_AB_FEATURE=true # Use when test same VH/VL Ab family

MODEL_FOLDER=$DIRECTORY_OUTPUT/$MODEL_NAME
PARAMS=""
[ "$USE_REGION" = true ] && PARAMS+="--use_region " || PARAMS+=""
[ "$USE_RELAXED" = true ] && PARAMS+="--use_relaxed " || PARAMS+=""
[ "$PLOT_NETWORK" = true ] && PARAMS+="--plot_network " || PARAMS+=""
[ "$INPUT_AB_FEATURE" = true ] && PARAMS+="--ab_feature_input " || PARAMS+=""

# =================================================================
# THE FIX: Add the current directory to Python's import path.
# This allows Python to find your 'use_model' and 'interface_prediction' modules.
# =================================================================
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Now, run the python script as before
python epi_model.py $DIRECTORY_DATA $DIRECTORY_PROCESSED_DATA $DIRECTORY_UNSEEN_PDB_LIST $MODEL_FOLDER $DIRECTORY_UNSEEN_OUTPUT $PARAMS
