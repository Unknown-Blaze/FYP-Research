#!/bin/bash

echo "🚀 Epi4Ab Master Control Script"
echo "================================"

# Function to display help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  train           - Run 3-fold cross-validation training (research mode)"
    echo "  model           - Generate model.pt file (production mode)"
    echo "  test            - Test existing model on unseen data"
    echo "  optuna          - Run hyperparameter optimization"
    echo "  diagnose        - Run system diagnostics"
    echo "  cdrs            - Generate CDR sequences"
    echo "  download        - Download PDB files"
    echo "  notebook        - Launch Jupyter notebook"
    echo "  help            - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./master_run.sh train     # Standard training"
    echo "  ./master_run.sh model     # Generate model.pt"
    echo "  ./master_run.sh diagnose  # Check system"
    echo ""
}

# Function to check environment
check_environment() {
    echo "🔍 Checking environment..."
    
    # Initialize conda if needed
    if ! command -v conda &> /dev/null; then
        echo "⚠️ Conda not found, attempting to initialize..."
        source ~/miniconda3/etc/profile.d/conda.sh
    fi
    
    # Check if in conda environment
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate epi4ab 2>/dev/null
    
    # Check key files
    if [[ ! -f "parameters.txt" ]]; then
        echo "❌ parameters.txt not found"
        return 1
    fi
    
    # Check Python dependencies
    if ! python -c "import torch; import torch_geometric; import pandas; import numpy; print('✅ Key dependencies available')" 2>/dev/null; then
        echo "❌ Missing Python dependencies"
        return 1
    fi
    
    echo "✅ Environment check passed"
    return 0
}

# Function to run diagnostics
run_diagnostics() {
    echo "🔍 Running system diagnostics..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate epi4ab
    python diagnose_paths.py --processed_dir "data/nodes_edges/" --pdb_dir "data/raw_pdb/" --sabdab_summary "data/sabdab_summary_master.tsv"
}

# Function to generate CDRs
generate_cdrs() {
    echo "🧬 Generating CDR sequences..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate epi4ab
    python generate_cdrs.py --processed_dir "data/nodes_edges/" --pdb_dir "data/raw_pdb/" --sabdab_summary "data/sabdab_summary_master.tsv"
}

# Function to download PDBs
download_pdbs() {
    echo "📥 Downloading PDB files..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate epi4ab
    python download_pdbs.py
}

# Function to run training
run_training() {
    echo "🏋️ Starting training (3-fold cross-validation)..."
    if ! check_environment; then
        echo "⚠️ Environment check failed, but proceeding with training..."
    fi
    bash run_epi_clean.sh
}

# Function to generate model
generate_model() {
    echo "🏗️ Generating production model..."
    if ! check_environment; then
        echo "⚠️ Environment check failed, but proceeding with model generation..."
    fi
    bash generate_model.sh
}

# Function to test model
test_model() {
    echo "🧪 Testing model..."
    if ! check_environment; then
        echo "⚠️ Environment check failed, but proceeding with testing..."
    fi
    bash run_model.sh
}

# Function to run optuna
run_optuna() {
    echo "🎯 Running hyperparameter optimization..."
    if ! check_environment; then
        echo "⚠️ Environment check failed, but proceeding with optuna..."
    fi
    bash run_optuna.sh
}

# Function to launch notebook
launch_notebook() {
    echo "📓 Launching Jupyter notebook..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate epi4ab
    jupyter notebook epi_graph.ipynb
}

# Main script logic
case "${1:-help}" in
    train)
        run_training
        ;;
    model)
        generate_model
        ;;
    test)
        test_model
        ;;
    optuna)
        run_optuna
        ;;
    diagnose)
        run_diagnostics
        ;;
    cdrs)
        generate_cdrs
        ;;
    download)
        download_pdbs
        ;;
    notebook)
        launch_notebook
        ;;
    help|*)
        show_help
        ;;
esac

echo "✅ Script completed"
