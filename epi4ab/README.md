# Epi4Ab: Prediction of conformational epitopes for specific antibody VH/VL families and CDRs sequences

Advanced epitope prediction using Graph Neural Networks (GNNs) with antibody-specific features.

## 🆕 New Features & Fixes (Aug 20, 2025)

### ✅ **Complete Model Pipeline**
- **`epi_model.py`**: Fixed inference script with corrected import paths
- **`epi_optuna.py`**: Hyperparameter optimization with Optuna
- **`generate_model.sh`**: Automated model.pt generation for production
- **`master_run.sh`**: One-stop script for all operations

### ✅ **Path & Configuration Fixes**
- **Fixed import paths**: Updated from old `use_model.*` to current `interface_prediction.*`
- **AlphaFold support**: Added proper AF file path configurations
- **Environment management**: Automated conda environment activation

### ✅ **Enhanced Training Modes**
- **Research Mode** (`train_all=no`): 3-fold cross-validation for validation
- **Production Mode** (`train_all=yes`): Full model training with model.pt output
- **Debug Mode**: Quick 2-epoch testing for development

## Quick Start

1. **Environment Setup:**
   ```bash
   ./setup_environment.sh
   ```

2. **Run Training (3-fold cross-validation):**
   ```bash
   ./master_run.sh train
   ```

3. **Generate Production Model:**
   ```bash
   ./master_run.sh model
   ```

4. **Test Model:**
   ```bash
   ./master_run.sh test
   ```

5. **System Diagnosis:**
   ```bash
   ./master_run.sh diagnose
   ```

### 🔧 **Individual Scripts**
```bash
# Warning-free training
bash run_epi_clean.sh

# Standard training with logs  
bash run_epi.sh

# Generate model.pt for inference
bash generate_model.sh

# Test model on unseen data
bash run_model.sh

# Hyperparameter optimization
bash run_optuna.sh
```

### 📓 **Interactive Development**
```bash
# Launch enhanced Jupyter notebook
./master_run.sh notebook
# OR
jupyter notebook epi_graph.ipynb
```

## Configuration Files

- **`.env`**: Directory paths configuration
- **`parameters.txt`**: Training hyperparameters (model, epochs, batch size, etc.)
- **`parameters_ab.json`**: Antibody-specific features for prediction

## Key Features

### CDR Sequence Generation
```bash
# Generate CDR sequences for all processed PDBs
python generate_cdrs.py --processed_dir "data/nodes_edges/" --pdb_dir "data/raw_pdb/" --sabdab_summary "data/sabdab_summary_master.tsv"

# Diagnose path issues if CDR generation fails
python diagnose_paths.py --processed_dir "data/nodes_edges/" --pdb_dir "data/raw_pdb/" --sabdab_summary "data/sabdab_summary_master.tsv"
```

### Download PDB Files
```bash
# Download all required PDB files automatically
python download_pdbs.py
```

## Training Results

The training produces comprehensive outputs:
- **Loss plots**: Visual training progress
- **Evaluation metrics**: Detailed performance statistics
- **Test records**: Per-PDB prediction results
- **Model checkpoints**: Saved models for inference
- **Execution logs**: Complete training history

## Dataset Information

- **Training PDBs**: 279 structures
- **Test PDBs**: 69 structures  
- **Validation PDBs**: 93 structures
- **Features**: Graph-based molecular representation with CDR sequences
- **Task**: Binary classification (interface vs non-interface residues)

## Model Architecture

- **Base Model**: GNNResNet with Chebyshev convolution blocks
- **Input Features**: ESM2 protein embeddings + structural features + CDR sequences
- **Training**: 3-fold cross-validation, 50 epochs per fold
- **Optimization**: Adam optimizer with learning rate 0.0001

## Recent Improvements

- ✅ Complete ROC-AUC error handling for edge cases
- ✅ Warning suppression for clean training output
- ✅ CDR sequence integration with ANARCI
- ✅ Comprehensive path configuration
- ✅ Interactive Jupyter notebook interface
- ✅ Automated PDB downloading utilities

## File Structure

```
Epi4Ab/
├── data/
│   ├── nodes_edges/          # Processed graph data
│   ├── raw_pdb/             # Original PDB files
│   └── sabdab_summary_*.tsv # Antibody metadata
├── interface_prediction/     # Core prediction modules
├── output/                  # Training results
├── lists/                   # Train/test PDB lists
├── epi_graph.py            # Main training script
├── epi_graph.ipynb         # Interactive training notebook
├── run_epi_clean.sh        # Warning-free training
├── generate_cdrs.py        # CDR sequence extraction
└── requirements.txt        # Python dependencies
```

Last updated: August 19, 2025
