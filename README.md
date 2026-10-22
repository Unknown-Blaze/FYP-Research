# ESM2 Multimodal Pipeline

MHC-peptide binding affinity prediction that combines LoRA fine-tuned ESM-2 650M sequence embeddings with AlphaFold-derived structure, reaching PCC 0.859 against 0.847 for the best retrained state-of-the-art baseline; submitted to ICLR 2027.

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ESM-2](https://img.shields.io/badge/ESM--2-650M-0467DF?style=flat-square)](https://github.com/facebookresearch/esm)
[![Status](https://img.shields.io/badge/ICLR%202027-submitted-6f42c1?style=flat-square)](#)

<img src="docs/results.png" alt="Bar chart of Pearson correlation: 0.859 for this model against 0.847 for the best retrained state-of-the-art baseline" width="100%">

This repository is organized for readers who want to reproduce a three-stage MHC-peptide binding workflow:

1. Sequence model training in [sequence](sequence)
2. Structure graph generation and GNN training in [structure](structure)
3. Fusion/ensemble stacking at repository root

<img src="docs/pipeline.png" alt="Pipeline: the IEDB dataset feeds an ESM2 sequence model and a structure GNN, whose predictions are combined by ensemble stacking" width="100%">

## Repository Layout

* [sequence](sequence): ESM2 fine-tuning and evaluation
* [structure](structure): template download, graph generation, GNN training
* root scripts: [extract_predictions.py](extract_predictions.py), [train_ensemble.py](train_ensemble.py), [extract_and_ensemble.sh](extract_and_ensemble.sh)
* visualizations: [training_visualizations/scripts](training_visualizations/scripts), outputs in `training_visualizations/figures` and `training_visualizations/reports`

## Environment Setup

Use one Python 3.8.10 virtual environment for the entire pipeline. Create it from the root so the same environment is used for sequence, structure, and ensemble steps.

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.8` is not the exact interpreter on your machine, use the full path to your Python 3.8.10 binary when creating the venv.

## Data Availability

Large generated assets are not versioned in this repository.

* The main dataset file used by the pipeline is [iedb_dataset.csv](iedb_dataset.csv).
* Structure-derived artifacts (for example generated graphs and intermediate structure outputs under [structure/data](structure/data)) are stored on BioNet internal storage for lab use.
* If you are an external reviewer and need those assets, please request access from the repository owner.

## End-to-End Run Order

Run from repository root unless otherwise noted.

1. Sequence training/evaluation

```bash
bash sequence/train.sh
bash sequence/test.sh
```

2. Structure pipeline

```bash
cd structure
python download_templates.py
python graph_generator.py
python train_gnn_model.py
# optional watchdog wrapper:
# bash train_gnn_model.sh
cd ..
```

The structure step runs in the same Python environment as the sequence step; it does not require a separate conda environment in the current repo layout.

3. Ensemble/fusion

```bash
bash extract_and_ensemble.sh
```

This executes [extract_predictions.py](extract_predictions.py) followed by [train_ensemble.py](train_ensemble.py). Legacy wrapper [run_full_pipeline.sh](run_full_pipeline.sh) delegates to the same flow.

## Visualization Scripts

Scripts:

* [training_visualizations/scripts/generate_requested_visualizations.py](training_visualizations/scripts/generate_requested_visualizations.py)
* [training_visualizations/scripts/visualize_training_logs.py](training_visualizations/scripts/visualize_training_logs.py)
* [training_visualizations/scripts/auc_calc.py](training_visualizations/scripts/auc_calc.py)

Run from repository root:

```bash
python training_visualizations/scripts/generate_requested_visualizations.py
python training_visualizations/scripts/visualize_training_logs.py
python training_visualizations/scripts/auc_calc.py
```

## Notes for Reproducibility

* Primary sequence config: [sequence/config.yaml](sequence/config.yaml)
* Integration reference config: [config_multimodal.yaml](config_multimodal.yaml)
* Root entry scripts are path-safe and can be called from any working directory.
* The shared environment is a root [venv](https://docs.python.org/3/library/venv.html) created from Python 3.8.10 and populated from [requirements.txt](requirements.txt).

## Generated Artifacts and Git Ignore Policy

Generated files are intentionally excluded from version control via [.gitignore](.gitignore), including:

* model checkpoints (`*.pt`, `*.pth`, `*.pkl`)
* logs and run reports
* ensemble outputs (`ensemble_meta_data.csv`, `ensemble_seed_results.csv`, `ensemble_mean_sd_summary.csv`, `final_ensemble_test_results.csv`)
* visualization outputs under `training_visualizations/figures` and `training_visualizations/reports`
* temporary structure chunks under structure/tmp
