# ESM2 Multimodal Pipeline

This repository is organized for readers who want to reproduce a three-stage MHC-peptide binding workflow:

1. Sequence model training in [sequence](sequence)
2. Structure graph generation and GNN training in [structure](structure)
3. Fusion/ensemble stacking at repository root

## Repository Layout

* [sequence](sequence): ESM2 fine-tuning and evaluation
* [structure](structure): template download, graph generation, GNN training
* root scripts: [extract_predictions.py](extract_predictions.py), [train_ensemble.py](train_ensemble.py), [extract_and_ensemble.sh](extract_and_ensemble.sh)
* visualizations: [training_visualizations/scripts](training_visualizations/scripts), outputs in [training_visualizations/figures](training_visualizations/figures) and [training_visualizations/reports](training_visualizations/reports)

## Environment Setup

Use one environment for sequence + ensemble and one optional structure-focused conda environment.

### Option A: Pip environment (sequence + ensemble)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda environment (structure pipeline)

```bash
conda env create -f structure/environment_esm_graph.yml
conda activate esm_graph
```

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

## Generated Artifacts and Git Ignore Policy

Generated files are intentionally excluded from version control via [.gitignore](.gitignore), including:

* model checkpoints (`*.pt`, `*.pth`, `*.pkl`)
* logs and run reports
* ensemble outputs (`ensemble_meta_data.csv`, `ensemble_seed_results.csv`, `ensemble_mean_sd_summary.csv`, `final_ensemble_test_results.csv`)
* visualization outputs under [training_visualizations/figures](training_visualizations/figures) and [training_visualizations/reports](training_visualizations/reports)
* temporary structure chunks under structure/tmp