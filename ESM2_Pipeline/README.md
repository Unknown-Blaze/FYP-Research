# ESM-2 Fine-Tuning for MHC-Peptide Binding

This project fine-tunes an ESM-2 model to predict the binding affinity (pIC50) between MHC and peptide sequences.

It is driven by a configuration file (`config.yaml`) for easy experiment management and includes robust logging, checkpointing, and evaluation.

## 1. Setup

### Create Conda Environment
We recommend using Conda to manage dependencies.

1.  **Create a new environment** (e.g., named `esm`):
    ```bash
    conda create -n esm python=3.10
    ```

2.  **Activate the environment:**
    ```bash
    conda activate esm
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Configuration

All experiment settings are controlled by the `config.yaml` file. Before running, edit this file to:

* Point `data_file` to your dataset.
* Set your desired `output_dir`.
* Choose your `model_name` (e.g., `facebook/esm2_t33_650M_UR50D` or `facebook/esm2_t36_3B_UR50D`).
* Adjust hyperparameters like `batch_size`, `lr`, and `num_workers` to match your GPU (see A100 recommendations in the file).

## 3. How to Run

### To Start a New Training
Use the provided shell script:
```bash
./main.sh