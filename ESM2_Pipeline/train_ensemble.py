import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# CONFIG
# ==============================================================================
REPORT_FILE = "ensemble_results_report.txt"
SEED = 42
TEST_SIZE = 0.1
VALID_SIZE = 0.1 

def log_print(message, file_handle=None):
    print(message)
    if file_handle:
        file_handle.write(message + "\n")

def print_section_header(title, file_handle=None):
    msg = "\n" + "="*40 + f"\n{title}\n" + "="*40
    log_print(msg, file_handle)

def evaluate_model(preds, y_true, name, file_handle=None):
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    pcc, _ = pearsonr(y_true, preds)
    scc, _ = spearmanr(y_true, preds)
    
    log_print(f"[{name}]", file_handle)
    log_print(f"  RMSE: {rmse:.4f}", file_handle)
    log_print(f"  R2:   {r2:.4f}", file_handle)
    log_print(f"  PCC:  {pcc:.4f}", file_handle)
    log_print(f"  SCC:  {scc:.4f}", file_handle)
    log_print("-" * 30, file_handle)
    return {
        'rmse': rmse,
        'r2': r2,
        'pcc': pcc,
        'scc': scc
    }


def summarize_metrics(all_seed_metrics):
    rows = []
    metric_keys = ['rmse', 'r2', 'pcc', 'scc']
    for model_name in all_seed_metrics[0].keys():
        model_rows = {'model': model_name}
        for key in metric_keys:
            vals = [seed_metrics[model_name][key] for seed_metrics in all_seed_metrics]
            model_rows[f'{key}_mean'] = float(np.mean(vals))
            model_rows[f'{key}_sd'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(model_rows)
    return pd.DataFrame(rows)


def run_one_seed(df, split_seed, meta_seed, save_artifacts=False, file_handle=None):
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=split_seed)

    valid_size_scaled = VALID_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(train_val_df, test_size=valid_size_scaled, random_state=split_seed)

    X_train = val_df.copy()
    y_train = val_df['actual_pic50']
    X_test = test_df.copy()
    y_test = test_df['actual_pic50']

    for data in [X_train, X_test]:
        data['len_x_diff'] = data['peptide_len'] * data['diff']

    features_linear = ['esm_pred', 'gnn_pred']
    features_mlp = ['esm_pred', 'gnn_pred', 'peptide_len', 'plddt', 'diff', 'len_x_diff']

    if file_handle is not None:
        log_print(f"Split seed: {split_seed} | Meta seed: {meta_seed}", file_handle)
        log_print(f"Data Loaded: {len(df)} total samples.", file_handle)
        log_print(f" > Discarded Base-Train: {len(train_df)} (Biased)", file_handle)
        log_print(f" > Ensemble Training:    {len(X_train)} (Unseen by Base Weights)", file_handle)
        log_print(f" > Final Testing:        {len(X_test)} (Held Out)", file_handle)

    linear_model = Ridge(alpha=1.0)
    linear_model.fit(X_train[features_linear], y_train)

    if file_handle is not None:
        print_section_header("TRAINING LINEAR MIXER", file_handle)
        log_print(f"ESM Weight: {linear_model.coef_[0]:.4f}", file_handle)
        log_print(f"GNN Weight: {linear_model.coef_[1]:.4f}", file_handle)
        log_print(f"Bias:       {linear_model.intercept_:.4f}", file_handle)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features_mlp])
    X_test_scaled = scaler.transform(X_test[features_mlp])

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=meta_seed
    )
    mlp_model.fit(X_train_scaled, y_train)

    if file_handle is not None:
        print_section_header("TRAINING NEURAL MIXER (MLP)", file_handle)
        log_print("✅ MLP Training Complete.", file_handle)

    lin_preds = linear_model.predict(X_test[features_linear])
    mlp_preds = mlp_model.predict(X_test_scaled)

    if file_handle is not None:
        print_section_header("🏆 FINAL RESULTS (Test Set)", file_handle)

    seed_metrics = {
        'ESM Only': evaluate_model(X_test['esm_pred'], y_test, "1. ESM Only", file_handle),
        'GNN Only': evaluate_model(X_test['gnn_pred'], y_test, "2. GNN Only", file_handle),
        'Linear Stack': evaluate_model(lin_preds, y_test, "3. Linear Stack", file_handle),
        'Neural Stack (MLP)': evaluate_model(mlp_preds, y_test, "4. Neural Stack (MLP)", file_handle),
    }

    if save_artifacts:
        joblib.dump(linear_model, "linear_meta_learner.pkl")
        joblib.dump(mlp_model, "neural_meta_learner.pkl")
        joblib.dump(scaler, "meta_learner_scaler.pkl")

        results_df = X_test.copy()
        results_df['actual'] = y_test
        results_df['pred_linear'] = lin_preds
        results_df['pred_mlp'] = mlp_preds
        results_df.to_csv("final_ensemble_test_results.csv", index=False)

    return seed_metrics

def train_ensemble(meta_seeds, split_seed):
    f = open(REPORT_FILE, "w")
    log_print(f"ENSEMBLE TRAINING REPORT - {datetime.now()}\n", f)

    if not os.path.exists("ensemble_meta_data.csv"):
        log_print("❌ Error: 'ensemble_meta_data.csv' not found.", f)
        f.close()
        return

    df = pd.read_csv("ensemble_meta_data.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    all_seed_metrics = []
    per_seed_rows = []

    for idx, meta_seed in enumerate(meta_seeds):
        print_section_header(f"SEED RUN {idx+1}/{len(meta_seeds)} (meta_seed={meta_seed})", f)
        seed_metrics = run_one_seed(
            df=df,
            split_seed=split_seed,
            meta_seed=meta_seed,
            save_artifacts=(idx == len(meta_seeds) - 1),
            file_handle=f
        )
        all_seed_metrics.append(seed_metrics)
        for model_name, vals in seed_metrics.items():
            per_seed_rows.append({
                'meta_seed': meta_seed,
                'split_seed': split_seed,
                'model': model_name,
                'rmse': vals['rmse'],
                'r2': vals['r2'],
                'pcc': vals['pcc'],
                'scc': vals['scc']
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv("ensemble_seed_results.csv", index=False)

    summary_df = summarize_metrics(all_seed_metrics)
    summary_df.to_csv("ensemble_mean_sd_summary.csv", index=False)

    print_section_header("MEAN ± SD SUMMARY", f)
    for _, row in summary_df.iterrows():
        log_print(
            (
                f"[{row['model']}] "
                f"RMSE {row['rmse_mean']:.4f} ± {row['rmse_sd']:.4f} | "
                f"R2 {row['r2_mean']:.4f} ± {row['r2_sd']:.4f} | "
                f"PCC {row['pcc_mean']:.4f} ± {row['pcc_sd']:.4f} | "
                f"SCC {row['scc_mean']:.4f} ± {row['scc_sd']:.4f}"
            ),
            f
        )

    log_print("\n✅ Saved: ensemble_seed_results.csv", f)
    log_print("✅ Saved: ensemble_mean_sd_summary.csv", f)
    log_print("✅ Report saved to 'ensemble_results_report.txt'", f)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-seeds", type=str, default="42,123,999")
    parser.add_argument("--split-seed", type=int, default=SEED)
    args = parser.parse_args()

    try:
        meta_seeds = [int(x.strip()) for x in args.meta_seeds.split(",") if x.strip()]
    except ValueError:
        print("Invalid --meta-seeds format. Use comma-separated integers, e.g. 42,123,999")
        sys.exit(1)

    if len(meta_seeds) == 0:
        print("No valid meta seeds provided.")
        sys.exit(1)

    train_ensemble(meta_seeds=meta_seeds, split_seed=args.split_seed)