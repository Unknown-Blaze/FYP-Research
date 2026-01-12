import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split # Still needed for logic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# CONFIG 
# ==============================================================================
REPORT_FILE = "ensemble_results_report.txt"
# MUST MATCH YOUR GNN/ESM CONFIG EXACTLY
SEED = 42
TEST_SIZE = 0.1
VALID_SIZE = 0.1 

def log_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

def train_ensemble():
    with open(REPORT_FILE, "w") as f:
        log_print("="*60, f)
        log_print(f"STRICT ENSEMBLE TRAINING (NO LEAKAGE) - {datetime.now()}", f)
        log_print("="*60, f)

        # 1. Load Data
        if not os.path.exists("ensemble_meta_data.csv"):
            log_print("❌ Error: 'ensemble_meta_data.csv' not found.", f)
            return

        df = pd.read_csv("ensemble_meta_data.csv")
        
        # 2. RECONSTRUCT THE SPLIT (CRITICAL STEP)
        # We need to replicate the exact shuffling logic used in the base models
        # so we know which rows correspond to Train, Val, and Test.
        
        # A. Clean NaNs (Must match the cleaning done in Inference/Base models)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # B. Split 1: Isolate Test Set
        # These are the rows your base models NEVER saw.
        train_val_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=SEED
        )
        
        # C. Split 2: Isolate Validation Set from Train
        # These are the rows your base models used for Early Stopping, but NOT training weights.
        valid_size_scaled = VALID_SIZE / (1.0 - TEST_SIZE)
        train_df, val_df = train_test_split(
            train_val_df, test_size=valid_size_scaled, random_state=SEED
        )
        
        log_print(f"\n[1] DATA PARTITIONING (STRICT)", f)
        log_print(f"   Base Model Train Set: {len(train_df)} (DISCARDED - Biased)", f)
        log_print(f"   Ensemble Train Set:   {len(val_df)}   (Used to train Ensemble)", f)
        log_print(f"   Ensemble Test Set:    {len(test_df)}  (Final Evaluation)", f)

        # 3. ASSIGN DATASETS
        # We TRAIN the ensemble on val_df (unseen by base model weights)
        # We TEST the ensemble on test_df (unseen by everyone)
        train_source = val_df 
        test_source = test_df 

        # Feature Engineering
        for data in [train_source, test_source]:
            data['len_x_diff'] = data['peptide_len'] * data['diff']

        features_mlp = ['esm_pred', 'gnn_pred', 'peptide_len', 'plddt', 'diff', 'len_x_diff']
        features_linear = ['esm_pred', 'gnn_pred']
        target = 'actual_pic50'

        X_train = train_source
        y_train = train_source[target]
        X_test = test_source
        y_test = test_source[target]

        # ==============================================================================
        # STRATEGY 1: LINEAR MIXER
        # ==============================================================================
        log_print("\n" + "-"*30, f)
        log_print("TRAINING MODEL 1: LINEAR MIXER", f)
        
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(X_train[features_linear], y_train)
        
        log_print(f"   ESM Weight: {linear_model.coef_[0]:.4f}", f)
        log_print(f"   GNN Weight: {linear_model.coef_[1]:.4f}", f)

        # ==============================================================================
        # STRATEGY 2: NEURAL MIXER
        # ==============================================================================
        log_print("\n" + "-"*30, f)
        log_print("TRAINING MODEL 2: NEURAL MIXER", f)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[features_mlp])
        X_test_scaled = scaler.transform(X_test[features_mlp])
        
        # Note: Since validation sets are small, we reduce complexity to prevent overfitting
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(64, 32), # Smaller architecture for smaller data
            activation='relu',
            solver='adam',
            max_iter=2000,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp_model.fit(X_train_scaled, y_train)

        # ==============================================================================
        # EVALUATION (ON TEST SET)
        # ==============================================================================
        def get_metrics(preds, y_true):
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            r2 = r2_score(y_true, preds)
            return rmse, r2

        log_print("\n" + "="*60, f)
        log_print("FINAL UNBIASED TEST RESULTS", f)
        log_print("="*60, f)
        
        # 1. Linear Stack
        lin_preds = linear_model.predict(X_test[features_linear])
        l_rmse, l_r2 = get_metrics(lin_preds, y_test)
        log_print(f"Linear Stack | RMSE: {l_rmse:.4f} | R2: {l_r2:.4f}", f)

        # 2. Neural Stack
        mlp_preds = mlp_model.predict(X_test_scaled)
        m_rmse, m_r2 = get_metrics(mlp_preds, y_test)
        log_print(f"Neural Stack | RMSE: {m_rmse:.4f} | R2: {m_r2:.4f}", f)
        
        # Save Models
        joblib.dump(linear_model, "linear_meta_learner.pkl")
        joblib.dump(mlp_model, "neural_meta_learner.pkl")
        joblib.dump(scaler, "meta_learner_scaler.pkl")

    print(f"\n✅ Done! Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    train_ensemble()