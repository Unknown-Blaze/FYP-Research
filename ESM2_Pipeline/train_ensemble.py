import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_ensemble():
    # 1. Load Data
    print("Loading ensemble data...")
    df = pd.read_csv("ensemble_meta_data.csv")
    
    # 2. Filter Extreme Outliers (Optional but recommended)
    # Sometimes models output -999 or NaN if something fails
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    features = ['esm_pred', 'gnn_pred', 'peptide_len', 'plddt', 'diff']
    target = 'actual_pic50'
    
    X = df[features]
    y = df[target]

    # 3. Scale Features for the MLP (Neural Nets need scaled inputs)
    # We don't scale 'esm_pred' and 'gnn_pred' usually for Linear Reg, 
    # but for MLP it helps convergence.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # --- STRATEGY 1: Linear Mixer ---
    # We use original X for Linear Regression interpretation
    X_lin_train, X_lin_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    linear_model = LinearRegression()
    linear_model.fit(X_lin_train[['esm_pred', 'gnn_pred']], y_train) # Only use predictions for Linear
    
    print("\n" + "="*40)
    print("📊 LINEAR MIXER WEIGHTS")
    print("="*40)
    print(f"ESM Weight: {linear_model.coef_[0]:.4f}")
    print(f"GNN Weight: {linear_model.coef_[1]:.4f}")
    print(f"Bias:       {linear_model.intercept_:.4f}")
    
    # --- STRATEGY 2: Neural Mixer ---
    print("\nTraining Neural Mixer (MLP)...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), 
        activation='relu', 
        solver='adam',
        max_iter=1000, 
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True
    )
    mlp_model.fit(X_train, y_train)

    # --- EVALUATION ---
    print("\n" + "="*40)
    print("🏆 FINAL RESULTS (Test Set)")
    print("="*40)
    
    def evaluate(preds, name):
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return r2, rmse

    # Get Single Model Performance (from the Test set slice)
    # Note: X_lin_test is a DataFrame, so we access columns by name
    esm_preds = X_lin_test['esm_pred']
    gnn_preds = X_lin_test['gnn_pred']
    
    r2_esm, rmse_esm = evaluate(esm_preds, "ESM Only")
    r2_gnn, rmse_gnn = evaluate(gnn_preds, "GNN Only")
    
    print(f"1. ESM Only    -> R2: {r2_esm:.4f} | RMSE: {rmse_esm:.4f}")
    print(f"2. GNN Only    -> R2: {r2_gnn:.4f} | RMSE: {rmse_gnn:.4f}")

    # Get Ensemble Performance
    lin_preds = linear_model.predict(X_lin_test[['esm_pred', 'gnn_pred']])
    r2_lin, rmse_lin = evaluate(lin_preds, "Linear Stack")
    print(f"3. Linear Stack-> R2: {r2_lin:.4f} | RMSE: {rmse_lin:.4f}")

    mlp_preds = mlp_model.predict(X_test)
    r2_mlp, rmse_mlp = evaluate(mlp_preds, "Neural Stack")
    print(f"4. Neural Stack-> R2: {r2_mlp:.4f} | RMSE: {rmse_mlp:.4f}")

    # Save
    joblib.dump(mlp_model, "best_meta_learner_mlp.pkl")
    joblib.dump(scaler, "meta_learner_scaler.pkl") # Save scaler for inference!
    print("\n✅ Saved MLP model and Scaler.")

if __name__ == "__main__":
    train_ensemble()