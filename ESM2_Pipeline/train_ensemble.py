import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_ensemble():
    # 1. Load Data
    print("Loading ensemble data...")
    df = pd.read_csv("ensemble_meta_data.csv")
    
    # 2. Define Features & Target
    # We use the model predictions + metadata to make the final decision
    # Note: 'diff' is useful for the Neural Net to find disagreements
    features = ['esm_pred', 'gnn_pred', 'peptide_len', 'plddt', 'diff']
    target = 'actual_pic50'
    
    X = df[features]
    y = df[target]

    # 3. Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # --- STRATEGY 1: Linear Mixer (The "Sanity Check") ---
    # This learns a simple weighted average: Final = w1*ESM + w2*GNN + bias
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    print("\n" + "="*40)
    print("📊 LINEAR MIXER WEIGHTS")
    print("="*40)
    for name, weight in zip(features, linear_model.coef_):
        print(f"{name:15s}: {weight:.4f}")
    print(f"Bias (Intercept): {linear_model.intercept_:.4f}")
    
    # --- STRATEGY 2: Neural Mixer (The "Smart" Gate) ---
    # A tiny neural net that can learn complex rules like:
    # "If pLDDT is low, ignore GNN prediction"
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        max_iter=500, 
        random_state=42,
        learning_rate_init=0.001
    )
    mlp_model.fit(X_train, y_train)

    # --- EVALUATION ---
    print("\n" + "="*40)
    print("🏆 FINAL RESULTS (Test Set)")
    print("="*40)
    
    def evaluate(model, name, preds=None):
        if preds is None:
            preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"[{name}]")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}\n")
        return preds

    # 1. Baselines
    print("--- BASELINES (Single Models) ---")
    evaluate(None, "ESM Only", preds=X_test['esm_pred'])
    evaluate(None, "GNN Only", preds=X_test['gnn_pred'])
    
    # 2. Ensembles
    print("--- ENSEMBLES ---")
    lin_preds = evaluate(linear_model, "Linear Stack")
    mlp_preds = evaluate(mlp_model, "Neural Stack")

    # 4. Save the best model
    # We save the MLP as it usually generalizes better if data is sufficient
    joblib.dump(mlp_model, "best_meta_learner.pkl")
    print("✅ Best Meta-Learner saved to 'best_meta_learner.pkl'")
    
    # 5. Save Predictions for Analysis
    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['pred_linear'] = lin_preds
    results_df['pred_mlp'] = mlp_preds
    results_df.to_csv("final_ensemble_test_results.csv", index=False)
    print("✅ Detailed test results saved to 'final_ensemble_test_results.csv'")

if __name__ == "__main__":
    train_ensemble()