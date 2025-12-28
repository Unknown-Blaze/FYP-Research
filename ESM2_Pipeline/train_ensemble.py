import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def train_ensemble():
    # 1. Load Data
    print("Loading ensemble data...")
    if not os.path.exists("ensemble_meta_data.csv"):
        print("❌ Error: 'ensemble_meta_data.csv' not found. Run inference first.")
        return

    df = pd.read_csv("ensemble_meta_data.csv")
    
    # 2. Filter Extreme Outliers
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    features = ['esm_pred', 'gnn_pred', 'peptide_len', 'plddt', 'diff']
    target = 'actual_pic50'
    
    X = df[features]
    y = df[target]

    # 3. Split Data (80/20) - Keep raw X for Linear model interpretation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # --- STRATEGY 1: Linear Mixer (Uses RAW Predictions) ---
    print("\nTraining Linear Mixer...")
    linear_model = LinearRegression()
    linear_model.fit(X_train[['esm_pred', 'gnn_pred']], y_train) 
    
    print("\n" + "="*40)
    print("📊 LINEAR MIXER WEIGHTS")
    print("="*40)
    print(f"ESM Weight: {linear_model.coef_[0]:.4f}")
    print(f"GNN Weight: {linear_model.coef_[1]:.4f}")
    print(f"Bias:       {linear_model.intercept_:.4f}")
    
    # --- STRATEGY 2: Neural Mixer (Uses SCALED Features) ---
    print("\nTraining Neural Mixer (MLP)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), 
        activation='relu', 
        solver='adam',
        max_iter=1000, 
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True
    )
    mlp_model.fit(X_train_scaled, y_train)

    # --- EVALUATION FUNCTION ---
    def evaluate(preds, name):
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        pcc, _ = pearsonr(y_test, preds)
        scc, _ = spearmanr(y_test, preds)
        
        print(f"[{name}]")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  PCC:  {pcc:.4f}")
        print(f"  SCC:  {scc:.4f}")
        print("-" * 30)
        return preds

    print("\n" + "="*40)
    print("🏆 FINAL RESULTS (Test Set)")
    print("="*40)
    
    # 1. Base Models (Raw Data)
    evaluate(X_test['esm_pred'], "1. ESM Only")
    evaluate(X_test['gnn_pred'], "2. GNN Only")

    # 2. Linear Ensemble
    lin_preds = linear_model.predict(X_test[['esm_pred', 'gnn_pred']])
    evaluate(lin_preds, "3. Linear Stack")

    # 3. Neural Ensemble (Using Scaled Data)
    mlp_preds = mlp_model.predict(X_test_scaled)
    evaluate(mlp_preds, "4. Neural Stack (MLP)")

    # Save Everything
    joblib.dump(linear_model, "linear_meta_learner.pkl")
    joblib.dump(mlp_model, "best_meta_learner_mlp.pkl")
    joblib.dump(scaler, "meta_learner_scaler.pkl") 
    print("\n✅ Saved Linear, MLP models, and Scaler.")
    
    # --- BONUS: Feature Importance for Neural Net ---
    from sklearn.inspection import permutation_importance
    
    print("\n" + "="*40)
    print("🧠 NEURAL NET FEATURE IMPORTANCE")
    print("="*40)
    
    # Calculate importance on Test Set
    perm_importance = permutation_importance(mlp_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    
    # Map back to feature names
    for i, feature in enumerate(features):
        print(f"{feature:15s}: {perm_importance.importances_mean[i]:.4f}")
    
    # Save Predictions for Plotting
    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['pred_linear'] = lin_preds
    results_df['pred_mlp'] = mlp_preds
    results_df.to_csv("final_ensemble_test_results.csv", index=False)
    print("✅ Detailed test results saved to 'final_ensemble_test_results.csv'")

if __name__ == "__main__":
    train_ensemble()