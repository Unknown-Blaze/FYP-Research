import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix

# 1. Configuration
# The standard threshold for MHC binding: 500nM = pIC50 ~6.3
THRESHOLD_PIC50 = 6.301  # -log10(500 * 10^-9)

def print_sota_comparison(my_metrics, model_name):
    # SOTA Values from your prompt
    sota_auc = 0.9831
    sota_acc = 0.9390
    sota_mcc = 0.0 # Nan in prompt, but usually ~0.8-0.9 for these stats
    
    print(f"\n--- {model_name} vs SOTA ---")
    print(f"AUC:         {my_metrics['auc']:.4f}  (SOTA: {sota_auc}) -> {'VICTORY! 🏆' if my_metrics['auc'] > sota_auc else 'Close...'}")
    print(f"Accuracy:    {my_metrics['acc']:.4f}  (SOTA: {sota_acc})")
    print(f"Sensitivity: {my_metrics['sens']:.4f}  (SOTA: 0.9421)")
    print(f"Specificity: {my_metrics['spec']:.4f}  (SOTA: 0.9360)")
    print(f"MCC:         {my_metrics['mcc']:.4f}")

def evaluate_file(filename):
    print(f"Reading {filename}...")
    df = pd.read_csv(filename)
    label_col = 'actual_pic50' if 'actual_pic50' in df.columns else 'actual'
    
    # 2. Convert Regression to Binary Classes
    # 1 = Binder (Stronger than 500nM / >6.3), 0 = Non-Binder
    y_true_bin = (df[label_col] >= THRESHOLD_PIC50).astype(int)
    
    # Evaluate both Linear and MLP stacks
    for col in ['pred_linear', 'pred_mlp']:
        y_score = df[col] # Continuous score for AUC
        y_pred_bin = (y_score >= THRESHOLD_PIC50).astype(int) # Binary label for Acc/MCC
        
        # Calculate Metrics
        auc = roc_auc_score(y_true_bin, y_score)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        
        metrics = {
            'auc': auc,
            'acc': acc,
            'mcc': mcc,
            'sens': tp / (tp + fn) if (tp+fn) > 0 else 0.0,
            'spec': tn / (tn + fp) if (tn+fp) > 0 else 0.0
        }
        
        print_sota_comparison(metrics, col.upper())

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    evaluate_file(str(repo_root / 'final_ensemble_test_results.csv'))