import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaFold pIC50 prediction performance from the final summary file.')
    parser.add_argument('--predictions_file', required=True, help='Path to the final TSV file from run_prediction_ic50.py.')
    parser.add_argument('--ground_truth_file', required=True, help='Path to the original CSV file used for testing.')
    parser.add_argument('--output_prefix', required=True, help='Prefix for saving output files (e.g., plots and merged data).')
    args = parser.parse_args()

    print("Loading prediction and ground truth data...")
    try:
        preds_df = pd.read_csv(args.predictions_file, sep='\t')
        truth_df = pd.read_csv(args.ground_truth_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. {e}")
        return
    
    # Debug: Show all columns in predictions file
    print(f"All columns in predictions file: {list(preds_df.columns)}")
    
    # Check for any columns that might contain prediction values
    potential_pred_cols = [col for col in preds_df.columns if 'pred' in col.lower() or 'pic50' in col.lower()]
    print(f"Potential prediction columns: {potential_pred_cols}")
    
    # Check non-null counts for prediction-related columns
    for col in potential_pred_cols:
        non_null_count = preds_df[col].notna().sum()
        print(f"Column '{col}': {non_null_count} non-null values out of {len(preds_df)}")
        if non_null_count > 0:
            print(f"  Sample values: {preds_df[col].dropna().head(5).tolist()}")

    # --- THIS IS THE CRITICAL FIX ---
    # Clean the 'targetid' columns in both dataframes before merging
    print("Cleaning merge keys (stripping whitespace and ensuring string type)...")
    preds_df['targetid'] = preds_df['targetid'].astype(str).str.strip()
    truth_df['targetid'] = truth_df['targetid'].astype(str).str.strip()
    # --- END OF FIX ---

    affinity_nm = truth_df['iedb_binding_affinity_nM'].clip(upper=50000)
    truth_df['true_pIC50'] = 9 - np.log10(affinity_nm)

    print("Attempting to merge dataframes...")
    
    # Debug information before merging
    print(f"Predictions dataframe shape: {preds_df.shape}")
    print(f"Ground truth dataframe shape: {truth_df.shape}")
    print(f"Sample prediction targetids: {preds_df['targetid'].head().tolist()}")
    print(f"Sample ground truth targetids: {truth_df['targetid'].head().tolist()}")
    
    merged_df = pd.merge(preds_df, truth_df[['targetid', 'true_pIC50']], on='targetid', how='inner')
    print(f"After merge - dataframe shape: {merged_df.shape}")

    if merged_df.empty:
        print("Error: Merging failed. No common 'targetid' found between files.")
        print("Investigating possible mismatches...")
        
        # Check for common targetids
        pred_ids = set(preds_df['targetid'].unique())
        truth_ids = set(truth_df['targetid'].unique())
        common_ids = pred_ids.intersection(truth_ids)
        
        print(f"Unique targetids in predictions: {len(pred_ids)}")
        print(f"Unique targetids in ground truth: {len(truth_ids)}")
        print(f"Common targetids: {len(common_ids)}")
        
        if len(common_ids) == 0:
            print("Sample prediction targetids:")
            for i, tid in enumerate(list(pred_ids)[:5]):
                print(f"  {i+1}. '{tid}' (type: {type(tid)})")
            print("Sample ground truth targetids:")
            for i, tid in enumerate(list(truth_ids)[:5]):
                print(f"  {i+1}. '{tid}' (type: {type(tid)})")
        
        return

    pred_col = None
    for col in merged_df.columns:
        if col.endswith('_pred_pIC50'):
            pred_col = col
            break
    
    if pred_col is None:
        print(f"Error: Could not find a prediction column ending with '_pred_pIC50'.")
        print(f"Available columns: {list(merged_df.columns)}")
        return
        
    print(f"Found prediction column: {pred_col}")
    
    # Check for NaN values in the prediction column before processing
    nan_count = merged_df[pred_col].isna().sum()
    print(f"NaN values in {pred_col}: {nan_count} out of {len(merged_df)}")
    
    # Show sample values from the prediction column
    print(f"Sample values from {pred_col}:")
    print(merged_df[pred_col].head(10).tolist())
    
    merged_df.rename(columns={pred_col: 'predicted_pIC50'}, inplace=True)
    
    print(f"Before dropna - dataframe shape: {merged_df.shape}")
    merged_df.dropna(subset=['predicted_pIC50'], inplace=True)
    print(f"After dropna - dataframe shape: {merged_df.shape}")
    
    print(f"Successfully merged and processed {len(merged_df)} data points.")

    # Check if we have enough data points for analysis
    if len(merged_df) < 2:
        print(f"Error: Not enough data points ({len(merged_df)}) for statistical analysis. Need at least 2 data points.")
        if len(merged_df) == 1:
            print("Single data point found:")
            print(merged_df[['targetid', 'true_pIC50', 'predicted_pIC50']].iloc[0])
        return

    # --- The rest of the script is the same ---
    print("\n--- Regression Performance ---")
    true_vals = merged_df['true_pIC50']
    pred_vals = merged_df['predicted_pIC50']

    corr, _ = pearsonr(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_absolute_error(true_vals, pred_vals))

    print(f"Pearson Correlation: {corr:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='true_pIC50', y='predicted_pIC50', data=merged_df, alpha=0.6)
    plt.plot([4, 10], [4, 10], 'r--', label='Ideal Prediction (y=x)')
    plt.title(f'Prediction vs. Ground Truth (Correlation: {corr:.2f})')
    plt.xlabel('True pIC50')
    plt.ylabel('Predicted pIC50')
    plt.grid(True)
    plt.legend()
    scatter_path = f'{args.output_prefix}_scatter_plot.png'
    plt.savefig(scatter_path)
    print(f"Saved scatter plot to: {scatter_path}")
    plt.close()

    print("\n--- Classification Performance (Binder vs. Non-Binder) ---")
    BINDER_THRESHOLD = 9 - np.log10(500) 
    merged_df['true_label'] = (merged_df['true_pIC50'] >= BINDER_THRESHOLD).astype(int)
    
    if len(merged_df['true_label'].unique()) < 2:
        print("Warning: Cannot calculate AUC because the test set contains only one class.")
    else:
        auc = roc_auc_score(merged_df['true_label'], merged_df['predicted_pIC50'])
        print(f"Area Under ROC Curve (AUC): {auc:.4f}")

        fpr, tpr, _ = roc_curve(merged_df['true_label'], merged_df['predicted_pIC50'])
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Binder Prediction')
        plt.legend(loc="lower right")
        roc_path = f'{args.output_prefix}_roc_curve.png'
        plt.savefig(roc_path)
        print(f"Saved ROC curve plot to: {roc_path}")
        plt.close()

    final_csv_path = f'{args.output_prefix}_merged_results.csv'
    merged_df.to_csv(final_csv_path, index=False)
    print(f"\nSaved merged data with all results to: {final_csv_path}")

if __name__ == '__main__':
    main()