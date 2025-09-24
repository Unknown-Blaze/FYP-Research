import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve

def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaFold pIC50 prediction performance from the final summary file (no plots version).')
    parser.add_argument('--predictions_file', required=True, help='Path to the final TSV file from run_prediction_ic50.py.')
    parser.add_argument('--ground_truth_file', required=True, help='Path to the original CSV file used for testing.')
    parser.add_argument('--output_prefix', required=True, help='Prefix for saving output files (merged data).')
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

    # Clean the 'targetid' columns in both dataframes before merging
    print("Cleaning merge keys (stripping whitespace and ensuring string type)...")
    preds_df['targetid'] = preds_df['targetid'].astype(str).str.strip()
    truth_df['targetid'] = truth_df['targetid'].astype(str).str.strip()

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

    # Regression Performance Analysis
    print("\n" + "="*50)
    print("--- REGRESSION PERFORMANCE ANALYSIS ---")
    print("="*50)
    
    true_vals = merged_df['true_pIC50']
    pred_vals = merged_df['predicted_pIC50']

    corr, p_value = pearsonr(true_vals, pred_vals)
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    
    # Calculate additional metrics
    bias = np.mean(pred_vals - true_vals)
    std_residuals = np.std(pred_vals - true_vals)
    
    print(f"Number of data points: {len(merged_df)}")
    print(f"Pearson Correlation: {corr:.4f} (p-value: {p_value:.4e})")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Bias (Mean prediction - Mean true): {bias:.4f}")
    print(f"Standard deviation of residuals: {std_residuals:.4f}")
    
    # Value ranges
    print(f"\nValue Ranges:")
    print(f"True pIC50 range: {true_vals.min():.4f} to {true_vals.max():.4f}")
    print(f"Predicted pIC50 range: {pred_vals.min():.4f} to {pred_vals.max():.4f}")

    # Classification Performance Analysis
    print("\n" + "="*50)
    print("--- CLASSIFICATION PERFORMANCE ANALYSIS ---")
    print("--- (Binder vs. Non-Binder at 500 nM threshold) ---")
    print("="*50)
    
    BINDER_THRESHOLD = 9 - np.log10(500)  # pIC50 threshold for 500 nM
    print(f"Binder threshold (500 nM): pIC50 >= {BINDER_THRESHOLD:.4f}")
    
    merged_df['true_label'] = (merged_df['true_pIC50'] >= BINDER_THRESHOLD).astype(int)
    merged_df['pred_label'] = (merged_df['predicted_pIC50'] >= BINDER_THRESHOLD).astype(int)
    
    # Count true positives, false positives, etc.
    true_binders = (merged_df['true_label'] == 1).sum()
    true_non_binders = (merged_df['true_label'] == 0).sum()
    
    print(f"True binders (>=500 nM): {true_binders}")
    print(f"True non-binders (<500 nM): {true_non_binders}")
    
    if len(merged_df['true_label'].unique()) < 2:
        print("Warning: Cannot calculate AUC because the test set contains only one class.")
    else:
        auc = roc_auc_score(merged_df['true_label'], merged_df['predicted_pIC50'])
        print(f"Area Under ROC Curve (AUC): {auc:.4f}")
        
        # Calculate confusion matrix metrics
        tp = ((merged_df['true_label'] == 1) & (merged_df['pred_label'] == 1)).sum()
        fp = ((merged_df['true_label'] == 0) & (merged_df['pred_label'] == 1)).sum()
        tn = ((merged_df['true_label'] == 0) & (merged_df['pred_label'] == 0)).sum()
        fn = ((merged_df['true_label'] == 1) & (merged_df['pred_label'] == 0)).sum()
        
        print(f"\nConfusion Matrix (at threshold {BINDER_THRESHOLD:.4f}):")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"Precision: {precision:.4f}")
        else:
            print("Precision: undefined (no predicted positives)")
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"Recall (Sensitivity): {recall:.4f}")
        else:
            print("Recall: undefined (no true positives)")
            
        if tn + fp > 0:
            specificity = tn / (tn + fp)
            print(f"Specificity: {specificity:.4f}")
        else:
            print("Specificity: undefined (no true negatives)")
            
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        print(f"Accuracy: {accuracy:.4f}")

    # Additional analysis: performance by ranges
    print("\n" + "="*50)
    print("--- PERFORMANCE BY pIC50 RANGES ---")
    print("="*50)
    
    # Define ranges
    ranges = [
        (0, 5.0, "Very weak (0-5.0)"),
        (5.0, 6.0, "Weak (5.0-6.0)"),
        (6.0, 7.0, "Moderate (6.0-7.0)"),
        (7.0, 8.0, "Strong (7.0-8.0)"),
        (8.0, 10.0, "Very strong (8.0+)")
    ]
    
    for min_val, max_val, label in ranges:
        mask = (merged_df['true_pIC50'] >= min_val) & (merged_df['true_pIC50'] < max_val)
        subset = merged_df[mask]
        
        if len(subset) > 1:
            subset_corr, _ = pearsonr(subset['true_pIC50'], subset['predicted_pIC50'])
            subset_mae = mean_absolute_error(subset['true_pIC50'], subset['predicted_pIC50'])
            print(f"{label}: n={len(subset)}, r={subset_corr:.3f}, MAE={subset_mae:.3f}")
        elif len(subset) == 1:
            print(f"{label}: n=1 (single point)")
        else:
            print(f"{label}: n=0 (no data)")

    # Save results
    final_csv_path = f'{args.output_prefix}_merged_results.csv'
    merged_df.to_csv(final_csv_path, index=False)
    print(f"\n" + "="*50)
    print(f"ANALYSIS COMPLETE")
    print("="*50)
    print(f"Saved merged data with all results to: {final_csv_path}")
    
    # Save summary stats
    summary_path = f'{args.output_prefix}_summary_stats.txt'
    with open(summary_path, 'w') as f:
        f.write("AlphaFold pIC50 Prediction Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {args.predictions_file}\n")
        f.write(f"Ground truth: {args.ground_truth_file}\n")
        f.write(f"Number of data points: {len(merged_df)}\n\n")
        
        f.write("Regression Performance:\n")
        f.write(f"  Pearson Correlation: {corr:.4f} (p-value: {p_value:.4e})\n")
        f.write(f"  Mean Absolute Error: {mae:.4f}\n")
        f.write(f"  Root Mean Squared Error: {rmse:.4f}\n")
        f.write(f"  Bias: {bias:.4f}\n")
        f.write(f"  Std of residuals: {std_residuals:.4f}\n\n")
        
        if len(merged_df['true_label'].unique()) >= 2:
            f.write("Classification Performance (500 nM threshold):\n")
            f.write(f"  AUC: {auc:.4f}\n")
            f.write(f"  Accuracy: {accuracy:.4f}\n")
            if 'precision' in locals():
                f.write(f"  Precision: {precision:.4f}\n")
            if 'recall' in locals():
                f.write(f"  Recall: {recall:.4f}\n")
            if 'specificity' in locals():
                f.write(f"  Specificity: {specificity:.4f}\n")
    
    print(f"Saved summary statistics to: {summary_path}")

if __name__ == '__main__':
    main()