"""
ConvNeXt vs ESM-2 Comparison Script

This script evaluates both models on the same test set to compare performance.
Use this to determine if ESM-2 actually improves over ConvNeXt baseline.

Usage:
    python compare_models.py --test test_data.csv --esm2_model esm2_model.pt
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CONVNEXT vs ESM-2 PERFORMANCE COMPARISON")
print("=" * 70)

def evaluate_convnext(test_df, mhc_col='mhc_seq', peptide_col='peptide', ic50_col='ic50'):
    """
    Evaluate ConvNeXt baseline on test data
    
    Note: This is a placeholder. You need to integrate with actual ConvNeXt model.
    For now, we'll load predictions if they exist, or generate dummy predictions.
    """
    print("\n[1/2] Evaluating ConvNeXt Baseline...")
    
    # Check if ConvNeXt predictions exist
    convnext_pred_file = "convnext_predictions.csv"
    
    try:
        convnext_preds = pd.read_csv(convnext_pred_file)
        print(f"✓ Loaded ConvNeXt predictions from {convnext_pred_file}")
        return convnext_preds['prediction'].values
    except FileNotFoundError:
        print(f"⚠ {convnext_pred_file} not found")
        print("  To get real ConvNeXt predictions:")
        print("  1. Run: python train_evaluate.py evaluate --test your_test.csv")
        print("  2. Save predictions to convnext_predictions.csv")
        print("\n  For now, using baseline predictor (mean IC50)...")
        
        # Baseline: predict mean IC50
        mean_ic50 = test_df[ic50_col].mean()
        return np.full(len(test_df), mean_ic50)

def evaluate_esm2(test_df, model_path, mhc_col='mhc_seq', peptide_col='peptide'):
    """
    Evaluate ESM-2 model on test data
    """
    print("\n[2/2] Evaluating ESM-2 Model...")
    
    try:
        import torch
        from esm2_simple import ESM2ConvNeXtModel, MHCPeptideDataset, collate_fn
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"✗ Error importing ESM-2 dependencies: {e}")
        return None
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM2ConvNeXtModel().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded ESM-2 model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("  Train ESM-2 first with: python esm2_simple.py --train ... --output esm2_model.pt")
        return None
    
    model.eval()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            mhc_seq = [str(row[mhc_col])]
            pep_seq = [str(row[peptide_col])]
            
            pred = model(mhc_seq, pep_seq).squeeze(-1).cpu().numpy()[0]
            predictions.append(pred)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(test_df)} samples...")
    
    return np.array(predictions)

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Regression metrics
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
    metrics['R²'] = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pcc, pcc_pval = pearsonr(y_true, y_pred)
    scc, scc_pval = spearmanr(y_true, y_pred)
    metrics['PCC'] = pcc
    metrics['PCC_pval'] = pcc_pval
    metrics['SCC'] = scc
    metrics['SCC_pval'] = scc_pval
    
    # Classification metrics (if we threshold at IC50=500nM → pIC50=6.3)
    threshold = 6.3  # Strong binder threshold
    y_true_class = (y_true >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)
    
    if len(np.unique(y_true_class)) > 1:
        metrics['AUC'] = roc_auc_score(y_true_class, y_pred)
        metrics['Accuracy'] = np.mean(y_true_class == y_pred_class)
    else:
        metrics['AUC'] = np.nan
        metrics['Accuracy'] = np.nan
    
    return metrics

def print_comparison_table(convnext_metrics, esm2_metrics):
    """Print comparison table"""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'ConvNeXt':<15} {'ESM-2':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics_to_compare = ['RMSE', 'MAE', 'R²', 'PCC', 'SCC', 'AUC', 'Accuracy']
    
    for metric in metrics_to_compare:
        if metric in convnext_metrics and metric in esm2_metrics:
            conv_val = convnext_metrics[metric]
            esm2_val = esm2_metrics[metric]
            
            if np.isnan(conv_val) or np.isnan(esm2_val):
                improvement = "N/A"
            else:
                if metric in ['RMSE', 'MAE']:  # Lower is better
                    improvement = f"{(conv_val - esm2_val) / conv_val * 100:.1f}%"
                    winner = "✓" if esm2_val < conv_val else ""
                else:  # Higher is better
                    improvement = f"{(esm2_val - conv_val) / conv_val * 100:.1f}%"
                    winner = "✓" if esm2_val > conv_val else ""
                
                improvement = winner + " " + improvement
            
            print(f"{metric:<20} {conv_val:<15.4f} {esm2_val:<15.4f} {improvement:<15}")
    
    print("=" * 70)

def plot_comparison(y_true, convnext_pred, esm2_pred, output_path='comparison_plot.png'):
    """Create comparison visualization"""
    print(f"\nCreating comparison plot: {output_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: ConvNeXt predictions
    axes[0].scatter(y_true, convnext_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True pIC50')
    axes[0].set_ylabel('Predicted pIC50')
    axes[0].set_title('ConvNeXt Predictions')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ESM-2 predictions
    axes[1].scatter(y_true, esm2_pred, alpha=0.5, s=20)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1].set_xlabel('True pIC50')
    axes[1].set_ylabel('Predicted pIC50')
    axes[1].set_title('ESM-2 Predictions')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    convnext_errors = np.abs(y_true - convnext_pred)
    esm2_errors = np.abs(y_true - esm2_pred)
    
    axes[2].hist(convnext_errors, bins=30, alpha=0.5, label='ConvNeXt', color='blue')
    axes[2].hist(esm2_errors, bins=30, alpha=0.5, label='ESM-2', color='orange')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare ConvNeXt and ESM-2 models')
    parser.add_argument('--test', type=str, required=True, help='Test CSV file')
    parser.add_argument('--esm2_model', type=str, default='esm2_model.pt', help='ESM-2 model path')
    parser.add_argument('--convnext_preds', type=str, default='convnext_predictions.csv', 
                        help='ConvNeXt predictions CSV (optional)')
    parser.add_argument('--output_plot', type=str, default='comparison_plot.png', help='Output plot path')
    args = parser.parse_args()
    
    # Load test data
    print(f"\nLoading test data from {args.test}...")
    try:
        test_df = pd.read_csv(args.test)
    except:
        test_df = pd.read_csv(args.test, sep='\t')
    
    print(f"Loaded {len(test_df)} samples")
    print(f"Columns: {list(test_df.columns)}")
    
    # Find columns
    mhc_col = next((col for col in ['mhc_seq', 'mhc_sequence', 'MHC_sequence', 'allele_seq'] 
                    if col in test_df.columns), None)
    peptide_col = next((col for col in ['peptide', 'Peptide', 'pep_seq', 'peptide_sequence'] 
                        if col in test_df.columns), None)
    ic50_col = next((col for col in test_df.columns 
                     if 'ic50' in col.lower() or 'affinity' in col.lower()), None)
    
    if not all([mhc_col, peptide_col, ic50_col]):
        print("✗ Required columns not found!")
        return
    
    print(f"Using columns: mhc={mhc_col}, peptide={peptide_col}, ic50={ic50_col}")
    
    # Convert to pIC50 if needed
    if test_df[ic50_col].max() > 50:
        print("Converting IC50 to pIC50...")
        y_true = -np.log10(test_df[ic50_col] / 1e9)
    else:
        y_true = test_df[ic50_col].values
    
    # Evaluate ConvNeXt
    convnext_pred = evaluate_convnext(test_df, mhc_col, peptide_col, ic50_col)
    if convnext_pred is None:
        return
    
    # Evaluate ESM-2
    esm2_pred = evaluate_esm2(test_df, args.esm2_model, mhc_col, peptide_col)
    if esm2_pred is None:
        return
    
    # Calculate metrics
    convnext_metrics = calculate_metrics(y_true, convnext_pred, "ConvNeXt")
    esm2_metrics = calculate_metrics(y_true, esm2_pred, "ESM-2")
    
    # Print comparison
    print_comparison_table(convnext_metrics, esm2_metrics)
    
    # Save results
    results_df = pd.DataFrame({
        'true_pIC50': y_true,
        'convnext_pred': convnext_pred,
        'esm2_pred': esm2_pred,
        'convnext_error': np.abs(y_true - convnext_pred),
        'esm2_error': np.abs(y_true - esm2_pred)
    })
    results_df.to_csv('comparison_results.csv', index=False)
    print(f"\n✓ Detailed results saved to comparison_results.csv")
    
    # Create plot
    plot_comparison(y_true, convnext_pred, esm2_pred, args.output_plot)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    esm2_better_count = sum([
        esm2_metrics['RMSE'] < convnext_metrics['RMSE'],
        esm2_metrics['R²'] > convnext_metrics['R²'],
        esm2_metrics['PCC'] > convnext_metrics['PCC']
    ])
    
    if esm2_better_count >= 2:
        print("✓ ESM-2 outperforms ConvNeXt on this dataset!")
        print(f"  ESM-2 wins on {esm2_better_count}/3 key metrics (RMSE, R², PCC)")
    elif esm2_better_count == 1:
        print("~ ESM-2 and ConvNeXt have comparable performance")
        print("  Consider hybrid approach combining both models")
    else:
        print("✗ ConvNeXt performs better on this dataset")
        print("  ESM-2 may need fine-tuning or larger model size")
    
    print("\nFor research paper:")
    print(f"  ConvNeXt: RMSE={convnext_metrics['RMSE']:.3f}, PCC={convnext_metrics['PCC']:.3f}")
    print(f"  ESM-2:    RMSE={esm2_metrics['RMSE']:.3f}, PCC={esm2_metrics['PCC']:.3f}")
    print(f"  Improvement: {(convnext_metrics['RMSE'] - esm2_metrics['RMSE']) / convnext_metrics['RMSE'] * 100:.1f}% RMSE reduction")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
