#!/usr/bin/env python3
"""
Simplified diagnostic: Check binding affinity distribution and feature analysis.
Since PAE is computed in-model, let's first check if the raw affinity data has issues.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("PAE FEATURE CORRELATION DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Load training data
    train_csv = "datasets_alphafold_finetune/pmhc_finetune/final_train_70pct.csv"
    print(f"Loading training data from: {train_csv}")
    
    if not os.path.exists(train_csv):
        print(f"ERROR: Training data not found at {train_csv}")
        return
    
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training examples")
    
    # Calculate pIC50 from binding affinity (nM)
    df['pIC50'] = -np.log10(df['iedb_binding_affinity_nM'] * 1e-9)
    
    print(f"pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
    print(f"pIC50 mean: {df['pIC50'].mean():.2f}, std: {df['pIC50'].std():.2f}")
    print()
    
    # Extract PAE features for each example
    print("Extracting PAE features...")
    pae_stats_list = []
    pic50_values = []
    valid_count = 0
    
    for idx, row in df.iterrows():
        pdb_code = row['targetid']  # Use targetid column
        pic50 = row['pIC50']
        
        pae_matrix = load_pae_features(pdb_code)
        pae_stats = calculate_pae_stats(pae_matrix)
        
        if pae_stats is not None:
            pae_stats_list.append(pae_stats)
            pic50_values.append(pic50)
            valid_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} examples ({valid_count} valid)")
    
    print(f"\nTotal valid examples with PAE features: {valid_count}/{len(df)}")
    print()
    
    if valid_count == 0:
        print("ERROR: No valid PAE features found. Check feature file paths.")
        return
    
    # Convert to DataFrame for easy analysis
    pae_df = pd.DataFrame(pae_stats_list)
    pic50_array = np.array(pic50_values)
    
    # Calculate correlations for each PAE statistic
    print("=" * 80)
    print("CORRELATION ANALYSIS: PAE Features vs pIC50")
    print("=" * 80)
    print()
    
    correlations = []
    for col in pae_df.columns:
        pearson_r, pearson_p = pearsonr(pae_df[col], pic50_array)
        spearman_r, spearman_p = spearmanr(pae_df[col], pic50_array)
        
        correlations.append({
            'Feature': col,
            'Pearson_R': pearson_r,
            'Pearson_P': pearson_p,
            'Spearman_R': spearman_r,
            'Spearman_P': spearman_p
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Pearson_R', key=abs, ascending=False)
    
    print(f"{'Feature':<20} {'Pearson R':>10} {'P-value':>10} {'Spearman R':>10} {'P-value':>10}")
    print("-" * 80)
    for _, row in corr_df.iterrows():
        print(f"{row['Feature']:<20} {row['Pearson_R']:>10.4f} {row['Pearson_P']:>10.4e} "
              f"{row['Spearman_R']:>10.4f} {row['Spearman_P']:>10.4e}")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    max_abs_corr = corr_df['Pearson_R'].abs().max()
    best_feature = corr_df.iloc[0]['Feature']
    best_corr = corr_df.iloc[0]['Pearson_R']
    
    print(f"Best feature: {best_feature}")
    print(f"Best correlation: {best_corr:.4f}")
    print()
    
    if max_abs_corr < 0.1:
        print("⚠️  CRITICAL: Maximum correlation < 0.1")
        print("    PAE features show NEGLIGIBLE correlation with pIC50.")
        print("    This explains why regression fails - features don't contain the signal!")
        print()
        print("    RECOMMENDATIONS:")
        print("    1. Try different features: pLDDT, interface contacts, energy terms")
        print("    2. Use sequence embeddings: ESM-2, ProtBERT")
        print("    3. Consider that affinity prediction from static structure may be fundamentally limited")
    elif max_abs_corr < 0.2:
        print("⚠️  WARNING: Maximum correlation < 0.2")
        print("    PAE features show WEAK correlation with pIC50.")
        print("    Regression is possible but will be difficult.")
        print()
        print("    RECOMMENDATIONS:")
        print("    1. Try augmenting with additional features")
        print("    2. Consider ensemble methods")
        print("    3. May need much more training data")
    elif max_abs_corr < 0.4:
        print("✓  MODERATE: Correlation 0.2-0.4")
        print("   PAE features show moderate correlation with pIC50.")
        print("   Regression should be achievable with proper architecture.")
        print()
        print("   RECOMMENDATIONS:")
        print("   1. Check if architecture has sufficient capacity")
        print("   2. Try different activation functions")
        print("   3. Increase model depth or width")
    else:
        print("✓  GOOD: Correlation > 0.4")
        print("   PAE features show strong correlation with pIC50.")
        print("   The model should be able to learn this - check architecture/training.")
        print()
        print("   RECOMMENDATIONS:")
        print("   1. Debug gradient flow")
        print("   2. Check learning rate")
        print("   3. Verify no bugs in loss calculation")
    
    print()
    print("=" * 80)
    
    # Save results
    output_file = "pae_correlation_diagnostic.csv"
    corr_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
