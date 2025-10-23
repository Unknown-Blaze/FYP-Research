#!/usr/bin/env python3
"""
Quick diagnostic: Analyze pIC50 distribution and check if the problem is in the data.
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 80)
    print("QUICK pIC50 DATA ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    train_csv = "datasets_alphafold_finetune/pmhc_finetune/final_train_70pct.csv"
    valid_csv = "datasets_alphafold_finetune/pmhc_finetune/final_valid_10pct.csv"
    
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    
    # Calculate pIC50
    df_train['pIC50'] = -np.log10(df_train['iedb_binding_affinity_nM'] * 1e-9)
    df_valid['pIC50'] = -np.log10(df_valid['iedb_binding_affinity_nM'] * 1e-9)
    
    print(f"Training set: {len(df_train)} examples")
    print(f"Validation set: {len(df_valid)} examples")
    print()
    
    # Statistics
    print("TRAINING SET pIC50 STATISTICS:")
    print(f"  Mean: {df_train['pIC50'].mean():.2f}")
    print(f"  Std: {df_train['pIC50'].std():.2f}")
    print(f"  Min: {df_train['pIC50'].min():.2f}")
    print(f"  Max: {df_train['pIC50'].max():.2f}")
    print(f"  Median: {df_train['pIC50'].median():.2f}")
    print()
    
    # Check bin distribution
    bins = [0, 5.0, 7.0, 9.0, 100]
    bin_labels = ['<5.0', '5-7', '7-9', '>=9']
    df_train['bin'] = pd.cut(df_train['pIC50'], bins=bins, labels=bin_labels)
    
    print("BIN DISTRIBUTION (Training):")
    for label in bin_labels:
        count = (df_train['bin'] == label).sum()
        pct = 100 * count / len(df_train)
        print(f"  Bin {label}: {count:5d} examples ({pct:5.1f}%)")
    print()
    
    # Check if binder_class matches pIC50
    if 'binder_class' in df_train.columns:
        print("BINDER_CLASS vs pIC50 ANALYSIS:")
        for bc in sorted(df_train['binder_class'].unique()):
            subset = df_train[df_train['binder_class'] == bc]
            print(f"  Binder class {bc}:")
            print(f"    Count: {len(subset)}")
            print(f"    pIC50 mean: {subset['pIC50'].mean():.2f} ± {subset['pIC50'].std():.2f}")
            print(f"    pIC50 range: [{subset['pIC50'].min():.2f}, {subset['pIC50'].max():.2f}]")
        print()
    
    # Check validation predictions from the model
    print("=" * 80)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Look for validation results
    import glob
    import os
    
    # Find most recent ordinal-only log
    log_files = glob.glob("quick_test_ordinal_only.log") + glob.glob("quick_test*.log")
    
    if not log_files:
        print("No validation logs found. Run quick_test.sh first.")
        return
    
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Analyzing latest log: {latest_log}")
    print()
    
    # Parse predictions from log
    with open(latest_log, 'r') as f:
        lines = f.readlines()
    
    predictions = []
    true_values = []
    ordinal_preds = []
    
    for line in lines:
        if 'Prediction:' in line and 'True:' in line:
            try:
                # Parse: "Prediction: X.XX, True: Y.YY, ordinal_pred: Z"
                parts = line.split(',')
                pred = float(parts[0].split(':')[1].strip())
                true = float(parts[1].split(':')[1].strip())
                if 'ordinal_pred:' in line:
                    ord_pred = int(parts[2].split(':')[1].strip())
                    ordinal_preds.append(ord_pred)
                predictions.append(pred)
                true_values.append(true)
            except:
                continue
    
    if len(predictions) == 0:
        print("Could not parse predictions from log file.")
        return
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    ordinal_preds = np.array(ordinal_preds)
    
    print(f"Parsed {len(predictions)} predictions")
    print()
    
    print("MODEL PREDICTIONS:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std: {predictions.std():.4f}  <-- THIS IS THE PROBLEM!")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")
    print(f"  Range: {predictions.max() - predictions.min():.2f}")
    print()
    
    print("TRUE VALUES:")
    print(f"  Mean: {true_values.mean():.2f}")
    print(f"  Std: {true_values.std():.2f}")
    print(f"  Min: {true_values.min():.2f}")
    print(f"  Max: {true_values.max():.2f}")
    print(f"  Range: {true_values.max() - true_values.min():.2f}")
    print()
    
    if len(ordinal_preds) > 0:
        print("ORDINAL PREDICTIONS:")
        for i in range(4):
            count = (ordinal_preds == i).sum()
            pct = 100 * count / len(ordinal_preds)
            print(f"  Bin {i}: {count:3d} predictions ({pct:5.1f}%)")
        print()
        
        # Check what true bins correspond to predicted bin 3
        if (ordinal_preds == 3).sum() > 0:
            true_bins_for_pred3 = []
            for i, (pred, true) in enumerate(zip(ordinal_preds, true_values)):
                if pred == 3:
                    if true < 5.0:
                        true_bins_for_pred3.append(0)
                    elif true < 7.0:
                        true_bins_for_pred3.append(1)
                    elif true < 9.0:
                        true_bins_for_pred3.append(2)
                    else:
                        true_bins_for_pred3.append(3)
            
            print("When model predicts bin 3, true bins are:")
            for i in range(4):
                count = true_bins_for_pred3.count(i)
                pct = 100 * count / len(true_bins_for_pred3)
                print(f"  True bin {i}: {count:3d} ({pct:5.1f}%)")
            print()
    
    # Correlation
    from scipy.stats import pearsonr
    corr, pval = pearsonr(predictions, true_values)
    print(f"Correlation: {corr:.4f} (p={pval:.4e})")
    print()
    
    # Key insight
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    
    if predictions.std() < 0.1:
        print("⚠️  CATASTROPHIC COLLAPSE: Prediction std < 0.1")
        print()
        print("The model is predicting nearly constant values (~{:.2f}).".format(predictions.mean()))
        print("This means the model has learned NOTHING about pIC50 variation.")
        print()
        
        if len(ordinal_preds) > 0 and (ordinal_preds == 3).sum() > len(ordinal_preds) * 0.9:
            print("The ordinal classifier predicts bin 3 for >90% of examples.")
            print("This suggests:")
            print("  1. The features don't contain signal to distinguish bins 0-2 from bin 3")
            print("  2. OR the model architecture can't learn from the features")
            print("  3. OR the loss/training is fundamentally broken")
            print()
            
            # Check if bin 3 is the majority class
            if (df_train['bin'] == '>=9').sum() < len(df_train) * 0.2:
                print(f"Bin 3 (>=9) is only {100*(df_train['bin']=='>=9').sum()/len(df_train):.1f}% of training data.")
                print("Model is NOT just predicting majority class.")
                print("It's predicting near the MEAN of the distribution.")
                print()
                print("CONCLUSION: Model defaults to dataset mean, indicating:")
                print("  → Features likely don't contain pIC50 discrimination signal")
                print("  → PAE may encode binding YES/NO but not STRENGTH")
                print()
    else:
        print(f"Prediction std = {predictions.std():.2f} (non-collapsed)")
        print("Model is learning SOME variation, but check correlation.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
