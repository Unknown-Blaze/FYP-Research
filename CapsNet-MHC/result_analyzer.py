#!/usr/bin/env python3
"""
CapsNet-MHC Result Analyzer
This script reads weekly_result.txt and creates comprehensive visualizations and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve
)
from collections import Counter
import os

def read_weekly_results(file_path):
    """Read weekly_result.txt and return a pandas DataFrame"""
    print(f"Reading results from: {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split('\t')
    print(f"Columns: {header}")
    
    # Parse data
    data = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            data.append({
                'HLA': parts[0],
                'Peptide': parts[1], 
                'True_Label': float(parts[2]),
                'Length': int(parts[3]),
                'Predicted_Score': float(parts[4])
            })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} predictions")
    print(f"Peptide lengths: {sorted(df['Length'].unique())}")
    print(f"True labels distribution: {Counter(df['True_Label'])}")
    
    return df

def calculate_metrics(y_true, y_prob, y_pred=None):
    """Calculate comprehensive metrics"""
    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)
    
    # Basic confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MCC calculation
    try:
        mcc = ((tp * tn) - (fn * fp)) / np.sqrt((tp + fn) * (tn + fp) * (tp + fp) * (tn + fn))
    except:
        mcc = np.nan
    
    # AUC calculations
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, prec)
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc,
        'roc_auc': roc_auc, 'aupr': aupr
    }

def plot_metrics_by_length(df, save_path=None):
    """Create plots showing metrics by peptide length"""
    lengths = sorted(df['Length'].unique())
    
    # Calculate metrics for each length
    metrics_by_length = {}
    for length in lengths:
        subset = df[df['Length'] == length]
        if len(subset) > 0:
            y_true = subset['True_Label'].values
            y_prob = subset['Predicted_Score'].values
            y_pred = (y_prob > 0.5).astype(int)
            
            metrics = calculate_metrics(y_true, y_prob, y_pred)
            metrics_by_length[length] = metrics
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CapsNet-MHC Performance Metrics by Peptide Length', fontsize=16)
    
    # Plot 1: Sample counts
    lengths_list = list(metrics_by_length.keys())
    counts = [len(df[df['Length'] == l]) for l in lengths_list]
    axes[0, 0].bar(lengths_list, counts, color='skyblue')
    axes[0, 0].set_title('Sample Count by Length')
    axes[0, 0].set_xlabel('Peptide Length')
    axes[0, 0].set_ylabel('Number of Samples')
    
    # Plot 2: AUC
    aucs = [metrics_by_length[l]['roc_auc'] for l in lengths_list]
    axes[0, 1].bar(lengths_list, aucs, color='lightgreen')
    axes[0, 1].set_title('ROC AUC by Length')
    axes[0, 1].set_xlabel('Peptide Length')
    axes[0, 1].set_ylabel('ROC AUC')
    axes[0, 1].set_ylim(0.8, 1.0)
    
    # Plot 3: AUPR
    auprs = [metrics_by_length[l]['aupr'] for l in lengths_list]
    axes[0, 2].bar(lengths_list, auprs, color='orange')
    axes[0, 2].set_title('AUPR by Length')
    axes[0, 2].set_xlabel('Peptide Length')
    axes[0, 2].set_ylabel('AUPR')
    axes[0, 2].set_ylim(0.8, 1.0)
    
    # Plot 4: Accuracy
    accuracies = [metrics_by_length[l]['accuracy'] for l in lengths_list]
    axes[1, 0].bar(lengths_list, accuracies, color='lightcoral')
    axes[1, 0].set_title('Accuracy by Length')
    axes[1, 0].set_xlabel('Peptide Length')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0.7, 1.0)
    
    # Plot 5: F1 Score
    f1_scores = [metrics_by_length[l]['f1'] for l in lengths_list]
    axes[1, 1].bar(lengths_list, f1_scores, color='plum')
    axes[1, 1].set_title('F1 Score by Length')
    axes[1, 1].set_xlabel('Peptide Length')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim(0.7, 1.0)
    
    # Plot 6: MCC
    mccs = [metrics_by_length[l]['mcc'] for l in lengths_list]
    axes[1, 2].bar(lengths_list, mccs, color='gold')
    axes[1, 2].set_title('MCC by Length')
    axes[1, 2].set_xlabel('Peptide Length')
    axes[1, 2].set_ylabel('MCC')
    axes[1, 2].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_by_length

def plot_roc_curves(df, save_path=None):
    """Plot ROC curves for different peptide lengths"""
    lengths = sorted(df['Length'].unique())
    
    plt.figure(figsize=(12, 8))
    
    for length in lengths:
        subset = df[df['Length'] == length]
        if len(subset) > 0:
            y_true = subset['True_Label'].values
            y_prob = subset['Predicted_Score'].values
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            plt.plot(fpr, tpr, label=f'{length}-mer (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Peptide Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pr_curves(df, save_path=None):
    """Plot Precision-Recall curves for different peptide lengths"""
    lengths = sorted(df['Length'].unique())
    
    plt.figure(figsize=(12, 8))
    
    for length in lengths:
        subset = df[df['Length'] == length]
        if len(subset) > 0:
            y_true = subset['True_Label'].values
            y_prob = subset['Predicted_Score'].values
            
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            aupr_score = auc(rec, prec)
            
            plt.plot(rec, prec, label=f'{length}-mer (AUPR = {aupr_score:.3f})', linewidth=2)
    
    # Calculate baseline (proportion of positive samples)
    baseline = df['True_Label'].mean()
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Peptide Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_distributions(df, save_path=None):
    """Plot score distributions for positive and negative samples"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Predicted Score Distributions by Peptide Length', fontsize=16)
    
    lengths = sorted(df['Length'].unique())
    
    for i, length in enumerate(lengths):
        row = i // 3
        col = i % 3
        
        subset = df[df['Length'] == length]
        if len(subset) > 0:
            pos_scores = subset[subset['True_Label'] == 1]['Predicted_Score']
            neg_scores = subset[subset['True_Label'] == 0]['Predicted_Score']
            
            axes[row, col].hist(neg_scores, bins=50, alpha=0.7, label='Negative', color='red')
            axes[row, col].hist(pos_scores, bins=50, alpha=0.7, label='Positive', color='blue')
            axes[row, col].set_title(f'{length}-mer Score Distribution')
            axes[row, col].set_xlabel('Predicted Score')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(metrics_by_length):
    """Print a summary table of all metrics"""
    print("\n" + "="*80)
    print("CapsNet-MHC PERFORMANCE SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Length':<8} {'Samples':<10} {'AUC':<8} {'AUPR':<8} {'Acc':<8} {'F1':<8} {'MCC':<8} {'Sn':<8} {'Sp':<8}")
    print("-"*80)
    
    # Data rows
    for length in sorted(metrics_by_length.keys()):
        m = metrics_by_length[length]
        print(f"{length:<8} {m['tp']+m['tn']+m['fp']+m['fn']:<10} {m['roc_auc']:<8.3f} {m['aupr']:<8.3f} "
              f"{m['accuracy']:<8.3f} {m['f1']:<8.3f} {m['mcc']:<8.3f} {m['sensitivity']:<8.3f} {m['specificity']:<8.3f}")

def main():
    """Main function to analyze results"""
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # File paths
    anthem_results = "./codes/Anthem_codes/Anthem_test/weekly_result.txt"
    iedb_results = "./codes/IEDB_codes/IEDB_test/weekly_result.txt"
    
    # Check which file exists
    if os.path.exists(anthem_results):
        results_file = anthem_results
        print("Analyzing Anthem results...")
    elif os.path.exists(iedb_results):
        results_file = iedb_results
        print("Analyzing IEDB results...")
    else:
        print("No weekly_result.txt found. Please run the test script first.")
        return
    
    # Read data
    df = read_weekly_results(results_file)
    
    # Calculate overall metrics
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE")
    print("="*50)
    
    y_true = df['True_Label'].values
    y_prob = df['Predicted_Score'].values
    y_pred = (y_prob > 0.5).astype(int)
    
    overall_metrics = calculate_metrics(y_true, y_prob, y_pred)
    
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
    print(f"Negative samples: {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {overall_metrics['tn']}, FP: {overall_metrics['fp']}")
    print(f"  FN: {overall_metrics['fn']}, TP: {overall_metrics['tp']}")
    print(f"\nMetrics:")
    print(f"  ROC AUC: {overall_metrics['roc_auc']:.4f}")
    print(f"  AUPR: {overall_metrics['aupr']:.4f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {overall_metrics['f1']:.4f}")
    print(f"  MCC: {overall_metrics['mcc']:.4f}")
    print(f"  Sensitivity: {overall_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {overall_metrics['specificity']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall: {overall_metrics['recall']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Metrics by length
    metrics_by_length = plot_metrics_by_length(df, "metrics_by_length.png")
    
    # ROC curves
    plot_roc_curves(df, "roc_curves.png")
    
    # PR curves
    plot_pr_curves(df, "pr_curves.png")
    
    # Score distributions
    plot_score_distributions(df, "score_distributions.png")
    
    # Summary table
    print_summary_table(metrics_by_length)
    
    print(f"\nPlots saved as:")
    print(f"  - metrics_by_length.png")
    print(f"  - roc_curves.png")
    print(f"  - pr_curves.png")
    print(f"  - score_distributions.png")

if __name__ == "__main__":
    main()

