#!/usr/bin/env python3
"""
Script to parse training logs and generate visualization graphs.
Handles both old format (train_*.log) and new format (training.log) logs.
"""

import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EpochMetrics:
    """Data class to store metrics for each epoch."""
    epoch: int
    train_loss: float
    valid_loss: float
    valid_rmse: float
    valid_r2: float
    is_best: bool = False


@dataclass
class TestMetrics:
    """Data class to store final test metrics."""
    loss: float
    rmse: float
    r2: float
    pcc: float
    scc: float
    best_epoch: Optional[int] = None


@dataclass
class TrainingRun:
    """Data class to store complete training run information."""
    checkpoint_name: str
    log_file: str
    epochs: List[EpochMetrics]
    test_metrics: TestMetrics
    model_name: Optional[str] = None
    total_epochs: Optional[int] = None


def parse_test_results(content: str) -> Optional[TestMetrics]:
    """Parse test results from log content, handling multiple formats."""
    # Try to extract test section (case-insensitive)
    test_section_match = re.search(r'TEST RESULTS.*?(?=\n\n|Training complete|Run finished|Early stopping|$)', content, re.DOTALL | re.IGNORECASE)
    
    if not test_section_match:
        return None
    
    test_section = test_section_match.group(0)
    
    # Extract individual metrics with flexible patterns
    loss_match = re.search(r'Loss:\s*([\d.]+)', test_section)
    rmse_match = re.search(r'RMSE:\s*([\d.]+)', test_section)
    r2_match = re.search(r'R²:\s*([\d.]+)', test_section)
    pcc_match = re.search(r'PCC:\s*([\d.]+)', test_section)
    scc_match = re.search(r'SCC:\s*([\d.]+)', test_section)
    
    # Check if we have at least RMSE, R², and PCC (minimum required)
    if rmse_match and r2_match and pcc_match:
        return TestMetrics(
            loss=float(loss_match.group(1)) if loss_match else 0.0,
            rmse=float(rmse_match.group(1)),
            r2=float(r2_match.group(1)),
            pcc=float(pcc_match.group(1)),
            scc=float(scc_match.group(1)) if scc_match else float(pcc_match.group(1))  # Use PCC as fallback
        )
    
    return None


def parse_old_format_log(log_path: str) -> Optional[TrainingRun]:
    """
    Parse log files in old format (like train_2025-10-21_13-11-30.log).
    
    Format example:
    Epoch 1/50
      Train Loss: 0.0275 (scaled)
      Valid Loss: 0.0214 (scaled), Valid RMSE: 1.0856, Valid R²: 0.4070
    """
    checkpoint_name = Path(log_path).parent.name
    epochs = []
    test_metrics = None
    best_epoch = None
    model_name = None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract model name
    model_match = re.search(r'model_name[=:]\s*["\']?([^"\')\s,]+)', content)
    if model_match:
        model_name = model_match.group(1).split('/')[-1]  # Get just the model name
    
    # Parse epoch data
    epoch_pattern = re.compile(
        r'Epoch (\d+)/(\d+)\s+'
        r'Train Loss: ([\d.]+) \(scaled\)\s+'
        r'Valid Loss: ([\d.]+) \(scaled\), Valid RMSE: ([\d.]+), Valid R²: ([\d.]+)'
    )
    
    # Find all epoch matches
    epoch_matches = epoch_pattern.findall(content)
    
    # Check for saved model indicators - find all positions
    saved_pattern = re.compile(r'✓ Saved new best model \(RMSE: ([\d.]+)\)')
    saved_rmses = set(float(m.group(1)) for m in saved_pattern.finditer(content))
    
    for match in epoch_matches:
        epoch = int(match[0])
        total_epochs = int(match[1])
        train_loss = float(match[2])
        valid_loss = float(match[3])
        valid_rmse = float(match[4])
        valid_r2 = float(match[5])
        
        # Check if this RMSE was saved as a best model
        is_best = valid_rmse in saved_rmses
        
        epochs.append(EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_rmse=valid_rmse,
            valid_r2=valid_r2,
            is_best=is_best
        ))
    
    # Parse test results
    test_metrics = parse_test_results(content)
    
    # Extract best epoch
    best_epoch_match = re.search(r'best model.*?from epoch (\d+)', content, re.IGNORECASE)
    if best_epoch_match:
        best_epoch = int(best_epoch_match.group(1))
        if test_metrics:
            test_metrics.best_epoch = best_epoch
    
    if epochs and test_metrics:
        return TrainingRun(
            checkpoint_name=checkpoint_name,
            log_file=log_path,
            epochs=epochs,
            test_metrics=test_metrics,
            model_name=model_name,
            total_epochs=len(epochs)
        )
    
    return None


def parse_new_format_log(log_path: str) -> Optional[TrainingRun]:
    """
    Parse log files in new format (training.log with timestamps).
    
    Format example:
    Epoch 1/50
    2025-10-23 12:00:12,769 [MainThread  ] [INFO ]    Train Loss: 0.0275 (scaled)
    2025-10-23 12:00:12,769 [MainThread  ] [INFO ]    Valid Loss: 0.0214 (scaled), Valid RMSE: 1.0856, Valid R²: 0.6930
    """
    checkpoint_name = Path(log_path).parent.name
    epochs = []
    test_metrics = None
    best_epoch = None
    model_name = None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract model name
    model_match = re.search(r'"model_name":\s*"([^"]+)"', content)
    if model_match:
        model_name = model_match.group(1).split('/')[-1]
    
    # Parse epoch data - use a combined pattern
    # This pattern matches the epoch header followed by train and valid lines (anywhere in between)
    # Format 1: Train Loss: X (scaled), Valid Loss: Y (scaled), Valid RMSE: Z, Valid R²: W
    # Format 2: Train Loss: X, Valid Loss: Y, RMSE: Z, R²: W
    # Format 3: Train Loss: X, Valid RMSE: Y, R²: Z, PCC: W (warmstart format)
    epoch_block_pattern_v1 = re.compile(
        r'Epoch (\d+)/(\d+).*?'
        r'Train Loss: ([\d.]+) \(scaled\).*?'
        r'Valid Loss: ([\d.]+) \(scaled\), Valid RMSE: ([\d.]+), Valid R²: ([\d.]+)',
        re.DOTALL
    )
    
    epoch_block_pattern_v2 = re.compile(
        r'Epoch (\d+)/(\d+).*?'
        r'Train Loss: ([\d.]+)(?!\)).*?'  # Not followed by )
        r'Valid Loss: ([\d.]+),\s*RMSE: ([\d.]+),\s*R²: ([\d.]+)',
        re.DOTALL
    )
    
    # Warmstart format: no Valid Loss, just RMSE and R²
    epoch_block_pattern_v3 = re.compile(
        r'Epoch (\d+)/(\d+).*?'
        r'Train Loss: ([\d.]+)(?!\)).*?'  # Not followed by )
        r'Valid RMSE: ([\d.]+),\s*R²: ([\d.]+)',
        re.DOTALL
    )
    
    # Find all epoch blocks - try all patterns
    epoch_matches = epoch_block_pattern_v1.findall(content)
    if not epoch_matches:
        epoch_matches = epoch_block_pattern_v2.findall(content)
    if not epoch_matches:
        # For v3, we need to adjust the tuple structure since it has one less element
        matches_v3 = epoch_block_pattern_v3.findall(content)
        if matches_v3:
            # Insert 0.0 as placeholder for valid_loss (index 3)
            epoch_matches = [(m[0], m[1], m[2], '0.0', m[3], m[4]) for m in matches_v3]
    
    # Check for saved model indicators
    saved_pattern = re.compile(r'✓ Saved new best model \(RMSE: ([\d.]+)\)')
    saved_rmses = set(float(m.group(1)) for m in saved_pattern.finditer(content))
    
    for match in epoch_matches:
        epoch = int(match[0])
        total_epochs = int(match[1])
        train_loss = float(match[2])
        valid_loss = float(match[3])
        valid_rmse = float(match[4])
        valid_r2 = float(match[5])
        
        # Check if this RMSE was saved as a best model
        is_best = valid_rmse in saved_rmses
        
        epochs.append(EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_rmse=valid_rmse,
            valid_r2=valid_r2,
            is_best=is_best
        ))
    
    # Parse test results
    test_metrics = parse_test_results(content)
    
    # Extract best epoch
    best_epoch_match = re.search(r'best model.*?from epoch (\d+)', content, re.IGNORECASE)
    if best_epoch_match:
        best_epoch = int(best_epoch_match.group(1))
        if test_metrics:
            test_metrics.best_epoch = best_epoch
    
    if epochs and test_metrics:
        return TrainingRun(
            checkpoint_name=checkpoint_name,
            log_file=log_path,
            epochs=epochs,
            test_metrics=test_metrics,
            model_name=model_name,
            total_epochs=len(epochs)
        )
    
    return None


def parse_log_file(log_path: str) -> Optional[TrainingRun]:
    """Automatically detect format and parse log file."""
    if 'training.log' in os.path.basename(log_path):
        return parse_new_format_log(log_path)
    else:
        return parse_old_format_log(log_path)


def plot_training_curves(run: TrainingRun, output_dir: str):
    """Generate comprehensive training visualization for a single run."""
    
    # Extract data
    epochs_num = [e.epoch for e in run.epochs]
    train_losses = [e.train_loss for e in run.epochs]
    valid_losses = [e.valid_loss for e in run.epochs]
    valid_rmses = [e.valid_rmse for e in run.epochs]
    valid_r2s = [e.valid_r2 for e in run.epochs]
    best_epochs = [e.epoch for e in run.epochs if e.is_best]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Metrics: {run.checkpoint_name}', fontsize=16, fontweight='bold')
    
    # Add model name and test results as subtitle
    subtitle = f'Model: {run.model_name or "Unknown"}'
    if run.test_metrics:
        subtitle += f' | Test RMSE: {run.test_metrics.rmse:.4f} | Test R²: {run.test_metrics.r2:.4f}'
        if run.test_metrics.best_epoch:
            subtitle += f' | Best Epoch: {run.test_metrics.best_epoch}'
    plt.figtext(0.5, 0.96, subtitle, ha='center', fontsize=11, style='italic')
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs_num, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_num, valid_losses, 'r-', label='Valid Loss', linewidth=2)
    for be in best_epochs:
        ax1.axvline(x=be, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (scaled)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE over epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs_num, valid_rmses, 'g-', linewidth=2)
    for be in best_epochs:
        ax2.axvline(x=be, color='green', linestyle='--', alpha=0.3, linewidth=1)
    if run.test_metrics:
        ax2.axhline(y=run.test_metrics.rmse, color='red', linestyle='--', 
                    label=f'Test RMSE: {run.test_metrics.rmse:.4f}', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Validation RMSE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. R² over epochs
    ax3 = axes[1, 0]
    ax3.plot(epochs_num, valid_r2s, 'purple', linewidth=2)
    for be in best_epochs:
        ax3.axvline(x=be, color='green', linestyle='--', alpha=0.3, linewidth=1)
    if run.test_metrics:
        ax3.axhline(y=run.test_metrics.r2, color='red', linestyle='--', 
                    label=f'Test R²: {run.test_metrics.r2:.4f}', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Validation R² Score', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Test metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if run.test_metrics:
        metrics_text = [
            "Final Test Results",
            "=" * 40,
            f"RMSE:  {run.test_metrics.rmse:.4f}",
            f"R²:    {run.test_metrics.r2:.4f}",
            f"PCC:   {run.test_metrics.pcc:.4f}",
            f"SCC:   {run.test_metrics.scc:.4f}" + (" *" if run.test_metrics.scc == run.test_metrics.pcc else ""),
            f"Loss:  {run.test_metrics.loss:.4f}" + (" (scaled)" if run.test_metrics.loss > 0 else " (N/A)"),
            "",
            f"Best Epoch: {run.test_metrics.best_epoch or 'N/A'}",
            f"Total Epochs: {run.total_epochs}",
            "",
            "Best validation checkpoints shown",
            "as green dashed lines",
        ]
        
        if run.test_metrics.scc == run.test_metrics.pcc:
            metrics_text.append("")
            metrics_text.append("* SCC not in log, using PCC")
        
        text_str = '\n'.join(metrics_text)
        ax4.text(0.1, 0.5, text_str, transform=ax4.transAxes,
                fontsize=13, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_path = os.path.join(output_dir, f'{run.checkpoint_name}_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: {output_path}')
    plt.close()


def plot_comparison_charts(runs: List[TrainingRun], output_dir: str):
    """Generate comparison charts across all training runs."""
    
    if len(runs) < 2:
        print("  ! Need at least 2 runs for comparison charts")
        return
    
    # Create figure with 2x2 comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
    
    # 1. Final RMSE comparison
    ax1 = axes[0, 0]
    names = [run.checkpoint_name for run in runs]
    rmses = [run.test_metrics.rmse for run in runs]
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))
    bars = ax1.bar(range(len(names)), rmses, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, rmses)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Final R² comparison
    ax2 = axes[0, 1]
    r2s = [run.test_metrics.r2 for run in runs]
    bars = ax2.bar(range(len(names)), r2s, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Test R² Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, r2s)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Correlation metrics (PCC & SCC)
    ax3 = axes[1, 0]
    pccs = [run.test_metrics.pcc for run in runs]
    sccs = [run.test_metrics.scc for run in runs]
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, pccs, width, label='PCC', alpha=0.8)
    ax3.bar(x + width/2, sccs, width, label='SCC', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Correlation Score', fontsize=12)
    ax3.set_title('Correlation Metrics (PCC & SCC)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. RMSE progression for all runs
    ax4 = axes[1, 1]
    for run in runs:
        epochs_num = [e.epoch for e in run.epochs]
        valid_rmses = [e.valid_rmse for e in run.epochs]
        ax4.plot(epochs_num, valid_rmses, label=run.checkpoint_name, linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation RMSE', fontsize=12)
    ax4.set_title('RMSE Progression - All Runs', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_path = os.path.join(output_dir, 'comparison_all_runs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: {output_path}')
    plt.close()


def generate_summary_table(runs: List[TrainingRun], output_dir: str):
    """Generate a summary table of all runs."""
    
    # Create summary text file
    output_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("TRAINING RUNS SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        
        # Header
        f.write(f"{'Checkpoint':<40} {'Model':<30} {'Epochs':<8} {'RMSE':<10} {'R²':<10} {'PCC':<10} {'SCC':<10}\n")
        f.write("-" * 120 + "\n")
        
        # Sort by RMSE (lower is better)
        sorted_runs = sorted(runs, key=lambda x: x.test_metrics.rmse)
        
        for i, run in enumerate(sorted_runs):
            rank_marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            model_name = (run.model_name or "Unknown")[:28]
            
            f.write(f"{rank_marker} {run.checkpoint_name:<38} {model_name:<30} "
                   f"{run.total_epochs:<8} {run.test_metrics.rmse:<10.4f} "
                   f"{run.test_metrics.r2:<10.4f} {run.test_metrics.pcc:<10.4f} "
                   f"{run.test_metrics.scc:<10.4f}\n")
        
        f.write("\n" + "=" * 120 + "\n\n")
        
        # Best performers
        f.write("BEST PERFORMERS:\n")
        f.write("-" * 120 + "\n")
        best_rmse = min(runs, key=lambda x: x.test_metrics.rmse)
        best_r2 = max(runs, key=lambda x: x.test_metrics.r2)
        best_pcc = max(runs, key=lambda x: x.test_metrics.pcc)
        
        f.write(f"Best RMSE: {best_rmse.checkpoint_name} ({best_rmse.test_metrics.rmse:.4f})\n")
        f.write(f"Best R²:   {best_r2.checkpoint_name} ({best_r2.test_metrics.r2:.4f})\n")
        f.write(f"Best PCC:  {best_pcc.checkpoint_name} ({best_pcc.test_metrics.pcc:.4f})\n")
        
        f.write("\n" + "=" * 120 + "\n")
    
    print(f'  ✓ Saved: {output_path}')
    
    # Also print to console
    with open(output_path, 'r') as f:
        print("\n" + f.read())


def main():
    """Main function to process all logs and generate visualizations."""
    
    # Set up paths
    checkpoints_dir = Path(__file__).parent / 'checkpoints'
    output_dir = Path(__file__).parent / 'training_visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Training Log Visualization Script")
    print("=" * 80)
    print(f"\nSearching for log files in: {checkpoints_dir}")
    
    # Find all log files
    log_files = list(checkpoints_dir.glob('*/train_*.log')) + list(checkpoints_dir.glob('*/training.log'))
    print(f"Found {len(log_files)} log file(s)\n")
    
    # Parse all logs
    runs = []
    print("Parsing log files...")
    for log_file in sorted(log_files):
        print(f"  Processing: {log_file.parent.name}/{log_file.name}")
        try:
            run = parse_log_file(str(log_file))
            if run:
                runs.append(run)
                print(f"    ✓ Parsed {len(run.epochs)} epochs, Test RMSE: {run.test_metrics.rmse:.4f}")
            else:
                # Try to give more info about why it failed
                with open(log_file, 'r') as f:
                    content = f.read()
                if 'Test Results' not in content:
                    print(f"    ✗ Incomplete run (no test results found)")
                elif 'Epoch' not in content:
                    print(f"    ✗ No epoch data found")
                else:
                    print(f"    ✗ Failed to parse (unknown reason)")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    if not runs:
        print("\n✗ No valid training runs found!")
        return
    
    print(f"\n✓ Successfully parsed {len(runs)} training run(s)\n")
    
    # Generate individual plots
    print("Generating individual training curves...")
    for run in runs:
        plot_training_curves(run, str(output_dir))
    
    # Generate comparison charts
    print("\nGenerating comparison charts...")
    plot_comparison_charts(runs, str(output_dir))
    
    # Generate summary table
    print("\nGenerating summary table...")
    generate_summary_table(runs, str(output_dir))
    
    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
