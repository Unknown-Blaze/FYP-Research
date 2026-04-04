#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split


PLOT_DPI = 300
BINDER_THRESHOLD = 6.3
RANDOM_STATE = 42


def ensure_out_dir(viz_root: Path) -> Path:
    out_dir = viz_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_results(repo_root: Path) -> pd.DataFrame:
    csv_path = repo_root / "final_ensemble_test_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def choose_sequence_prediction_column(df: pd.DataFrame) -> str:
    preferred = ["esm_pred", "sequence_pred", "seq_pred", "pred_sequence", "base_pred"]
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        col_l = col.lower()
        if "pred" in col_l and ("esm" in col_l or "seq" in col_l):
            return col

    raise ValueError(
        "Could not find a sequence prediction column (e.g., esm_pred) for allele stratification plot."
    )


def infer_allele_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["allele", "mhc_allele", "hla", "hla_allele"]:
        if candidate in df.columns:
            return candidate
    return None


def recover_alleles_from_split(repo_root: Path, final_df: pd.DataFrame) -> pd.Series:
    meta_path = repo_root / "ensemble_meta_data.csv"
    iedb_path = repo_root / "iedb_dataset.csv"

    if not meta_path.exists() or not iedb_path.exists():
        raise FileNotFoundError(
            "Need ensemble_meta_data.csv and iedb_dataset.csv to recover alleles when allele column is absent."
        )

    meta_df = pd.read_csv(meta_path).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    iedb_df = pd.read_csv(iedb_path).dropna(subset=["mhc_sequence", "peptide"]).reset_index(drop=True)

    if len(meta_df) != len(iedb_df):
        raise ValueError(
            "Row count mismatch between ensemble_meta_data.csv and cleaned iedb_dataset.csv; cannot recover alleles safely."
        )

    _, test_idx = train_test_split(
        np.arange(len(meta_df)), test_size=0.1, random_state=RANDOM_STATE
    )

    reconstructed = meta_df.iloc[test_idx].reset_index(drop=True)
    compare_cols = [c for c in ["esm_pred", "gnn_pred", "actual_pic50", "peptide_len", "plddt", "diff"] if c in final_df.columns]
    if not compare_cols:
        raise ValueError("Cannot validate reconstructed test split because required comparison columns are missing.")

    if len(reconstructed) != len(final_df):
        raise ValueError("Reconstructed test split length does not match final_ensemble_test_results.csv length.")

    left = reconstructed[compare_cols].to_numpy(dtype=float)
    right = final_df[compare_cols].to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True):
        raise ValueError(
            "Reconstructed split does not match final_ensemble_test_results.csv. Unable to assign alleles confidently."
        )

    allele_col = "mhc_allele" if "mhc_allele" in iedb_df.columns else "allele"
    if allele_col not in iedb_df.columns:
        raise ValueError("No allele-like column found in iedb_dataset.csv (expected 'mhc_allele' or 'allele').")

    return iedb_df.iloc[test_idx][allele_col].reset_index(drop=True)


def compute_partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(fpr) == 0:
        return float("nan")

    idx = np.searchsorted(fpr, max_fpr, side="right")
    fpr_clip = fpr[:idx]
    tpr_clip = tpr[:idx]

    if len(fpr_clip) == 0 or fpr_clip[-1] < max_fpr:
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr_clip = np.append(fpr_clip, max_fpr)
        tpr_clip = np.append(tpr_clip, tpr_at_max)

    return float(np.trapz(tpr_clip, fpr_clip))


def plot_density_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    req = {"actual_pic50", "pred_linear"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns for density scatter: {sorted(req - set(df.columns))}")

    x = df["actual_pic50"].to_numpy(dtype=float)
    y = df["pred_linear"].to_numpy(dtype=float)
    pcc, _ = pearsonr(x, y)

    xy = np.vstack([x, y])
    try:
        density = gaussian_kde(xy)(xy)
    except Exception:
        density = np.ones_like(x)

    order = np.argsort(density)
    x_sorted, y_sorted, d_sorted = x[order], y[order], density[order]

    plt.figure(figsize=(7.5, 6.5))
    sc = plt.scatter(x_sorted, y_sorted, c=d_sorted, s=18, cmap="viridis", alpha=0.85, edgecolor="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Point density")

    lim_min = float(min(np.min(x), np.min(y)))
    lim_max = float(max(np.max(x), np.max(y)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.5, label="Perfect prediction ($y=x$)")

    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50 (Linear Ensemble)")
    plt.title(f"Density Scatter: Actual vs Predicted (PCC = {pcc:.4f})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "density_scatter_ensemble.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_clinical_roc(df: pd.DataFrame, out_dir: Path) -> None:
    req = {"actual_pic50", "pred_linear"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns for ROC plot: {sorted(req - set(df.columns))}")

    y_true = (df["actual_pic50"].to_numpy(dtype=float) >= BINDER_THRESHOLD).astype(int)
    y_score = df["pred_linear"].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_01 = compute_partial_auc(fpr, tpr, max_fpr=0.1)

    tpr_01 = np.interp(0.1, fpr, tpr)
    mask = fpr <= 0.1

    plt.figure(figsize=(7.5, 6.5))
    plt.plot(fpr, tpr, linewidth=2.0, label="ROC (pred_linear)")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.8)

    fpr_fill = np.append(fpr[mask], 0.1) if np.any(mask) and fpr[mask][-1] < 0.1 else fpr[mask]
    tpr_fill = np.append(tpr[mask], tpr_01) if np.any(mask) and fpr[mask][-1] < 0.1 else tpr[mask]
    if len(fpr_fill) > 1:
        plt.fill_between(fpr_fill, tpr_fill, alpha=0.25, color="tab:blue", label=r"Partial AUC region ($FPR\leq0.1$)")

    plt.axvline(0.1, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.9)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Clinical ROC Curve (Binder threshold: pIC50 ≥ 6.3)")
    plt.legend(loc="lower right")

    textbox = (
        rf"$AUC_{{0.1}}$ = {auc_01:.5f}" + "\n" +
        rf"Max possible area (0-0.1): {0.1:.1f}"
    )
    plt.text(
        0.58,
        0.17,
        textbox,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(out_dir / "clinical_roc_curve.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_allele_stratification(df: pd.DataFrame, out_dir: Path, repo_root: Path) -> None:
    req = {"actual_pic50", "pred_linear"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns for allele stratification: {sorted(req - set(df.columns))}")

    sequence_pred_col = choose_sequence_prediction_column(df)

    allele_col = infer_allele_column(df)
    if allele_col is None:
        df = df.copy()
        df["allele"] = recover_alleles_from_split(repo_root, df)
        allele_col = "allele"

    allele_counts = df[allele_col].value_counts(dropna=False)
    df = df.copy()
    df["allele_group"] = np.where(df[allele_col].map(allele_counts) < 10, "Rare (<10)", "Common (≥10)")

    rows = []
    for group_name in ["Rare (<10)", "Common (≥10)"]:
        subset = df[df["allele_group"] == group_name]
        if subset.empty:
            seq_rmse = np.nan
            ens_rmse = np.nan
        else:
            seq_rmse = mean_squared_error(subset["actual_pic50"], subset[sequence_pred_col], squared=False)
            ens_rmse = mean_squared_error(subset["actual_pic50"], subset["pred_linear"], squared=False)

        rows.append({"Allele Group": group_name, "Model": sequence_pred_col, "RMSE": seq_rmse})
        rows.append({"Allele Group": group_name, "Model": "pred_linear", "RMSE": ens_rmse})

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(8.5, 6.0))
    ax = sns.barplot(data=plot_df, x="Allele Group", y="RMSE", hue="Model", palette="Set2")
    ax.set_title("RMSE on Rare vs Common Alleles")
    ax.set_xlabel("Allele Frequency Group")
    ax.set_ylabel("RMSE")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(out_dir / "allele_stratification_barplot.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def plot_peptide_length_error(df: pd.DataFrame, out_dir: Path) -> None:
    req = {"actual_pic50", "pred_linear", "peptide_len"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns for peptide length error boxplot: {sorted(req - set(df.columns))}")

    plot_df = df.copy()
    plot_df["abs_error"] = (plot_df["actual_pic50"] - plot_df["pred_linear"]).abs()
    plot_df = plot_df[plot_df["peptide_len"].isin([8, 9, 10, 11])]

    plt.figure(figsize=(8.5, 6.0))
    sns.boxplot(
        data=plot_df,
        x="peptide_len",
        y="abs_error",
        order=[8, 9, 10, 11],
        showfliers=True,
    )
    plt.xlabel("Peptide Length")
    plt.ylabel("Absolute Error |actual - pred_linear|")
    plt.title("Prediction Error Distribution Across Peptide Lengths")
    plt.tight_layout()
    plt.savefig(out_dir / "peptide_length_error_boxplot.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def parse_epoch_metrics_from_log(log_content: str) -> Tuple[List[int], List[float], List[float], Optional[int], Optional[str]]:
    epochs: List[int] = []
    train_losses: List[float] = []
    val_rmses: List[float] = []

    epoch_pattern = re.compile(r"Epoch\s+(\d+)(?:\s*/\s*\d+)?", re.IGNORECASE)
    train_pattern = re.compile(r"Train\s+Loss\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    val_pattern = re.compile(r"(?:Val|Valid)(?:id)?\s+RMSE\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    current_epoch = None
    current_train = None

    for line in log_content.splitlines():
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        train_match = train_pattern.search(line)
        if train_match:
            current_train = float(train_match.group(1))

        val_match = val_pattern.search(line)
        if val_match and current_epoch is not None and current_train is not None:
            epochs.append(current_epoch)
            train_losses.append(current_train)
            val_rmses.append(float(val_match.group(1)))
            current_train = None

    marker_epoch = None
    marker_label = None
    marker_patterns = [
        (re.compile(r"Early\s+Stopping.*?epoch\s*(\d+)", re.IGNORECASE), "Early stopping"),
        (re.compile(r"best\s+model.*?epoch\s*(\d+)", re.IGNORECASE), "Best model"),
        (re.compile(r"saved\s+new\s+best\s+model.*?epoch\s*(\d+)", re.IGNORECASE), "Best model"),
    ]

    for pattern, label in marker_patterns:
        match = pattern.search(log_content)
        if match:
            marker_epoch = int(match.group(1))
            marker_label = label
            break

    if marker_epoch is None and len(epochs) > 0 and len(val_rmses) == len(epochs):
        best_idx = int(np.argmin(val_rmses))
        marker_epoch = epochs[best_idx]
        marker_label = "Best Val RMSE"

    return epochs, train_losses, val_rmses, marker_epoch, marker_label


def pick_recent_log_with_metrics(base_dir: Path) -> Tuple[Optional[Path], List[int], List[float], List[float], Optional[int], Optional[str]]:
    logs = sorted(base_dir.glob("gnn_training_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None, [], [], [], None, None

    for log_file in logs:
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        epochs, train_losses, val_rmses, marker_epoch, marker_label = parse_epoch_metrics_from_log(content)
        if len(epochs) > 0:
            return log_file, epochs, train_losses, val_rmses, marker_epoch, marker_label

    newest = logs[0]
    return newest, [], [], [], None, None


def plot_gnn_convergence(repo_root: Path, out_dir: Path) -> None:
    log_file, epochs, train_losses, val_rmses, marker_epoch, marker_label = pick_recent_log_with_metrics(repo_root)

    plt.figure(figsize=(9.0, 6.0))
    ax1 = plt.gca()

    if log_file is None:
        ax1.text(0.5, 0.5, "No gnn_training_log_*.txt file found", ha="center", va="center", fontsize=12)
        ax1.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "gnn_convergence_curve.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        return

    if len(epochs) == 0:
        ax1.text(
            0.5,
            0.5,
            f"No parsable epoch metrics found in\n{log_file.name}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax1.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "gnn_convergence_curve.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        return

    line1 = ax1.plot(epochs, train_losses, color="tab:blue", marker="o", linewidth=2.0, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    line2 = ax2.plot(epochs, val_rmses, color="tab:orange", marker="s", linewidth=2.0, label="Val RMSE")
    ax2.set_ylabel("Val RMSE", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    if marker_epoch is not None:
        ax1.axvline(marker_epoch, color="tab:red", linestyle="--", linewidth=1.5)
        ax1.text(
            marker_epoch,
            max(train_losses) if len(train_losses) else 0,
            f" {marker_label}: epoch {marker_epoch}",
            color="tab:red",
            fontsize=9,
            va="top",
        )

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title(f"GNN Convergence Curve ({log_file.name})")
    plt.tight_layout()
    plt.savefig(out_dir / "gnn_convergence_curve.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    viz_root = Path(__file__).resolve().parents[1]
    out_dir = ensure_out_dir(viz_root)

    sns.set_theme(style="whitegrid", context="notebook")

    df = load_results(repo_root)

    plot_density_scatter(df, out_dir)
    plot_clinical_roc(df, out_dir)
    plot_allele_stratification(df, out_dir, repo_root)
    plot_peptide_length_error(df, out_dir)
    plot_gnn_convergence(repo_root, out_dir)

    print("Saved figures:")
    print(f" - {out_dir / 'density_scatter_ensemble.png'}")
    print(f" - {out_dir / 'clinical_roc_curve.png'}")
    print(f" - {out_dir / 'allele_stratification_barplot.png'}")
    print(f" - {out_dir / 'peptide_length_error_boxplot.png'}")
    print(f" - {out_dir / 'gnn_convergence_curve.png'}")


if __name__ == "__main__":
    main()
