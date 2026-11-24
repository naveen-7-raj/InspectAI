"""InspectAI - Task 3.3: Leakage-Safe Dataset Split & PyTorch Dataset Validation.

Generates recording-level final test holdout and 2-fold group-aware development partitions,
validates zero overlap, and checks PyTorch Dataset tensor output shapes and dtypes.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.training.dataset import CWRUVibrationDataset
from src.training.dataset_splitter import DatasetSplitter
from src.utils.config_parser import load_config, get_resolved_paths


def run_split_validation() -> Dict[str, Any]:
    """Execute leakage-safe splitting, verify PyTorch Dataset, and output reports & figures."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    figures_dir = resolved_paths["figures_dir"]
    reports_dir = resolved_paths["reports_dir"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("InspectAI - Task 3.3: Leakage-Safe Dataset Splitting Validation")
    print("Partitioning Variable: Recording ID / Source File (Group-Aware)")
    print(f"Random Seed: {config['dataset_split']['random_seed']}")
    print(f"Validation Strategy: {config['dataset_split']['validation_strategy']} ({config['dataset_split']['n_validation_folds']} Folds)")
    print("=================================================================\n")

    # 1. Create and save split metadata
    splitter = DatasetSplitter(config=config)
    split_csv_path, summary = splitter.save_split_metadata()
    print(f"[Artifact Created] Split metadata saved to: {split_csv_path.resolve()}")

    split_df = pd.read_csv(split_csv_path)

    # 2. Display Partition Summary
    test_recs = summary["test_recordings"]
    dev_recs = summary["dev_recordings"]
    print(f"\nFinal Test Set ({len(test_recs)} recordings): {test_recs}")
    print(f"  -> Total Test Windows: {summary['test_windows']:,} ({summary['test_class_distribution']})")

    print(f"\nDevelopment Pool ({len(dev_recs)} recordings): {dev_recs}")
    print(f"  -> Total Dev Windows:  {summary['dev_windows']:,} ({summary['dev_class_distribution']})")

    # 3. Verify Group Isolation
    overlap = set(test_recs) & set(dev_recs)
    assert len(overlap) == 0, f"Critical leakage! Test and Dev share recordings: {overlap}"
    print("\n[Assertion PASSED] Zero recording leakage between Test and Development pool.")

    # 4. Display 2-Fold CV Distribution
    for fold, finfo in summary["folds"].items():
        print(f"\n--- Fold {fold} ---")
        print(f"  Val Recordings ({len(finfo['val_recordings'])}):   {finfo['val_recordings']}")
        print(f"  Val Classes ({len(finfo['val_classes'])}):        {finfo['val_classes']}")
        print(f"  Train Recordings ({len(finfo['train_recordings'])}): {finfo['train_recordings']}")
        print(f"  Train Classes ({len(finfo['train_classes'])}):      {finfo['train_classes']}")
        print(f"  Sample Counts: Train={finfo['train_windows']:,} | Val={finfo['val_windows']:,}")

        # Assert full class representation
        assert len(finfo["val_classes"]) == 4, f"Fold {fold} validation set missing classes!"
        assert len(finfo["train_classes"]) == 4, f"Fold {fold} training set missing classes!"

    print("\n[Assertion PASSED] 100% class coverage across all validation folds.")

    # 5. Verify PyTorch Dataset Interface
    ds_test = CWRUVibrationDataset(split="test")
    ds_train0 = CWRUVibrationDataset(split="train", fold=0)
    ds_val0 = CWRUVibrationDataset(split="val", fold=0)

    win_t, lbl_t, meta = ds_test[0]
    print(f"\nPyTorch Dataset Sample Check (Test set):")
    print(f"  Window Tensor: shape={list(win_t.shape)}, dtype={win_t.dtype}")
    print(f"  Label Tensor:  val={lbl_t.item()} ({meta['fault_class']}), dtype={lbl_t.dtype}")
    assert win_t.shape == (1, 2048), f"Expected shape (1, 2048), got {win_t.shape}"
    assert win_t.dtype == torch.float32, f"Expected float32, got {win_t.dtype}"
    assert lbl_t.dtype == torch.long, f"Expected int64/long, got {lbl_t.dtype}"

    # 6. Save split summary table to outputs/reports/
    rows = []
    for fname, group in split_df.groupby("source_file"):
        row = {
            "source_file": fname,
            "fault_class": group["fault_class"].iloc[0],
            "defect_diameter_in": group["defect_diameter_in"].iloc[0],
            "motor_load_hp": group["motor_load_hp"].iloc[0],
            "motor_rpm": group["motor_rpm"].iloc[0],
            "split": group["split"].iloc[0],
            "fold_0": group["fold_0"].iloc[0],
            "fold_1": group["fold_1"].iloc[0],
            "window_count": len(group),
        }
        rows.append(row)

    split_summary_df = pd.DataFrame(rows)
    report_csv_path = reports_dir / "dataset_split_summary.csv"
    split_summary_df.to_csv(report_csv_path, index=False)
    print(f"[Artifact Created] Split summary CSV saved to: {report_csv_path.resolve()}")

    # 7. Generate Visualizations
    fig1_path = figures_dir / "dataset_split_recording_partition.png"
    plot_recording_partition(split_summary_df, fig1_path)

    fig2_path = figures_dir / "dataset_split_class_distribution.png"
    plot_split_class_distribution(split_df, fig2_path)

    return summary


def plot_recording_partition(df: pd.DataFrame, save_path: Path) -> None:
    """Generate visual diagram showing recording assignment across Test and CV Folds."""
    fig, ax = plt.subplots(figsize=(12, 7))

    recordings = df["source_file"].tolist()
    y_pos = np.arange(len(recordings))

    # Determine partition role for each recording
    roles = []
    colors = []
    for _, row in df.iterrows():
        if row["split"] == "test":
            roles.append(f"FINAL TEST SET (Held Out: 0.014\" Defect / 0 HP Baseline)")
            colors.append("#2ca02c")
        elif row["fold_0"] == "val":
            roles.append(f"DEV POOL (Fold 0 Val / Fold 1 Train)")
            colors.append("#1f77b4")
        else:
            roles.append(f"DEV POOL (Fold 1 Val / Fold 0 Train)")
            colors.append("#ff7f0e")

    bars = ax.barh(y_pos, df["window_count"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    yticklabels = [
        f"{r['source_file']} [{r['fault_class']}] (Load: {r['motor_load_hp']} HP, Defect: {r['defect_diameter_in']}\")"
        for _, r in df.iterrows()
    ]
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Extracted Windows (75% Overlap)", fontsize=10, fontweight="semibold")
    ax.set_title(
        "InspectAI - Leakage-Safe Recording-Level Dataset Partitioning\n"
        "Strict Disjoint Partitioning by Machine Recording ID (No Overlapping Windows Shared)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Bar value annotations
    for bar, role in zip(bars, roles):
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height() / 2, f"{int(w)} wins | {role}", va="center", fontsize=8)

    ax.set_xlim(0, max(df["window_count"]) + 550)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Partition figure saved to: {save_path.resolve()}")


def plot_split_class_distribution(split_df: pd.DataFrame, save_path: Path) -> None:
    """Generate side-by-side bar chart showing balanced class distribution across partitions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    palette = {
        "Normal": "#2ca02c",
        "Ball_Fault": "#ff7f0e",
        "Inner_Race_Fault": "#d62728",
        "Outer_Race_Fault": "#9467bd",
    }

    # Left: Test Set class distribution
    test_df = split_df[split_df["split"] == "test"]
    test_counts = test_df["fault_class"].value_counts()
    classes = list(test_counts.index)
    bars1 = ax1.bar(
        classes,
        test_counts.values,
        color=[palette.get(c, "#333") for c in classes],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax1.set_title(f"Final Test Set Distribution (Total = {len(test_df):,} windows)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Window Count", fontsize=10)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{int(b.get_height())}", ha="center", fontsize=9, fontweight="bold")

    # Right: Development Pool class distribution
    dev_df = split_df[split_df["split"] == "dev"]
    dev_counts = dev_df["fault_class"].value_counts()
    dev_classes = list(dev_counts.index)
    bars2 = ax2.bar(
        dev_classes,
        dev_counts.values,
        color=[palette.get(c, "#333") for c in dev_classes],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax2.set_title(f"Development Pool Distribution (Total = {len(dev_df):,} windows)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Window Count", fontsize=10)
    ax2.set_xticks(range(len(dev_classes)))
    ax2.set_xticklabels(dev_classes, rotation=20, ha="right", fontsize=9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, f"{int(b.get_height())}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Class distribution figure saved to: {save_path.resolve()}")


if __name__ == "__main__":
    run_split_validation()

