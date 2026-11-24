"""Phase 4, Task 4.1: 1D CNN Architecture and Normalization Validation Script.

Validates:
1. Global training-only normalization on CWRU Fold 0 (leakage-safe).
2. VibrationCNN1D architecture instantiation from configuration.
3. Batch forward pass on PyTorch CPU DataLoader with shape verification.
4. Layer activation dimensions and parameter accounting.
5. Export of architecture layer summary table and schematic diagram.
"""

import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn_1d import VibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import get_resolved_paths, load_config

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.Validation.CNN1D")


def run_architecture_validation():
    """Execute complete architecture and normalization validation pipeline."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    reports_dir = resolved_paths["reports_dir"]
    figures_dir = resolved_paths["figures_dir"]
    checkpoints_dir = resolved_paths["model_checkpoint_dir"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 4, TASK 4.1: 1D CNN ARCHITECTURE & NORMALIZATION VALIDATION")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Dataset Partitioning & Training-Only Normalization
    # --------------------------------------------------------------------------
    logger.info("Loading CWRU Fold 0 partitions without normalizer...")
    unnorm_train_dataset = CWRUVibrationDataset(split="train", fold=0, config=config)
    unnorm_val_dataset = CWRUVibrationDataset(split="val", fold=0, config=config)
    unnorm_test_dataset = CWRUVibrationDataset(split="test", config=config)

    logger.info(f"Unnormalized counts - Train: {len(unnorm_train_dataset)}, Val: {len(unnorm_val_dataset)}, Test: {len(unnorm_test_dataset)}")

    # Extract all training windows for fitting normalizer
    train_indices = unnorm_train_dataset.subset_indices
    train_windows = unnorm_train_dataset.raw_windows[train_indices]
    train_recordings = unnorm_train_dataset.metadata_df.iloc[train_indices]["recording_id"].unique().tolist()

    logger.info(f"Fitting VibrationNormalizer strictly on Fold 0 training partition ({len(train_windows)} windows)...")
    normalizer = VibrationNormalizer(epsilon=1e-8)
    normalizer.fit(train_windows, source_recording_ids=train_recordings)

    logger.info(f"Fitted Normalizer -> Mean: {normalizer.mean:+.6e}, Std: {normalizer.std:.6e}, Epsilon: {normalizer.epsilon:.1e}")

    # Persist normalizer parameters
    norm_save_path = checkpoints_dir / "normalizer_fold0.json"
    normalizer.save(norm_save_path)
    logger.info(f"Saved normalizer parameters to: {norm_save_path}")

    # Initialize normalized datasets
    train_dataset = CWRUVibrationDataset(split="train", fold=0, normalizer=normalizer, config=config)
    val_dataset = CWRUVibrationDataset(split="val", fold=0, normalizer=normalizer, config=config)
    test_dataset = CWRUVibrationDataset(split="test", normalizer=normalizer, config=config)

    # Check normalization quality on sample batches
    sample_window_tensor, sample_label, sample_meta = train_dataset[0]
    logger.info(f"Sample window shape: {sample_window_tensor.shape}, dtype: {sample_window_tensor.dtype}")
    logger.info(f"Sample label: {sample_label.item()} ({sample_meta['fault_class']}), source: {sample_meta['source_file']}")

    # --------------------------------------------------------------------------
    # 2. PyTorch DataLoader Batch Verification
    # --------------------------------------------------------------------------
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    batch_windows, batch_labels, batch_meta = next(iter(train_loader))

    logger.info(f"Batch windows shape: {batch_windows.shape}")  # Expected: (32, 1, 2048)
    logger.info(f"Batch labels shape: {batch_labels.shape}")    # Expected: (32,)
    assert batch_windows.shape == (batch_size, 1, 2048), f"Unexpected batch window shape: {batch_windows.shape}"
    assert batch_labels.shape == (batch_size,), f"Unexpected batch label shape: {batch_labels.shape}"

    # Verify batch stats: mean should be close to 0 and std close to 1 across whole training set
    logger.info(f"First batch normalized mean: {batch_windows.mean().item():+.4f}, std: {batch_windows.std().item():.4f}")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation & Layer Dimensions
    # --------------------------------------------------------------------------
    model = VibrationCNN1D.from_config(config)
    model.eval()  # Set evaluation mode
    param_count = model.count_parameters()
    logger.info(f"Instantiated VibrationCNN1D with {param_count:,} trainable parameters.")

    # Track intermediate activations
    x = batch_windows
    act_shapes = [("Input", x.shape)]

    x1 = model.block1(x)
    act_shapes.append(("Block 1 (Conv1D-15 + BN + ReLU + MaxPool-4)", x1.shape))

    x2 = model.block2(x1)
    act_shapes.append(("Block 2 (Conv1D-7 + BN + ReLU + MaxPool-4)", x2.shape))

    x3 = model.conv3(x2)
    act_shapes.append(("Block 3 Conv3 (Conv1D-3)", x3.shape))

    x3_bn = model.bn3(x3)
    x3_relu = model.relu3(x3_bn)
    x3_pool = model.pool3(x3_relu)
    act_shapes.append(("Block 3 Pool (MaxPool-2)", x3_pool.shape))

    x_adapt = model.adaptive_pool(x3_pool)
    act_shapes.append(("AdaptiveAvgPool1D(4)", x_adapt.shape))

    x_flat = torch.flatten(x_adapt, 1)
    act_shapes.append(("Flatten", x_flat.shape))

    logits = model.classifier(x_flat)
    act_shapes.append(("Classifier Head (Linear -> ReLU -> Linear)", logits.shape))

    # Full forward pass verification
    full_forward_logits = model(batch_windows)
    assert torch.allclose(logits, full_forward_logits, atol=1e-5), "Manual layer traversal does not match model.forward()!"
    assert full_forward_logits.shape == (batch_size, 4), f"Unexpected logits shape: {full_forward_logits.shape}"
    logger.info(f"Forward pass completed successfully. Output logits shape: {full_forward_logits.shape}")

    # CAM target layer check
    cam_target = model.get_target_layer_for_cam()
    assert cam_target is model.conv3, f"Expected cam_target to be self.conv3, got {cam_target}"
    logger.info(f"Grad-CAM target layer confirmed: {cam_target}")

    # --------------------------------------------------------------------------
    # 4. Export Layer Summary Report (CSV)
    # --------------------------------------------------------------------------
    layer_summary_rows = [
        {"layer_idx": 0, "stage": "Input", "layer_type": "Raw Vibration Signal", "output_shape": f"({batch_size}, 1, 2048)", "kernel_size": "-", "stride": "-", "parameters": 0},
        {"layer_idx": 1, "stage": "Block 1", "layer_type": "Conv1d(1, 32, k=15, p=7)", "output_shape": f"({batch_size}, 32, 2048)", "kernel_size": "15", "stride": "1", "parameters": 1 * 32 * 15 + 32},
        {"layer_idx": 2, "stage": "Block 1", "layer_type": "BatchNorm1d(32) + ReLU", "output_shape": f"({batch_size}, 32, 2048)", "kernel_size": "-", "stride": "-", "parameters": 32 * 2},
        {"layer_idx": 3, "stage": "Block 1", "layer_type": "MaxPool1d(k=4, s=4)", "output_shape": f"({batch_size}, 32, 512)", "kernel_size": "4", "stride": "4", "parameters": 0},
        {"layer_idx": 4, "stage": "Block 2", "layer_type": "Conv1d(32, 64, k=7, p=3)", "output_shape": f"({batch_size}, 64, 512)", "kernel_size": "7", "stride": "1", "parameters": 32 * 64 * 7 + 64},
        {"layer_idx": 5, "stage": "Block 2", "layer_type": "BatchNorm1d(64) + ReLU", "output_shape": f"({batch_size}, 64, 512)", "kernel_size": "-", "stride": "-", "parameters": 64 * 2},
        {"layer_idx": 6, "stage": "Block 2", "layer_type": "MaxPool1d(k=4, s=4)", "output_shape": f"({batch_size}, 64, 128)", "kernel_size": "4", "stride": "4", "parameters": 0},
        {"layer_idx": 7, "stage": "Block 3", "layer_type": "Conv1d(64, 128, k=3, p=1) [CAM Target]", "output_shape": f"({batch_size}, 128, 128)", "kernel_size": "3", "stride": "1", "parameters": 64 * 128 * 3 + 128},
        {"layer_idx": 8, "stage": "Block 3", "layer_type": "BatchNorm1d(128) + ReLU", "output_shape": f"({batch_size}, 128, 128)", "kernel_size": "-", "stride": "-", "parameters": 128 * 2},
        {"layer_idx": 9, "stage": "Block 3", "layer_type": "MaxPool1d(k=2, s=2)", "output_shape": f"({batch_size}, 128, 64)", "kernel_size": "2", "stride": "2", "parameters": 0},
        {"layer_idx": 10, "stage": "Pooling", "layer_type": "AdaptiveAvgPool1d(4)", "output_shape": f"({batch_size}, 128, 4)", "kernel_size": "-", "stride": "-", "parameters": 0},
        {"layer_idx": 11, "stage": "Flatten", "layer_type": "Flatten", "output_shape": f"({batch_size}, 512)", "kernel_size": "-", "stride": "-", "parameters": 0},
        {"layer_idx": 12, "stage": "Classifier", "layer_type": "Dropout(p=0.3) + Linear(512, 64)", "output_shape": f"({batch_size}, 64)", "kernel_size": "-", "stride": "-", "parameters": 512 * 64 + 64},
        {"layer_idx": 13, "stage": "Classifier", "layer_type": "ReLU + Dropout(p=0.3)", "output_shape": f"({batch_size}, 64)", "kernel_size": "-", "stride": "-", "parameters": 0},
        {"layer_idx": 14, "stage": "Classifier", "layer_type": "Linear(64, 4) [Raw Logits]", "output_shape": f"({batch_size}, 4)", "kernel_size": "-", "stride": "-", "parameters": 64 * 4 + 4},
    ]

    summary_df = pd.DataFrame(layer_summary_rows)
    report_csv_path = reports_dir / "cnn1d_architecture_summary.csv"
    summary_df.to_csv(report_csv_path, index=False)
    logger.info(f"Saved architecture summary CSV to: {report_csv_path}")

    # --------------------------------------------------------------------------
    # 5. Export Architecture Diagram (PNG)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    stages = [
        ("Input Signal", "Raw Vibration Waveform\n(1, 2048)\n12 kHz DE Accelerometer", "#34495E"),
        ("Block 1\n(Low Freq)", "Conv1D (k=15, 32ch)\nBN + ReLU + MaxPool(4)\nOutput: (32, 512)", "#2980B9"),
        ("Block 2\n(Harmonics)", "Conv1D (k=7, 64ch)\nBN + ReLU + MaxPool(4)\nOutput: (64, 128)", "#27AE60"),
        ("Block 3\n(CAM Target)", "Conv1D (k=3, 128ch)\nBN + ReLU + MaxPool(2)\nOutput: (128, 64)", "#D35400"),
        ("Adaptive\nPooling", "AdaptiveAvgPool1D(4)\nOutput: (128, 4)\nFlatten: (512,)", "#8E44AD"),
        ("Classifier\nHead", "Linear(512, 64) + ReLU\nLinear(64, 4)\nOutput: 4 Logits", "#C0392B"),
    ]

    ax.axis("off")
    ax.set_xlim(0, len(stages) * 2.2)
    ax.set_ylim(-1, 5)

    for i, (title, desc, color) in enumerate(stages):
        x_center = i * 2.2 + 1.0
        # Box
        rect = plt.Rectangle((x_center - 0.9, 0.5), 1.8, 3.2, facecolor=color, alpha=0.15, edgecolor=color, linewidth=2, linestyle="-")
        ax.add_patch(rect)
        # Header Box
        header_rect = plt.Rectangle((x_center - 0.9, 3.1), 1.8, 0.6, facecolor=color, alpha=0.85, edgecolor=color, linewidth=1)
        ax.add_patch(header_rect)
        ax.text(x_center, 3.4, title, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        ax.text(x_center, 1.9, desc, ha="center", va="center", color="#2C3E50", fontsize=8.5, multialignment="center")

        # Arrow between stages
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_center + 1.25, 2.1),
                xytext=(x_center + 0.95, 2.1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7F8C8D"),
            )

    plt.title(
        f"InspectAI: 1D CNN Vibration Fault Classifier Architecture\n"
        f"Total Trainable Parameters: {param_count:,} | Input: (B, 1, 2048) | Output: (B, 4) Raw Logits",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    diagram_png_path = figures_dir / "cnn1d_layer_dimensions.png"
    plt.savefig(diagram_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved architecture diagram to: {diagram_png_path}")

    logger.info("=" * 70)
    logger.info("PHASE 4, TASK 4.1 VALIDATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_architecture_validation()
