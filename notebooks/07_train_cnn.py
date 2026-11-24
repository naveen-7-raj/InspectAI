"""Phase 4, Task 4.2: CNN Training and Two-Fold Group Cross-Validation Execution Script.

Executes:
1. Strict recording-level validation of Fold 0 and Fold 1 assignments.
2. Verification that held-out test recordings (97, 185, 169, 197) remain untouched.
3. Fold 0: Independent normalizer fitting, CNN initialization, training, evaluation, and checkpointing.
4. Fold 1: Independent normalizer fitting, CNN initialization, training, evaluation, and checkpointing.
5. Export of per-fold histories, classification reports, comparison CSV, and loss/F1 curves.
"""

import json
import logging
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models.cnn_1d import VibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.training.trainer import DEFAULT_CLASS_NAMES, ModelTrainer
from src.utils.config_parser import get_resolved_paths, load_config

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.TrainCV")

# Expected recording assignments for verification
EXPECTED_TEST_RECS = {"97", "185", "169", "197"}
EXPECTED_FOLD_0_TRAIN = {"100", "105", "222", "234"}
EXPECTED_FOLD_0_VAL = {"98", "99", "118", "130", "209"}
EXPECTED_FOLD_1_TRAIN = {"98", "99", "118", "130", "209"}
EXPECTED_FOLD_1_VAL = {"100", "105", "222", "234"}


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across PyTorch, NumPy, and Python."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_split_assignments(metadata_path: Path) -> None:
    """Verify that split assignments strictly match the design specification."""
    df = pd.read_csv(metadata_path)
    df["rec_str"] = df["recording_id"].astype(str)

    actual_test = set(df[df["split"] == "test"]["rec_str"].unique())
    actual_f0_train = set(df[df["fold_0"] == "train"]["rec_str"].unique())
    actual_f0_val = set(df[df["fold_0"] == "val"]["rec_str"].unique())
    actual_f1_train = set(df[df["fold_1"] == "train"]["rec_str"].unique())
    actual_f1_val = set(df[df["fold_1"] == "val"]["rec_str"].unique())

    assert actual_test == EXPECTED_TEST_RECS, f"Mismatch in Test recordings: {actual_test} vs {EXPECTED_TEST_RECS}"
    assert actual_f0_train == EXPECTED_FOLD_0_TRAIN, f"Mismatch in Fold 0 Train: {actual_f0_train} vs {EXPECTED_FOLD_0_TRAIN}"
    assert actual_f0_val == EXPECTED_FOLD_0_VAL, f"Mismatch in Fold 0 Val: {actual_f0_val} vs {EXPECTED_FOLD_0_VAL}"
    assert actual_f1_train == EXPECTED_FOLD_1_TRAIN, f"Mismatch in Fold 1 Train: {actual_f1_train} vs {EXPECTED_FOLD_1_TRAIN}"
    assert actual_f1_val == EXPECTED_FOLD_1_VAL, f"Mismatch in Fold 1 Val: {actual_f1_val} vs {EXPECTED_FOLD_1_VAL}"

    # Verify complete isolation: Test recordings must never appear in dev partitions
    dev_recs = actual_f0_train | actual_f0_val | actual_f1_train | actual_f1_val
    overlap = actual_test.intersection(dev_recs)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Test recordings found in dev pool: {overlap}"

    logger.info("Split verification successful: All recording partitions strictly validated. Zero test set leakage.")


def train_single_fold(
    fold_id: int,
    config: dict,
    checkpoints_dir: Path,
    reports_dir: Path,
) -> dict:
    """Train and evaluate a fresh CNN model for a single cross-validation fold."""
    logger.info("=" * 70)
    logger.info(f"STARTING CROSS-VALIDATION FOLD {fold_id}")
    logger.info("=" * 70)

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    max_epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("learning_rate", 0.001))
    wd = float(train_cfg.get("weight_decay", 0.0001))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 7))
    min_delta = float(train_cfg.get("early_stopping", {}).get("min_delta", 1e-4))
    device = train_cfg.get("device", "cpu")

    # Set seed before data loading & model instantiation
    set_seed(seed + fold_id)

    # 1. Unnormalized Training Partition to fit Normalizer
    unnorm_train_ds = CWRUVibrationDataset(split="train", fold=fold_id, config=config)
    train_windows = unnorm_train_ds.raw_windows[unnorm_train_ds.subset_indices]
    train_recs = unnorm_train_ds.metadata_df.iloc[unnorm_train_ds.subset_indices]["recording_id"].astype(str).unique().tolist()

    logger.info(f"Fold {fold_id}: Fitting fresh normalizer strictly on {len(train_windows)} training windows (Recordings: {train_recs})...")
    normalizer = VibrationNormalizer(epsilon=1e-8)
    normalizer.fit(train_windows, source_recording_ids=train_recs)

    # Save fold normalizer
    norm_path = checkpoints_dir / f"normalizer_fold_{fold_id}.json"
    normalizer.save(norm_path)
    logger.info(f"Fold {fold_id}: Saved normalizer to {norm_path} (Mean: {normalizer.mean:+.6e}, Std: {normalizer.std:.6e})")

    # 2. Normalized DataLoaders
    train_dataset = CWRUVibrationDataset(split="train", fold=fold_id, normalizer=normalizer, config=config)
    val_dataset = CWRUVibrationDataset(split="val", fold=fold_id, normalizer=normalizer, config=config)

    logger.info(f"Fold {fold_id}: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    # Shuffle training data only; never shuffle validation data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Fresh Model Instantiation
    set_seed(seed + fold_id * 10)  # Distinct seed for model initialization
    model = VibrationCNN1D.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoints_dir,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        class_names=DEFAULT_CLASS_NAMES,
        config=config,
    )

    checkpoint_path = checkpoints_dir / f"fold_{fold_id}_best.pth"

    start_time = time.time()
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        fold_id=fold_id,
        checkpoint_path=checkpoint_path,
        normalizer_meta=normalizer.to_dict(),
        lr_scheduler=lr_scheduler,
    )
    duration = time.time() - start_time

    logger.info(
        f"Fold {fold_id} completed in {duration:.1f}s. "
        f"Best Epoch: {fit_result['best_epoch']}, Best Val Macro-F1: {fit_result['best_macro_f1']:.4f}, "
        f"Best Val Loss: {fit_result['best_val_loss']:.4f}."
    )

    # 4. Save Training History CSV
    history_df = pd.DataFrame(fit_result["history"])
    history_csv = reports_dir / f"fold_{fold_id}_history.csv"
    history_df.to_csv(history_csv, index=False)
    logger.info(f"Saved Fold {fold_id} history to: {history_csv}")

    # 5. Save Classification Report CSV
    best_metrics = fit_result["best_metrics"]
    per_class_rows = []
    for cls_name, m in best_metrics["per_class"].items():
        per_class_rows.append({
            "fold_id": fold_id,
            "fault_class": cls_name,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "support": m["support"],
        })
    clf_df = pd.DataFrame(per_class_rows)
    clf_csv = reports_dir / f"fold_{fold_id}_classification_report.csv"
    clf_df.to_csv(clf_csv, index=False)
    logger.info(f"Saved Fold {fold_id} classification report to: {clf_csv}")

    return {
        "fold_id": fold_id,
        "duration_seconds": duration,
        "best_epoch": fit_result["best_epoch"],
        "total_epochs": fit_result["total_epochs_trained"],
        "stopped_early": fit_result["stopped_early"],
        "best_macro_f1": fit_result["best_macro_f1"],
        "best_val_loss": fit_result["best_val_loss"],
        "best_val_accuracy": best_metrics["accuracy"],
        "best_macro_precision": best_metrics["macro_precision"],
        "best_macro_recall": best_metrics["macro_recall"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "per_class": best_metrics["per_class"],
        "history": fit_result["history"],
        "checkpoint_path": str(checkpoint_path),
        "normalizer_path": str(norm_path),
    }


def plot_training_results(
    fold_0_res: dict,
    fold_1_res: dict,
    figures_dir: Path,
) -> None:
    """Generate loss curves, Macro-F1 curves, and confusion matrix heatmaps."""
    h0 = pd.DataFrame(fold_0_res["history"])
    h1 = pd.DataFrame(fold_1_res["history"])

    # 1. Loss Curves
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(h0["epoch"], h0["train_loss"], label="Fold 0 Train Loss", color="#2980b9", linestyle="--", lw=1.8)
    ax.plot(h0["epoch"], h0["val_loss"], label="Fold 0 Val Loss", color="#2980b9", lw=2.2)
    ax.plot(h1["epoch"], h1["train_loss"], label="Fold 1 Train Loss", color="#e67e22", linestyle="--", lw=1.8)
    ax.plot(h1["epoch"], h1["val_loss"], label="Fold 1 Val Loss", color="#e67e22", lw=2.2)

    ax.scatter([fold_0_res["best_epoch"]], [fold_0_res["best_val_loss"]], color="#2980b9", s=80, zorder=5, label=f"Fold 0 Best (Ep {fold_0_res['best_epoch']})")
    ax.scatter([fold_1_res["best_epoch"]], [fold_1_res["best_val_loss"]], color="#e67e22", s=80, zorder=5, label=f"Fold 1 Best (Ep {fold_1_res['best_epoch']})")

    ax.set_title("InspectAI 1D CNN: Training & Validation Loss Across Cross-Validation Folds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("CrossEntropy Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    loss_fig_path = figures_dir / "loss_curves.png"
    plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curves to: {loss_fig_path}")

    # 2. Macro-F1 Curves
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(h0["epoch"], h0["val_macro_f1"], label="Fold 0 Val Macro-F1", color="#27ae60", lw=2.2)
    ax.plot(h1["epoch"], h1["val_macro_f1"], label="Fold 1 Val Macro-F1", color="#8e44ad", lw=2.2)

    ax.scatter([fold_0_res["best_epoch"]], [fold_0_res["best_macro_f1"]], color="#27ae60", s=80, zorder=5, label=f"Fold 0 Peak F1: {fold_0_res['best_macro_f1']:.4f}")
    ax.scatter([fold_1_res["best_epoch"]], [fold_1_res["best_macro_f1"]], color="#8e44ad", s=80, zorder=5, label=f"Fold 1 Peak F1: {fold_1_res['best_macro_f1']:.4f}")

    ax.set_title("InspectAI 1D CNN: Validation Macro-F1 Performance Across Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Macro-F1 Score", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    f1_fig_path = figures_dir / "macro_f1_curves.png"
    plt.savefig(f1_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Macro-F1 curves to: {f1_fig_path}")

    # 3. Confusion Matrices
    for res, fold_name in [(fold_0_res, "Fold 0"), (fold_1_res, "Fold 1")]:
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=DEFAULT_CLASS_NAMES,
            yticklabels=DEFAULT_CLASS_NAMES,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"InspectAI: {fold_name} Validation Confusion Matrix\n(Epoch {res['best_epoch']} | Macro-F1: {res['best_macro_f1']:.4f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Fault Class", fontsize=10)
        ax.set_ylabel("True Fault Class", fontsize=10)
        plt.tight_layout()
        cm_path = figures_dir / f"{fold_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {fold_name} confusion matrix to: {cm_path}")


def main():
    """Main execution function for two-fold group cross-validation."""
    config = load_config()
    paths = get_resolved_paths(config)
    reports_dir = paths["reports_dir"]
    figures_dir = paths["figures_dir"]
    checkpoints_dir = paths["model_checkpoint_dir"]
    metadata_path = paths["processed_data_dir"] / "window_split_metadata.csv"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Verify split integrity
    verify_split_assignments(metadata_path)

    # Step 2: Run Fold 0
    fold_0_res = train_single_fold(
        fold_id=0,
        config=config,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
    )

    # Step 3: Run Fold 1
    fold_1_res = train_single_fold(
        fold_id=1,
        config=config,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
    )

    # Step 4: Generate Comparative Report
    comparison_rows = [
        {
            "fold_id": "Fold 0",
            "train_recordings": "100, 105, 222, 234",
            "val_recordings": "98, 99, 118, 130, 209",
            "train_samples": 938,
            "val_samples": 1172,
            "best_epoch": fold_0_res["best_epoch"],
            "total_epochs": fold_0_res["total_epochs"],
            "duration_seconds": round(fold_0_res["duration_seconds"], 1),
            "val_loss": round(fold_0_res["best_val_loss"], 4),
            "val_accuracy": round(fold_0_res["best_val_accuracy"], 4),
            "val_macro_precision": round(fold_0_res["best_macro_precision"], 4),
            "val_macro_recall": round(fold_0_res["best_macro_recall"], 4),
            "val_macro_f1": round(fold_0_res["best_macro_f1"], 4),
            "f1_normal": round(fold_0_res["per_class"]["Normal"]["f1_score"], 4),
            "f1_ball_fault": round(fold_0_res["per_class"]["Ball Fault"]["f1_score"], 4),
            "f1_inner_race": round(fold_0_res["per_class"]["Inner Race Fault"]["f1_score"], 4),
            "f1_outer_race": round(fold_0_res["per_class"]["Outer Race Fault"]["f1_score"], 4),
            "checkpoint": fold_0_res["checkpoint_path"],
        },
        {
            "fold_id": "Fold 1",
            "train_recordings": "98, 99, 118, 130, 209",
            "val_recordings": "100, 105, 222, 234",
            "train_samples": 1172,
            "val_samples": 938,
            "best_epoch": fold_1_res["best_epoch"],
            "total_epochs": fold_1_res["total_epochs"],
            "duration_seconds": round(fold_1_res["duration_seconds"], 1),
            "val_loss": round(fold_1_res["best_val_loss"], 4),
            "val_accuracy": round(fold_1_res["best_val_accuracy"], 4),
            "val_macro_precision": round(fold_1_res["best_macro_precision"], 4),
            "val_macro_recall": round(fold_1_res["best_macro_recall"], 4),
            "val_macro_f1": round(fold_1_res["best_macro_f1"], 4),
            "f1_normal": round(fold_1_res["per_class"]["Normal"]["f1_score"], 4),
            "f1_ball_fault": round(fold_1_res["per_class"]["Ball Fault"]["f1_score"], 4),
            "f1_inner_race": round(fold_1_res["per_class"]["Inner Race Fault"]["f1_score"], 4),
            "f1_outer_race": round(fold_1_res["per_class"]["Outer Race Fault"]["f1_score"], 4),
            "checkpoint": fold_1_res["checkpoint_path"],
        },
        {
            "fold_id": "Cross-Validation Mean",
            "train_recordings": "-",
            "val_recordings": "-",
            "train_samples": 1055,
            "val_samples": 1055,
            "best_epoch": round((fold_0_res["best_epoch"] + fold_1_res["best_epoch"]) / 2, 1),
            "total_epochs": round((fold_0_res["total_epochs"] + fold_1_res["total_epochs"]) / 2, 1),
            "duration_seconds": round(fold_0_res["duration_seconds"] + fold_1_res["duration_seconds"], 1),
            "val_loss": round((fold_0_res["best_val_loss"] + fold_1_res["best_val_loss"]) / 2, 4),
            "val_accuracy": round((fold_0_res["best_val_accuracy"] + fold_1_res["best_val_accuracy"]) / 2, 4),
            "val_macro_precision": round((fold_0_res["best_macro_precision"] + fold_1_res["best_macro_precision"]) / 2, 4),
            "val_macro_recall": round((fold_0_res["best_macro_recall"] + fold_1_res["best_macro_recall"]) / 2, 4),
            "val_macro_f1": round((fold_0_res["best_macro_f1"] + fold_1_res["best_macro_f1"]) / 2, 4),
            "f1_normal": round((fold_0_res["per_class"]["Normal"]["f1_score"] + fold_1_res["per_class"]["Normal"]["f1_score"]) / 2, 4),
            "f1_ball_fault": round((fold_0_res["per_class"]["Ball Fault"]["f1_score"] + fold_1_res["per_class"]["Ball Fault"]["f1_score"]) / 2, 4),
            "f1_inner_race": round((fold_0_res["per_class"]["Inner Race Fault"]["f1_score"] + fold_1_res["per_class"]["Inner Race Fault"]["f1_score"]) / 2, 4),
            "f1_outer_race": round((fold_0_res["per_class"]["Outer Race Fault"]["f1_score"] + fold_1_res["per_class"]["Outer Race Fault"]["f1_score"]) / 2, 4),
            "checkpoint": "-",
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = reports_dir / "cv_folds_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"Saved cross-validation comparison CSV to: {comp_csv}")

    # Step 5: Generate Visualizations
    plot_training_results(fold_0_res, fold_1_res, figures_dir)

    logger.info("=" * 70)
    logger.info("PHASE 4, TASK 4.2 CROSS-VALIDATION TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

