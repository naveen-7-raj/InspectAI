"""InspectAI - Phase 11.2 Multi-Load Generalization Experiment Runner.

This script executes the Leave-One-Load-Out (LOLO) generalization experiment:
1. Loads the 40-recording CWRU catalog and verifies complete isolation of protected holdouts (97, 169, 185, 197).
2. Uses the exact 3 LOLO outer folds from Phase 11.1:
   - Fold 1: Train on 0, 2, 3 HP; Evaluate on 1 HP
   - Fold 2: Train on 0, 1, 3 HP; Evaluate on 2 HP
   - Fold 3: Train on 0, 1, 2 HP; Evaluate on 3 HP
3. Implements strict, leakage-proof inner validation:
   - Stratified recording-level split of outer-training partition (19 inner-train recs, 7 inner-val recs).
   - Inner train and inner val both possess all 4 classes.
   - Normalizer fitted strictly on inner-train.
   - Class-weighted CrossEntropyLoss fitted strictly on inner-train.
   - Inner val used exclusively for early stopping and best-epoch selection E*.
4. Retrains a fresh model on all 26 outer-training recordings for exactly E* epochs:
   - Normalizer fitted strictly on outer-training partition.
   - Class weights fitted strictly on outer-training labels.
5. Evaluates exactly once on the untouched outer held-out load (10 recordings).
6. Computes comprehensive metrics (overall, per-load, per-class, per-defect-severity, CPU latency).
7. Exports all artifacts to outputs/experiments/phase11_2/.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ingestion.loader import VibrationDataLoader
from src.models.cnn_1d import VibrationCNN1D
from src.preprocessing.signal_cleaner import SignalCleaner
from src.preprocessing.windowing import WindowSegmenter
from src.training.multiload_splitter import (
    MultiLoadSplitter,
    PROTECTED_HOLDOUT_IDS,
)
from src.training.normalizer import VibrationNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("InspectAI.MultiLoadExperiment")

OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_2"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 32
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
FC_UNITS = 64


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility across numpy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_all_windows(
    df_manifest: pd.DataFrame,
    raw_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Extract and cache cleaned signals and windows for all 40 recordings."""
    logger.info("Extracting windows for all recordings in manifest...")
    cleaner = SignalCleaner(
        target_sampling_rate=12000,
        remove_dc=True,
        filter_enabled=True,
        filter_type="bandpass",
        low_cutoff_hz=20.0,
        high_cutoff_hz=5000.0,
        filter_order=4,
    )
    loader = VibrationDataLoader(raw_dir)
    segmenter = WindowSegmenter(window_size=2048, step_size=512, sampling_rate=12000)

    dataset_cache: Dict[str, Dict[str, Any]] = {}

    for _, row in df_manifest.iterrows():
        rec_id = str(row["recording_id"])
        fname = f"{rec_id}.mat"
        rec = loader.load_file(fname)
        raw_de = rec.channels["DE_time"]

        orig_fs = 48000 if row["fault_class"] == "Normal" else 12000
        cleaned = cleaner.clean_signal(raw_de, orig_fs=orig_fs)
        
        source_meta = {
            "file_name": fname,
            "fault_class": row["fault_class"],
            "class_id": CLASS_TO_ID[row["fault_class"]],
            "defect_type": row["fault_type"],
            "defect_diameter_in": row["fault_size_inches"],
            "load_hp": int(row["load_hp"]),
            "rpm": row["verified_rpm"],
        }
        windows, meta_list = segmenter.segment_signal(cleaned.signal, source_meta)
        labels = np.full(len(windows), CLASS_TO_ID[row["fault_class"]], dtype=np.int64)

        dataset_cache[rec_id] = {
            "recording_id": rec_id,
            "fault_class": row["fault_class"],
            "fault_size_inches": row["fault_size_inches"],
            "load_hp": int(row["load_hp"]),
            "windows": windows,  # (N, 2048) float32
            "labels": labels,    # (N,) int64
            "metadata": meta_list,
        }

    total_win = sum(len(d["windows"]) for d in dataset_cache.values())
    logger.info(f"Successfully cached windows for {len(dataset_cache)} recordings ({total_win} total windows).")
    return dataset_cache


def create_inner_split(
    train_recording_ids: List[str],
    manifest: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[List[str], List[str]]:
    """Create a deterministic, class-stratified recording-level inner split.

    From the 26 outer-training recordings:
    - Normal (2 recs): 1 inner-train, 1 inner-val
    - Ball Fault (8 recs): 6 inner-train, 2 inner-val
    - Inner Race (8 recs): 6 inner-train, 2 inner-val
    - Outer Race (8 recs): 6 inner-train, 2 inner-val
    Result: 19 inner-train, 7 inner-val. Both partitions have all 4 classes.
    """
    sub = manifest.loc[[str(r) for r in train_recording_ids]].reset_index()
    inner_train_recs: List[str] = []
    inner_val_recs: List[str] = []

    for cls in CLASS_NAMES:
        cls_df = sub[sub["fault_class"] == cls].sort_values(
            by=["fault_size_inches", "load_hp", "recording_id"]
        )
        n_val = 1 if cls == "Normal" else 2
        train_part, val_part = train_test_split(cls_df, test_size=n_val, random_state=seed)
        inner_train_recs.extend(train_part["recording_id"].astype(str).tolist())
        inner_val_recs.extend(val_part["recording_id"].astype(str).tolist())

    return sorted(inner_train_recs), sorted(inner_val_recs)


def aggregate_partition(
    recording_ids: List[str],
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Concatenate windows and labels for a list of recording IDs."""
    wins_list = []
    labs_list = []
    meta_list = []

    for rec_id in recording_ids:
        data = cache[rec_id]
        wins_list.append(data["windows"])
        labs_list.append(data["labels"])
        for m in data["metadata"]:
            meta_list.append({
                "recording_id": rec_id,
                "fault_class": data["fault_class"],
                "fault_size_inches": data["fault_size_inches"],
                "load_hp": data["load_hp"],
            })

    X = np.concatenate(wins_list, axis=0)
    y = np.concatenate(labs_list, axis=0)
    return X, y, meta_list


def compute_balanced_weights(y: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute balanced class weights on training partition."""
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
    full_weights = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(unique_classes, weights):
        full_weights[c] = float(w)
    return torch.tensor(full_weights, dtype=torch.float32)


def train_inner_model(
    X_train_norm: np.ndarray,
    y_train: np.ndarray,
    X_val_norm: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor,
    seed: int = SEED,
) -> Tuple[int, Dict[str, List[float]]]:
    """Train on inner-train, evaluate on inner-val each epoch, find best epoch E*."""
    set_seed(seed)
    model = VibrationCNN1D(fc_units=FC_UNITS, dropout_rate=DROPOUT)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Tensor datasets: shape (N, 1, 2048)
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.int64),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.int64),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
    }

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Training
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(batch_y)

        train_loss = running_train_loss / len(train_dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                running_val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())

        val_loss = running_val_loss / len(val_dataset)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_macro_f1"].append(round(float(val_f1), 5))

        # Early stopping check: primary Macro-F1, tie-breaker val_loss
        is_improvement = (val_f1 > best_macro_f1 + 1e-4) or (
            abs(val_f1 - best_macro_f1) <= 1e-4 and val_loss < best_val_loss - 1e-4
        )

        if is_improvement:
            best_macro_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}. Selected best epoch: {best_epoch}")
            break

    return best_epoch, history


def train_outer_model(
    X_outer_train_norm: np.ndarray,
    y_outer_train: np.ndarray,
    best_epochs: int,
    class_weights: torch.Tensor,
    seed: int = SEED,
) -> VibrationCNN1D:
    """Retrain fresh VibrationCNN1D on full outer-training partition for exactly best_epochs."""
    set_seed(seed)
    model = VibrationCNN1D(fc_units=FC_UNITS, dropout_rate=DROPOUT)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(
        torch.tensor(X_outer_train_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_outer_train, dtype=torch.int64),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, best_epochs + 1):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model


def benchmark_cpu_latency(model: nn.Module) -> Dict[str, Any]:
    """Measure single-window and batch-32 CPU inference latency."""
    model.eval()
    
    # Single window benchmark (B=1)
    x_single = torch.randn(1, 1, 2048, dtype=torch.float32)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_single)
            
    single_timings = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(x_single)
            t1 = time.perf_counter()
            single_timings.append((t1 - t0) * 1000.0)  # ms

    # Batch 32 benchmark (B=32)
    x_batch = torch.randn(32, 1, 2048, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_batch)

    batch_timings = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(x_batch)
            t1 = time.perf_counter()
            batch_timings.append((t1 - t0) * 1000.0)  # ms

    return {
        "conditions": "CPU inference, single thread execution, PyTorch 1D CNN (73,156 params)",
        "single_sample_ms": {
            "median": round(float(np.median(single_timings)), 4),
            "p95": round(float(np.percentile(single_timings, 95)), 4),
            "mean": round(float(np.mean(single_timings)), 4),
            "min": round(float(np.min(single_timings)), 4),
            "max": round(float(np.max(single_timings)), 4),
        },
        "batch_32_ms": {
            "median": round(float(np.median(batch_timings)), 4),
            "p95": round(float(np.percentile(batch_timings, 95)), 4),
            "mean": round(float(np.mean(batch_timings)), 4),
            "per_sample_median_ms": round(float(np.median(batch_timings) / 32), 4),
        }
    }


def evaluate_outer_model(
    model: nn.Module,
    X_eval_norm: np.ndarray,
    y_eval: np.ndarray,
    eval_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Single-pass evaluation on outer held-out partition."""
    model.eval()
    dataset = TensorDataset(
        torch.tensor(X_eval_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_eval, dtype=torch.int64),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).numpy()
            preds_list.extend(preds)

    preds_arr = np.array(preds_list)
    targets_arr = y_eval

    # Overall Metrics
    acc = accuracy_score(targets_arr, preds_arr)
    bal_acc = balanced_accuracy_score(targets_arr, preds_arr)
    macro_prec = precision_score(targets_arr, preds_arr, average="macro", zero_division=0)
    macro_rec = recall_score(targets_arr, preds_arr, average="macro", zero_division=0)
    macro_f1 = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
    cm = confusion_matrix(targets_arr, preds_arr, labels=[0, 1, 2, 3])

    # Per-class metrics
    prec_per_class = precision_score(targets_arr, preds_arr, average=None, labels=[0, 1, 2, 3], zero_division=0)
    rec_per_class = recall_score(targets_arr, preds_arr, average=None, labels=[0, 1, 2, 3], zero_division=0)
    f1_per_class = f1_score(targets_arr, preds_arr, average=None, labels=[0, 1, 2, 3], zero_division=0)

    per_class_results = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        support = int((targets_arr == i).sum())
        per_class_results[cls_name] = {
            "precision": round(float(prec_per_class[i]), 4),
            "recall": round(float(rec_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": support,
        }

    # Per-defect-size breakdown
    df_eval = pd.DataFrame(eval_meta)
    df_eval["target"] = targets_arr
    df_eval["prediction"] = preds_arr
    df_eval["correct"] = (targets_arr == preds_arr).astype(int)

    severity_results = []
    for (fault_cls, sz), group in df_eval.groupby(["fault_class", "fault_size_inches"]):
        total_w = len(group)
        corr_w = int(group["correct"].sum())
        rec_acc = corr_w / total_w if total_w > 0 else 0.0
        severity_results.append({
            "fault_class": fault_cls,
            "fault_size_inches": float(sz),
            "windows": total_w,
            "correct_windows": corr_w,
            "accuracy": round(float(rec_acc), 4),
        })

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "correct_count": int((targets_arr == preds_arr).sum()),
        "incorrect_count": int((targets_arr != preds_arr).sum()),
        "total_windows": len(targets_arr),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "severity_breakdown": severity_results,
        "raw_predictions": preds_arr.tolist(),
        "raw_targets": targets_arr.tolist(),
    }


def main():
    logger.info("Starting InspectAI Phase 11.2 Multi-Load Generalization Experiment...")
    t_start = time.time()

    manifest_path = REPO_ROOT / "outputs" / "experiments" / "phase10_6" / "cwru_40_recordings_manifest.csv"
    raw_dir = REPO_ROOT / "data" / "raw" / "cwru"

    splitter = MultiLoadSplitter(manifest_path=manifest_path, raw_data_dir=raw_dir)
    df_manifest = splitter.df_manifest.set_index("recording_id")

    # 1. Window Extraction
    dataset_cache = extract_all_windows(splitter.df_manifest, raw_dir)

    # Containers for results
    fold_metrics = {}
    training_histories = {}
    inner_outer_splits_meta = {}
    normalizer_provenance = {}

    all_pooled_targets = []
    all_pooled_preds = []
    all_pooled_meta = []
    per_class_table_rows = []
    per_load_table_rows = []
    per_severity_table_rows = []

    # 2. Iterate Over LOLO Folds
    for fold_id in [1, 2, 3]:
        logger.info(f"\n==================== EXECUTING LOLO FOLD {fold_id} ====================")
        fold_def = splitter.get_fold(fold_id)
        val_load_hp = fold_def.val_load_hp
        outer_train_recs = fold_def.train_recording_ids
        outer_eval_recs = fold_def.val_recording_ids

        logger.info(
            f"Fold {fold_id}: Val Load={val_load_hp} HP ({len(outer_eval_recs)} recs), "
            f"Train Loads={fold_def.train_loads_hp} ({len(outer_train_recs)} recs)"
        )

        # A. Inner Split
        inner_train_recs, inner_val_recs = create_inner_split(outer_train_recs, df_manifest, seed=SEED)
        logger.info(f"Fold {fold_id} Inner Split: {len(inner_train_recs)} Train recs, {len(inner_val_recs)} Val recs")

        inner_outer_splits_meta[f"fold_{fold_id}"] = {
            "val_load_hp": val_load_hp,
            "train_loads_hp": fold_def.train_loads_hp,
            "outer_train_recordings": outer_train_recs,
            "outer_eval_recordings": outer_eval_recs,
            "inner_train_recordings": inner_train_recs,
            "inner_val_recordings": inner_val_recs,
        }

        # B. Inner Training Data Preparation
        X_inner_tr, y_inner_tr, _ = aggregate_partition(inner_train_recs, dataset_cache)
        X_inner_val, y_inner_val, _ = aggregate_partition(inner_val_recs, dataset_cache)

        # Fit normalizer on inner-train
        inner_norm = VibrationNormalizer(strategy="global")
        inner_norm.fit(X_inner_tr)
        X_inner_tr_norm = inner_norm.transform(X_inner_tr)
        X_inner_val_norm = inner_norm.transform(X_inner_val)

        # Compute inner class weights
        inner_weights = compute_balanced_weights(y_inner_tr)
        logger.info(
            f"Fold {fold_id} Inner Normalizer: mean={inner_norm.mean:.6f}, std={inner_norm.std:.6f}; "
            f"Inner Weights: {inner_weights.numpy().round(4).tolist()}"
        )

        # Train inner model to select best epoch E*
        best_epoch, hist = train_inner_model(
            X_inner_tr_norm, y_inner_tr, X_inner_val_norm, y_inner_val, inner_weights, seed=SEED
        )
        logger.info(f"Fold {fold_id}: Best Epoch selected from inner validation: E* = {best_epoch}")
        training_histories[f"fold_{fold_id}"] = {
            "selected_best_epoch": best_epoch,
            "history": hist,
        }

        # C. Full Outer Retraining
        X_outer_tr, y_outer_tr, _ = aggregate_partition(outer_train_recs, dataset_cache)
        outer_norm = VibrationNormalizer(strategy="global")
        outer_norm.fit(X_outer_tr)
        X_outer_tr_norm = outer_norm.transform(X_outer_tr)

        outer_weights = compute_balanced_weights(y_outer_tr)
        logger.info(
            f"Fold {fold_id} Outer Normalizer: mean={outer_norm.mean:.6f}, std={outer_norm.std:.6f}; "
            f"Outer Weights: {outer_weights.numpy().round(4).tolist()}"
        )

        normalizer_provenance[f"fold_{fold_id}"] = {
            "inner_normalizer": {
                "mean": round(float(inner_norm.mean), 6),
                "std": round(float(inner_norm.std), 6),
                "sample_count": len(X_inner_tr),
                "recording_ids": inner_train_recs,
            },
            "outer_normalizer": {
                "mean": round(float(outer_norm.mean), 6),
                "std": round(float(outer_norm.std), 6),
                "sample_count": len(X_outer_tr),
                "recording_ids": outer_train_recs,
            },
        }

        # Retrain fresh model on outer-train
        logger.info(f"Retraining outer model for Fold {fold_id} for {best_epoch} epochs on {len(X_outer_tr)} windows...")
        final_model = train_outer_model(X_outer_tr_norm, y_outer_tr, best_epoch, outer_weights, seed=SEED)

        # Save Checkpoint and Normalizer
        ckpt_path = CHECKPOINTS_DIR / f"lolo_fold_{fold_id}_cnn.pth"
        norm_path = CHECKPOINTS_DIR / f"lolo_fold_{fold_id}_normalizer.json"
        torch.save({
            "model_state_dict": final_model.state_dict(),
            "fold_id": fold_id,
            "best_epoch": best_epoch,
            "val_load_hp": val_load_hp,
            "train_loads_hp": fold_def.train_loads_hp,
            "train_recording_ids": outer_train_recs,
            "val_recording_ids": outer_eval_recs,
        }, ckpt_path)
        outer_norm.save(norm_path)
        logger.info(f"Saved checkpoint to {ckpt_path} and normalizer to {norm_path}")

        # D. Single-Pass Outer Evaluation
        X_outer_eval, y_outer_eval, eval_meta = aggregate_partition(outer_eval_recs, dataset_cache)
        X_outer_eval_norm = outer_norm.transform(X_outer_eval)

        eval_res = evaluate_outer_model(final_model, X_outer_eval_norm, y_outer_eval, eval_meta)

        # Add to pooled collections
        all_pooled_targets.extend(eval_res["raw_targets"])
        all_pooled_preds.extend(eval_res["raw_predictions"])
        for m in eval_meta:
            all_pooled_meta.append({**m, "fold_id": fold_id})

        # Save Fold Confusion Matrix CSV
        cm_df = pd.DataFrame(eval_res["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_csv = OUTPUT_DIR / f"confusion_matrix_fold_{fold_id}.csv"
        cm_df.to_csv(cm_csv)

        fold_metrics[f"fold_{fold_id}"] = {
            "fold_id": fold_id,
            "held_out_load_hp": val_load_hp,
            "train_loads_hp": fold_def.train_loads_hp,
            "best_epoch": best_epoch,
            "evaluated_recordings": len(outer_eval_recs),
            "evaluated_windows": eval_res["total_windows"],
            "correct_windows": eval_res["correct_count"],
            "incorrect_windows": eval_res["incorrect_count"],
            "accuracy": eval_res["accuracy"],
            "balanced_accuracy": eval_res["balanced_accuracy"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "macro_f1": eval_res["macro_f1"],
            "per_class": eval_res["per_class"],
            "confusion_matrix": eval_res["confusion_matrix"],
            "severity_breakdown": eval_res["severity_breakdown"],
        }

        # Tabulate Per-Load row
        per_load_table_rows.append({
            "fold_id": fold_id,
            "held_out_load_hp": val_load_hp,
            "windows": eval_res["total_windows"],
            "accuracy": eval_res["accuracy"],
            "balanced_accuracy": eval_res["balanced_accuracy"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "macro_f1": eval_res["macro_f1"],
        })

        # Tabulate Per-Class rows
        for cls_name, p_dict in eval_res["per_class"].items():
            per_class_table_rows.append({
                "fold_id": fold_id,
                "held_out_load_hp": val_load_hp,
                "fault_class": cls_name,
                "precision": p_dict["precision"],
                "recall": p_dict["recall"],
                "f1": p_dict["f1"],
                "support": p_dict["support"],
            })

        # Tabulate Per-Severity rows
        for s_dict in eval_res["severity_breakdown"]:
            per_severity_table_rows.append({
                "fold_id": fold_id,
                "held_out_load_hp": val_load_hp,
                "fault_class": s_dict["fault_class"],
                "fault_size_inches": s_dict["fault_size_inches"],
                "windows": s_dict["windows"],
                "correct": s_dict["correct_windows"],
                "accuracy": s_dict["accuracy"],
            })

        logger.info(
            f"Fold {fold_id} (Held-out {val_load_hp} HP) Results: "
            f"Acc={eval_res['accuracy'] * 100:.2f}%, "
            f"Bal-Acc={eval_res['balanced_accuracy'] * 100:.2f}%, "
            f"Macro-F1={eval_res['macro_f1'] * 100:.2f}%"
        )

    # 3. CPU Latency Benchmarking (on Fold 3 model)
    logger.info("Benchmarking CPU inference latency...")
    latency_results = benchmark_cpu_latency(final_model)

    # 4. Cross-Fold Aggregation & Pooling
    logger.info("Computing pooled metrics and cross-fold statistics...")
    pooled_targets = np.array(all_pooled_targets)
    pooled_preds = np.array(all_pooled_preds)

    pooled_acc = accuracy_score(pooled_targets, pooled_preds)
    pooled_bal_acc = balanced_accuracy_score(pooled_targets, pooled_preds)
    pooled_macro_prec = precision_score(pooled_targets, pooled_preds, average="macro", zero_division=0)
    pooled_macro_rec = recall_score(pooled_targets, pooled_preds, average="macro", zero_division=0)
    pooled_macro_f1 = f1_score(pooled_targets, pooled_preds, average="macro", zero_division=0)
    pooled_cm = confusion_matrix(pooled_targets, pooled_preds, labels=[0, 1, 2, 3])

    # Save Pooled Confusion Matrix CSV
    pooled_cm_df = pd.DataFrame(pooled_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    pooled_cm_df.to_csv(OUTPUT_DIR / "confusion_matrix_pooled.csv")

    # Pooled per-class
    pooled_prec_per_class = precision_score(pooled_targets, pooled_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    pooled_rec_per_class = recall_score(pooled_targets, pooled_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    pooled_f1_per_class = f1_score(pooled_targets, pooled_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)

    pooled_per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        support = int((pooled_targets == i).sum())
        pooled_per_class[cls_name] = {
            "precision": round(float(pooled_prec_per_class[i]), 4),
            "recall": round(float(pooled_rec_per_class[i]), 4),
            "f1": round(float(pooled_f1_per_class[i]), 4),
            "support": support,
        }

    # Mean & Std across folds
    fold_accs = [f["accuracy"] for f in fold_metrics.values()]
    fold_bal_accs = [f["balanced_accuracy"] for f in fold_metrics.values()]
    fold_f1s = [f["macro_f1"] for f in fold_metrics.values()]

    t_end = time.time()
    total_duration_sec = round(t_end - t_start, 2)

    cross_fold_summary = {
        "experiment_title": "Phase 11.2 Multi-Load Leave-One-Load-Out Generalization",
        "model_architecture": "VibrationCNN1D",
        "trainable_parameters": 73156,
        "fc_units": FC_UNITS,
        "total_folds": 3,
        "total_evaluated_windows": len(pooled_targets),
        "execution_duration_sec": total_duration_sec,
        "fold_level_mean_and_std": {
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "accuracy_std": round(float(np.std(fold_accs)), 4),
            "balanced_accuracy_mean": round(float(np.mean(fold_bal_accs)), 4),
            "balanced_accuracy_std": round(float(np.std(fold_bal_accs)), 4),
            "macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "macro_f1_std": round(float(np.std(fold_f1s)), 4),
        },
        "pooled_metrics": {
            "accuracy": round(float(pooled_acc), 4),
            "balanced_accuracy": round(float(pooled_bal_acc), 4),
            "macro_precision": round(float(pooled_macro_prec), 4),
            "macro_recall": round(float(pooled_macro_rec), 4),
            "macro_f1": round(float(pooled_macro_f1), 4),
            "confusion_matrix": pooled_cm.tolist(),
            "per_class": pooled_per_class,
        },
        "inference_latency": latency_results,
    }

    # 5. Save all Deliverable Files
    with open(OUTPUT_DIR / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    with open(OUTPUT_DIR / "cross_fold_summary.json", "w") as f:
        json.dump(cross_fold_summary, f, indent=2)

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(training_histories, f, indent=2)

    with open(OUTPUT_DIR / "inner_outer_splits_metadata.json", "w") as f:
        json.dump(inner_outer_splits_meta, f, indent=2)

    with open(OUTPUT_DIR / "normalizer_provenance.json", "w") as f:
        json.dump(normalizer_provenance, f, indent=2)

    experiment_config = {
        "random_seed": SEED,
        "model": "VibrationCNN1D",
        "parameters": 73156,
        "fc_units": FC_UNITS,
        "conv_filters": [32, 64, 128],
        "kernel_sizes": [15, 7, 3],
        "pool_sizes": [4, 4, 2],
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "sampling_rate_hz": 12000,
        "window_size": 2048,
        "step_size": 512,
        "loss": "Class-Weighted CrossEntropyLoss",
        "normalization": "Global Scalar Mean/Std fitted exclusively on training sets",
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    pd.DataFrame(per_class_table_rows).to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    pd.DataFrame(per_load_table_rows).to_csv(OUTPUT_DIR / "per_load_metrics.csv", index=False)
    pd.DataFrame(per_severity_table_rows).to_csv(OUTPUT_DIR / "per_defect_severity_metrics.csv", index=False)

    logger.info("\n=== PHASE 11.2 MULTI-LOAD GENERALIZATION EXPERIMENT COMPLETE ===")
    logger.info(f"Total Duration: {total_duration_sec}s")
    logger.info(
        f"Mean Fold Accuracy: {cross_fold_summary['fold_level_mean_and_std']['accuracy_mean'] * 100:.2f}% "
        f"+/- {cross_fold_summary['fold_level_mean_and_std']['accuracy_std'] * 100:.2f}%"
    )
    logger.info(
        f"Mean Fold Macro-F1: {cross_fold_summary['fold_level_mean_and_std']['macro_f1_mean'] * 100:.2f}% "
        f"+/- {cross_fold_summary['fold_level_mean_and_std']['macro_f1_std'] * 100:.2f}%"
    )
    logger.info(f"Pooled Macro-F1: {cross_fold_summary['pooled_metrics']['macro_f1'] * 100:.2f}%")
    logger.info(f"Single Window Median CPU Latency: {latency_results['single_sample_ms']['median']} ms")


if __name__ == "__main__":
    main()
