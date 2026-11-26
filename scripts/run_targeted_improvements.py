"""InspectAI - Phase 11.4: Targeted Model Improvement & Controlled Ablation Experiments.

Implements controlled, independent ablation experiments across the 3 Leave-One-Load-Out (LOLO) folds:
- Baseline: Frozen VibrationCNN1D (73,156 params, lr=0.0005, AdamW, class-weighted CE, seed=42).
- Experiment A: Training-only signal augmentation (amplitude scaling, low-level Gaussian noise, circular time shift).
- Experiment B: Envelope-spectrum feature fusion (dual-branch SpectralFusionCNN with Hilbert envelope spectrum).
- Experiment C: Training schedule adjustment (inner-validation minimum epoch floor E_min = 5).

Guarantees:
- Zero data leakage across recording partitions.
- Protected historical holdout recordings (97, 169, 185, 197) quarantined.
- Outer evaluation data is never used for training, validation, or hyperparameter selection.
- Normalizers fitted strictly on training recordings.
- Full artifact reproducibility with deterministic random seeds.
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Prevent OpenMP conflict and ensure headless plotting
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.ingestion.loader import VibrationDataLoader
from src.models.cnn_1d import VibrationCNN1D
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.preprocessing.signal_cleaner import SignalCleaner
from src.preprocessing.windowing import WindowSegmenter
from src.training.multiload_splitter import PROTECTED_HOLDOUT_IDS
from src.training.normalizer import VibrationNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.Phase11_4")

# Directories
EXP_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_4"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "phase11_4"
SPLITS_PATH = REPO_ROOT / "outputs" / "experiments" / "phase11_1" / "multiload_splits.json"
MANIFEST_PATH = REPO_ROOT / "outputs" / "experiments" / "phase10_6" / "cwru_40_recordings_manifest.csv"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "cwru"

EXP_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 7
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


class AugmentedVibrationDataset(Dataset):
    """Vibration dataset with training-only signal augmentation.
    
    Transforms (applied strictly during training):
    1. Small amplitude scaling: s ~ Uniform(0.95, 1.05) (+/- 5%)
    2. Low-level Gaussian noise: noise ~ N(0, (0.01 * std(x))^2) (SNR ~ 40 dB)
    3. Small circular temporal shift: delta ~ Uniform(-32, 32) samples (~ +/- 2.67 ms)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        seed: int = SEED,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float32).copy()
        self.y = np.asarray(y, dtype=np.int64).copy()
        self.augment = augment
        self.seed = seed

        if self.augment:
            for idx in range(len(self.X)):
                rng = np.random.RandomState(self.seed + idx * 37)
                scale = rng.uniform(0.95, 1.05)
                self.X[idx] = self.X[idx] * scale
                std_x = np.std(self.X[idx])
                noise_sigma = 0.01 * (std_x if std_x > 1e-8 else 1.0)
                noise = rng.normal(0.0, noise_sigma, size=self.X[idx].shape).astype(np.float32)
                self.X[idx] = self.X[idx] + noise
                shift = int(rng.randint(-32, 33))
                self.X[idx] = np.roll(self.X[idx], shift)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        tensor_y = torch.tensor(self.y[idx], dtype=torch.int64)
        return tensor_x, tensor_y


class DualBranchVibrationDataset(Dataset):
    """Dual-branch dataset providing normalized waveform and envelope spectrum tensors."""

    def __init__(
        self,
        waveforms: np.ndarray,
        spectra: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.waveforms = np.asarray(waveforms, dtype=np.float32)
        self.spectra = np.asarray(spectra, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wave = torch.tensor(self.waveforms[idx], dtype=torch.float32).unsqueeze(0)
        spec = torch.tensor(self.spectra[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return wave, spec, label


def extract_all_windows(
    df_manifest: pd.DataFrame,
    raw_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Extract and cache cleaned signals, windows, and Hilbert envelope spectra for all recordings."""
    logger.info("Extracting windows and envelope spectra for all recordings in manifest...")
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

        # Compute Hilbert envelope spectrum for each window: shape (N, 1025)
        _, env_spectra, _ = compute_hilbert_envelope_spectrum(windows, sampling_rate_hz=12000)

        dataset_cache[rec_id] = {
            "recording_id": rec_id,
            "fault_class": row["fault_class"],
            "fault_size_inches": row["fault_size_inches"],
            "load_hp": int(row["load_hp"]),
            "windows": windows,         # (N, 2048) float32
            "env_spectra": env_spectra, # (N, 1025) float32
            "labels": labels,           # (N,) int64
            "metadata": meta_list,
        }

    total_win = sum(len(d["windows"]) for d in dataset_cache.values())
    logger.info(f"Cached windows for {len(dataset_cache)} recordings ({total_win} total windows).")
    return dataset_cache


def create_inner_split(
    train_recording_ids: List[str],
    manifest: pd.DataFrame,
    seed: int = SEED,
) -> Tuple[List[str], List[str]]:
    """Create a deterministic, class-stratified recording-level inner split."""
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Concatenate waveforms, envelope spectra, labels, and metadata for given recording IDs."""
    wins_list = []
    spec_list = []
    labs_list = []
    meta_list = []

    for rec_id in recording_ids:
        data = cache[rec_id]
        wins_list.append(data["windows"])
        spec_list.append(data["env_spectra"])
        labs_list.append(data["labels"])
        for m in data["metadata"]:
            meta_list.append({
                "recording_id": rec_id,
                "fault_class": data["fault_class"],
                "fault_size_inches": data["fault_size_inches"],
                "load_hp": data["load_hp"],
            })

    X_wave = np.concatenate(wins_list, axis=0)
    X_spec = np.concatenate(spec_list, axis=0)
    y = np.concatenate(labs_list, axis=0)
    return X_wave, X_spec, y, meta_list


def compute_balanced_weights(y: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute balanced class weights on training partition."""
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
    full_weights = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(unique_classes, weights):
        full_weights[c] = float(w)
    return torch.tensor(full_weights, dtype=torch.float32)


def train_inner_model(
    experiment_name: str,
    X_tr_wave: np.ndarray,
    X_tr_spec: np.ndarray,
    y_tr: np.ndarray,
    X_val_wave: np.ndarray,
    X_val_spec: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor,
    seed: int = SEED,
    min_epochs: int = 1,
) -> Tuple[int, Dict[str, List[float]]]:
    """Train on inner-train, evaluate on inner-val each epoch, find best epoch E*."""
    set_seed(seed)
    is_dual = (experiment_name == "experiment_b")
    is_augmented = (experiment_name == "experiment_a")

    if is_dual:
        model = SpectralFusionCNN(num_classes=4, input_channels=1, waveform_length=2048, spectral_bins=1025, fc_units=FC_UNITS, dropout_rate=DROPOUT)
        train_ds = DualBranchVibrationDataset(X_tr_wave, X_tr_spec, y_tr)
        val_ds = DualBranchVibrationDataset(X_val_wave, X_val_spec, y_val)
    else:
        model = VibrationCNN1D(fc_units=FC_UNITS, dropout_rate=DROPOUT)
        train_ds = AugmentedVibrationDataset(X_tr_wave, y_tr, augment=is_augmented, seed=seed)
        val_ds = AugmentedVibrationDataset(X_val_wave, y_val, augment=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_train_loss = 0.0

        if is_dual:
            for b_wave, b_spec, b_y in train_loader:
                optimizer.zero_grad()
                logits = model(b_wave, spectral_input=b_spec)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * len(b_y)
        else:
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                logits = model(b_x)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * len(b_y)

        train_loss = running_train_loss / len(train_ds)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            if is_dual:
                for b_wave, b_spec, b_y in val_loader:
                    logits = model(b_wave, spectral_input=b_spec)
                    loss = criterion(logits, b_y)
                    running_val_loss += loss.item() * len(b_y)
                    preds = torch.argmax(logits, dim=1).numpy()
                    val_preds.extend(preds)
                    val_targets.extend(b_y.numpy())
            else:
                for b_x, b_y in val_loader:
                    logits = model(b_x)
                    loss = criterion(logits, b_y)
                    running_val_loss += loss.item() * len(b_y)
                    preds = torch.argmax(logits, dim=1).numpy()
                    val_preds.extend(preds)
                    val_targets.extend(b_y.numpy())

        val_loss = running_val_loss / len(val_ds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_macro_f1"].append(round(float(val_f1), 5))

        # Check early stopping improvement
        is_improvement = (val_f1 > best_macro_f1 + 1e-4) or (
            abs(val_f1 - best_macro_f1) <= 1e-4 and val_loss < best_val_loss - 1e-4
        )

        if epoch >= min_epochs:
            if is_improvement:
                best_macro_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            # During epoch floor, track the best so far
            if is_improvement:
                best_macro_f1 = val_f1
                best_val_loss = val_loss
                best_epoch = epoch

        if epoch >= min_epochs and patience_counter >= PATIENCE:
            logger.info(f"[{experiment_name}] Early stopping at epoch {epoch}. Selected best epoch: {best_epoch}")
            break

    return best_epoch, history


def train_outer_model(
    experiment_name: str,
    X_outer_wave: np.ndarray,
    X_outer_spec: np.ndarray,
    y_outer: np.ndarray,
    best_epochs: int,
    class_weights: torch.Tensor,
    seed: int = SEED,
) -> nn.Module:
    """Retrain fresh model on full outer-training partition for exactly best_epochs."""
    set_seed(seed)
    is_dual = (experiment_name == "experiment_b")
    is_augmented = (experiment_name == "experiment_a")

    if is_dual:
        model = SpectralFusionCNN(num_classes=4, input_channels=1, waveform_length=2048, spectral_bins=1025, fc_units=FC_UNITS, dropout_rate=DROPOUT)
        dataset = DualBranchVibrationDataset(X_outer_wave, X_outer_spec, y_outer)
    else:
        model = VibrationCNN1D(fc_units=FC_UNITS, dropout_rate=DROPOUT)
        dataset = AugmentedVibrationDataset(X_outer_wave, y_outer, augment=is_augmented, seed=seed)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(1, best_epochs + 1):
        if is_dual:
            for b_wave, b_spec, b_y in loader:
                optimizer.zero_grad()
                logits = model(b_wave, spectral_input=b_spec)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
        else:
            for b_x, b_y in loader:
                optimizer.zero_grad()
                logits = model(b_x)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

    return model


def benchmark_cpu_latency(model: nn.Module, is_dual: bool = False) -> Dict[str, Any]:
    """Measure single-window and batch-32 CPU inference latency."""
    model.eval()
    times_single = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            if is_dual:
                _ = model(torch.randn(1, 1, 2048), spectral_input=torch.randn(1, 1, 1025))
            else:
                _ = model(torch.randn(1, 1, 2048))

    # Benchmark 200 single-window calls
    with torch.no_grad():
        for _ in range(200):
            if is_dual:
                w = torch.randn(1, 1, 2048)
                s = torch.randn(1, 1, 1025)
                t0 = time.perf_counter()
                _ = model(w, spectral_input=s)
                t1 = time.perf_counter()
            else:
                x = torch.randn(1, 1, 2048)
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
            times_single.append((t1 - t0) * 1000.0)

    return {
        "single_window_median_ms": round(float(np.median(times_single)), 4),
        "single_window_p95_ms": round(float(np.percentile(times_single, 95)), 4),
        "single_window_mean_ms": round(float(np.mean(times_single)), 4),
    }


def run_experiment_fold(
    experiment_name: str,
    fold_idx: int,
    split_info: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    """Execute one fold for a given experiment."""
    logger.info(f"--- Running {experiment_name} | Fold {fold_idx} (held-out {split_info['val_load_hp']} HP) ---")
    t_fold_start = time.time()

    train_recs = [str(r) for r in split_info["train_recording_ids"]]
    eval_recs = [str(r) for r in split_info["val_recording_ids"]]

    # 1. Deterministic inner split
    inner_train_recs, inner_val_recs = create_inner_split(train_recs, manifest, seed=SEED)

    # 2. Extract raw partitions
    X_in_tr_wave, X_in_tr_spec, y_in_tr, _ = aggregate_partition(inner_train_recs, cache)
    X_in_val_wave, X_in_val_spec, y_in_val, _ = aggregate_partition(inner_val_recs, cache)
    X_out_tr_wave, X_out_tr_spec, y_out_tr, _ = aggregate_partition(train_recs, cache)
    X_out_ev_wave, X_out_ev_spec, y_out_ev, ev_meta = aggregate_partition(eval_recs, cache)

    # 3. Fit normalizers strictly on training partition
    wave_norm = VibrationNormalizer()
    wave_norm.fit(X_out_tr_wave)

    X_in_tr_wave_norm = wave_norm.transform(X_in_tr_wave)
    X_in_val_wave_norm = wave_norm.transform(X_in_val_wave)
    X_out_tr_wave_norm = wave_norm.transform(X_out_tr_wave)
    X_out_ev_wave_norm = wave_norm.transform(X_out_ev_wave)

    # For Experiment B, fit SpectralNormalizer on training spectra
    is_dual = (experiment_name == "experiment_b")
    if is_dual:
        spec_norm = SpectralNormalizer()
        spec_norm.fit(X_out_tr_spec)
        X_in_tr_spec_norm = spec_norm.transform(X_in_tr_spec)
        X_in_val_spec_norm = spec_norm.transform(X_in_val_spec)
        X_out_tr_spec_norm = spec_norm.transform(X_out_tr_spec)
        X_out_ev_spec_norm = spec_norm.transform(X_out_ev_spec)
    else:
        X_in_tr_spec_norm = X_in_tr_spec
        X_in_val_spec_norm = X_in_val_spec
        X_out_tr_spec_norm = X_out_tr_spec
        X_out_ev_spec_norm = X_out_ev_spec

    ckpt_name = f"{experiment_name}_fold_{fold_idx}.pth"
    ckpt_path = CHECKPOINTS_DIR / ckpt_name

    if ckpt_path.exists():
        logger.info(f"[{experiment_name} Fold {fold_idx}] Found existing checkpoint at {ckpt_path}. Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_epoch = ckpt.get("best_epoch", 1)
        history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_macro_f1": []})
        if is_dual:
            model = SpectralFusionCNN(
                num_classes=4, input_channels=1, waveform_length=2048, spectral_bins=1025, fc_units=FC_UNITS, dropout_rate=DROPOUT
            )
        else:
            model = VibrationCNN1D(fc_units=FC_UNITS, dropout_rate=DROPOUT)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # 4. Class weights strictly on training
        weights = compute_balanced_weights(y_out_tr)

        # 5. Inner training to find best epoch E*
        min_epochs = 5 if experiment_name == "experiment_c" else 1
        best_epoch, history = train_inner_model(
            experiment_name=experiment_name,
            X_tr_wave=X_in_tr_wave_norm,
            X_tr_spec=X_in_tr_spec_norm,
            y_tr=y_in_tr,
            X_val_wave=X_in_val_wave_norm,
            X_val_spec=X_in_val_spec_norm,
            y_val=y_in_val,
            class_weights=weights,
            seed=SEED,
            min_epochs=min_epochs,
        )
        logger.info(f"[{experiment_name} Fold {fold_idx}] Selected best epoch E* = {best_epoch}")

        # 6. Retrain outer model on full outer-training partition
        model = train_outer_model(
            experiment_name=experiment_name,
            X_outer_wave=X_out_tr_wave_norm,
            X_outer_spec=X_out_tr_spec_norm,
            y_outer=y_out_tr,
            best_epochs=best_epoch,
            class_weights=weights,
            seed=SEED,
        )

        # Save checkpoint
        torch.save({
            "experiment_name": experiment_name,
            "fold_idx": fold_idx,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "is_dual": is_dual,
            "history": history,
        }, ckpt_path)

    # 7. Evaluate on held-out evaluation load
    model.eval()
    with torch.no_grad():
        if is_dual:
            w_tensor = torch.tensor(X_out_ev_wave_norm, dtype=torch.float32).unsqueeze(1)
            s_tensor = torch.tensor(X_out_ev_spec_norm, dtype=torch.float32).unsqueeze(1)
            logits = model(w_tensor, spectral_input=s_tensor)
        else:
            x_tensor = torch.tensor(X_out_ev_wave_norm, dtype=torch.float32).unsqueeze(1)
            logits = model(x_tensor)

        probs = torch.softmax(logits, dim=1).numpy()
        preds = torch.argmax(logits, dim=1).numpy()

    # Metrics
    acc = float(accuracy_score(y_out_ev, preds))
    bal_acc = float(balanced_accuracy_score(y_out_ev, preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_out_ev, preds, average="macro", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_out_ev, preds, average=None, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_out_ev, preds, labels=[0, 1, 2, 3])

    # Per-recording accuracy
    rec_results = {}
    for r_id in eval_recs:
        indices = [i for i, m in enumerate(ev_meta) if str(m["recording_id"]) == str(r_id)]
        r_y = y_out_ev[indices]
        r_p = preds[indices]
        r_acc = float(accuracy_score(r_y, r_p))
        r_errs = int(np.sum(r_y != r_p))
        rec_results[r_id] = {
            "total_windows": len(indices),
            "correct_windows": len(indices) - r_errs,
            "error_windows": r_errs,
            "accuracy": round(r_acc, 4),
            "fault_class": CLASS_NAMES[r_y[0]],
        }

    # Latency benchmark
    latency = benchmark_cpu_latency(model, is_dual=is_dual)
    fold_duration = time.time() - t_fold_start

    # Parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fold_result = {
        "experiment_name": experiment_name,
        "fold": fold_idx,
        "held_out_load_hp": split_info["val_load_hp"],
        "best_epoch": best_epoch,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class": {
            cls: {
                "precision": round(float(prec_cls[i]), 4),
                "recall": round(float(rec_cls[i]), 4),
                "f1": round(float(f1_cls[i]), 4),
                "support": int(supp_cls[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "recording_results": rec_results,
        "recording_172": rec_results.get("172", None),
        "parameter_count": param_count,
        "latency_ms": latency,
        "training_duration_sec": round(fold_duration, 2),
        "history": history,
        "predictions": preds.tolist(),
        "targets": y_out_ev.tolist(),
        "metadata": ev_meta,
    }

    logger.info(
        f"[{experiment_name} Fold {fold_idx}] Acc: {acc*100:.2f}% | Macro-F1: {f1_macro*100:.2f}% | "
        f"Rec 172 Acc: {rec_results.get('172', {}).get('accuracy', 'N/A')}"
    )
    return fold_result


def main():
    logger.info("Starting Phase 11.4: Targeted Model Improvement & Controlled Ablation Experiments...")
    t0_main = time.time()

    with open(SPLITS_PATH, "r") as f:
        splits_data = json.load(f)
    splits = splits_data["folds"]

    df_manifest = pd.read_csv(MANIFEST_PATH)
    df_manifest["recording_id"] = df_manifest["recording_id"].astype(str)
    manifest_idx = df_manifest.set_index("recording_id")

    # Extract windows and spectra
    cache = extract_all_windows(df_manifest, RAW_DATA_DIR)

    experiments = [
        ("baseline", "Frozen Baseline 1D CNN"),
        ("experiment_a", "Training-Only Signal Augmentation"),
        ("experiment_b", "Hilbert Envelope-Spectrum Dual-Branch CNN"),
        ("experiment_c", "Training Schedule Adjustment (Min-Epoch Floor = 5)"),
    ]

    all_results: Dict[str, Dict[str, Any]] = {}
    cross_exp_summary: List[Dict[str, Any]] = []

    for exp_id, exp_desc in experiments:
        logger.info(f"\n==========================================")
        logger.info(f"EXECUTING EXPERIMENT: {exp_id.upper()} ({exp_desc})")
        logger.info(f"==========================================")

        exp_folds = []
        pooled_preds = []
        pooled_targets = []
        pooled_meta = []

        for fold_idx in [1, 2, 3]:
            split_info = splits[f"fold_{fold_idx}"]
            res = run_experiment_fold(
                experiment_name=exp_id,
                fold_idx=fold_idx,
                split_info=split_info,
                cache=cache,
                manifest=manifest_idx,
            )
            exp_folds.append(res)
            pooled_preds.extend(res["predictions"])
            pooled_targets.extend(res["targets"])
            pooled_meta.extend(res["metadata"])

        # Compute pooled metrics across all 7,038 windows
        pooled_acc = float(accuracy_score(pooled_targets, pooled_preds))
        pooled_bal_acc = float(balanced_accuracy_score(pooled_targets, pooled_preds))
        p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(
            pooled_targets, pooled_preds, average="macro", zero_division=0
        )
        cm_pooled = confusion_matrix(pooled_targets, pooled_preds, labels=[0, 1, 2, 3])

        mean_acc = float(np.mean([f["accuracy"] for f in exp_folds]))
        std_acc = float(np.std([f["accuracy"] for f in exp_folds]))
        mean_f1 = float(np.mean([f["macro_f1"] for f in exp_folds]))
        std_f1 = float(np.std([f["macro_f1"] for f in exp_folds]))

        # Recording 172 performance in Fold 3
        rec172_res = exp_folds[2]["recording_172"]

        exp_summary = {
            "experiment_id": exp_id,
            "description": exp_desc,
            "mean_fold_accuracy": round(mean_acc, 4),
            "std_fold_accuracy": round(std_acc, 4),
            "mean_fold_macro_f1": round(mean_f1, 4),
            "std_fold_macro_f1": round(std_f1, 4),
            "pooled_accuracy": round(pooled_acc, 4),
            "pooled_macro_f1": round(float(p_f1), 4),
            "fold_1_f1": exp_folds[0]["macro_f1"],
            "fold_2_f1": exp_folds[1]["macro_f1"],
            "fold_3_f1": exp_folds[2]["macro_f1"],
            "recording_172_accuracy": rec172_res["accuracy"] if rec172_res else None,
            "recording_172_errors": rec172_res["error_windows"] if rec172_res else None,
            "parameters": exp_folds[0]["parameter_count"],
            "latency_median_ms": exp_folds[0]["latency_ms"]["single_window_median_ms"],
            "latency_p95_ms": exp_folds[0]["latency_ms"]["single_window_p95_ms"],
        }
        cross_exp_summary.append(exp_summary)

        # Save fold metrics for this experiment
        all_results[exp_id] = {
            "summary": exp_summary,
            "folds": exp_folds,
            "pooled_confusion_matrix": cm_pooled.tolist(),
        }

    # 1. Save fold_metrics.json
    with open(EXP_DIR / "fold_metrics.json", "w") as f:
        # Strip large raw predictions before saving
        clean_results = {}
        for eid, edata in all_results.items():
            clean_folds = []
            for fld in edata["folds"]:
                fcopy = {k: v for k, v in fld.items() if k not in ["predictions", "targets", "metadata"]}
                clean_folds.append(fcopy)
            clean_results[eid] = {
                "summary": edata["summary"],
                "folds": clean_folds,
                "pooled_confusion_matrix": edata["pooled_confusion_matrix"],
            }
        json.dump(clean_results, f, indent=2)

    # 2. Save cross_fold_comparison.json
    with open(EXP_DIR / "cross_fold_comparison.json", "w") as f:
        json.dump(cross_exp_summary, f, indent=2)

    # 3. Save per_class_comparison.csv
    per_class_rows = []
    for exp_id in all_results:
        for f_idx, fld in enumerate(all_results[exp_id]["folds"], start=1):
            for cls_name, c_metrics in fld["per_class"].items():
                per_class_rows.append({
                    "experiment": exp_id,
                    "fold": f_idx,
                    "held_out_load_hp": fld["held_out_load_hp"],
                    "class_name": cls_name,
                    "precision": c_metrics["precision"],
                    "recall": c_metrics["recall"],
                    "f1_score": c_metrics["f1"],
                    "support": c_metrics["support"],
                })
    pd.DataFrame(per_class_rows).to_csv(EXP_DIR / "per_class_comparison.csv", index=False)

    # 4. Save per_load_comparison.csv
    per_load_rows = []
    for exp_id in all_results:
        for fld in all_results[exp_id]["folds"]:
            per_load_rows.append({
                "experiment": exp_id,
                "held_out_load_hp": fld["held_out_load_hp"],
                "accuracy": fld["accuracy"],
                "balanced_accuracy": fld["balanced_accuracy"],
                "macro_f1": fld["macro_f1"],
                "rec_172_acc": fld["recording_results"].get("172", {}).get("accuracy", None),
                "rec_172_errs": fld["recording_results"].get("172", {}).get("error_windows", None),
            })
    pd.DataFrame(per_load_rows).to_csv(EXP_DIR / "per_load_comparison.csv", index=False)

    # 5. Save per_severity_comparison.csv
    per_sev_rows = []
    for exp_id in all_results:
        for f_idx, fld in enumerate(all_results[exp_id]["folds"], start=1):
            for r_id, r_info in fld["recording_results"].items():
                meta_row = df_manifest[df_manifest["recording_id"] == str(r_id)].iloc[0]
                per_sev_rows.append({
                    "experiment": exp_id,
                    "fold": f_idx,
                    "recording_id": r_id,
                    "fault_class": meta_row["fault_class"],
                    "defect_size_inches": meta_row["fault_size_inches"],
                    "load_hp": meta_row["load_hp"],
                    "accuracy": r_info["accuracy"],
                    "error_windows": r_info["error_windows"],
                })
    pd.DataFrame(per_sev_rows).to_csv(EXP_DIR / "per_severity_comparison.csv", index=False)

    # 6. Save training_history.json
    history_dict = {}
    for exp_id in all_results:
        history_dict[exp_id] = {
            f"fold_{i+1}": fld["history"] for i, fld in enumerate(all_results[exp_id]["folds"])
        }
    with open(EXP_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # 7. Save model_parameters.json and latency_measurements.json
    params_dict = {exp["experiment_id"]: exp["parameters"] for exp in cross_exp_summary}
    latency_dict = {
        exp["experiment_id"]: {
            "median_ms": exp["latency_median_ms"],
            "p95_ms": exp["latency_p95_ms"],
        }
        for exp in cross_exp_summary
    }
    with open(EXP_DIR / "model_parameters.json", "w") as f:
        json.dump(params_dict, f, indent=2)
    with open(EXP_DIR / "latency_measurements.json", "w") as f:
        json.dump(latency_dict, f, indent=2)

    # 8. Save experiment_config.json
    config_dict = {
        "experiments": [
            {"id": "baseline", "model": "VibrationCNN1D", "params": 73156, "augmentation": False, "min_epochs": 1},
            {"id": "experiment_a", "model": "VibrationCNN1D", "params": 73156, "augmentation": True, "min_epochs": 1},
            {"id": "experiment_b", "model": "SpectralFusionCNN", "params": 89332, "augmentation": False, "min_epochs": 1},
            {"id": "experiment_c", "model": "VibrationCNN1D", "params": 73156, "augmentation": False, "min_epochs": 5},
        ],
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "seed": SEED,
            "optimizer": "AdamW",
            "loss": "ClassWeightedCrossEntropyLoss",
        },
        "validation_protocol": "Leave-One-Load-Out (LOLO) 3 Folds",
        "protected_holdout_quarantined": sorted(list(PROTECTED_HOLDOUT_IDS)),
    }
    with open(EXP_DIR / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 9. Generate Publication Figures
    logger.info("Generating publication figures in outputs/figures/phase11_4/...")
    
    # Figure 1: Model Comparison (Macro-F1 & Recording 172 Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    df_comp = pd.DataFrame(cross_exp_summary)
    
    colors = ["#2b5c8f", "#388e3c", "#d32f2f", "#7b1fa2"]
    bars1 = ax1.bar(df_comp["experiment_id"], df_comp["pooled_macro_f1"] * 100, color=colors, alpha=0.85, width=0.55)
    ax1.set_ylim(95, 100.5)
    ax1.set_title("Pooled Cross-Load Macro-F1 (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.grid(True, alpha=0.3, ls="--")
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(df_comp["experiment_id"], df_comp["recording_172_accuracy"] * 100, color=colors, alpha=0.85, width=0.55)
    ax2.set_ylim(60, 102)
    ax2.set_title("Recording 172 (0.014\" IR @ 3 HP) Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3, ls="--")
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "cross_experiment_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: Confusion Matrices for All 4 Experiments (Pooled)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for idx, (exp_id, exp_desc) in enumerate(experiments):
        ax = ax_list[idx]
        cm = np.array(all_results[exp_id]["pooled_confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            annot_kws={"size": 11, "fontweight": "bold"}
        )
        acc_val = all_results[exp_id]["summary"]["pooled_accuracy"] * 100
        f1_val = all_results[exp_id]["summary"]["pooled_macro_f1"] * 100
        ax.set_title(f"{exp_id.upper()} | Acc: {acc_val:.2f}% | F1: {f1_val:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "confusion_matrices_comparison.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Figure 3: Fold 3 Training History Comparison
    fig, (ax_tl, ax_vf) = plt.subplots(1, 2, figsize=(14, 5))
    has_history = False
    for exp_id, exp_desc in experiments:
        h3 = history_dict[exp_id]["fold_3"]
        if h3.get("train_loss") and len(h3["train_loss"]) > 0:
            epochs_arr = range(1, len(h3["train_loss"]) + 1)
            ax_tl.plot(epochs_arr, h3["train_loss"], marker="o", label=exp_id)
            ax_vf.plot(epochs_arr, h3["val_macro_f1"], marker="s", label=exp_id)
            has_history = True

    ax_tl.set_title("Fold 3 Training Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("CrossEntropy Loss")
    ax_tl.grid(True, alpha=0.3, ls="--")
    if has_history:
        ax_tl.legend()

    ax_vf.set_title("Fold 3 Inner-Validation Macro-F1 vs. Epoch", fontsize=11, fontweight="bold")
    ax_vf.set_xlabel("Epoch")
    ax_vf.set_ylabel("Inner-Val Macro-F1")
    ax_vf.grid(True, alpha=0.3, ls="--")
    if has_history:
        ax_vf.legend()

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "training_curves_fold_3.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    total_duration = time.time() - t0_main
    logger.info(f"Phase 11.4 experiment run completed in {total_duration:.2f} seconds.")
    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(pd.DataFrame(cross_exp_summary)[[
        "experiment_id", "pooled_accuracy", "pooled_macro_f1", "fold_1_f1", "fold_2_f1", "fold_3_f1", "recording_172_accuracy", "parameters", "latency_median_ms"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
