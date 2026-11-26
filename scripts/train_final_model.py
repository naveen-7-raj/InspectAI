"""InspectAI - Phase 11.5: Final Model Selection, Training, Freezing & Model Card Generation.

Trains the final release model on all 36 eligible CWRU development recordings:
- Architecture: SpectralFusionCNN (89,604 trainable parameters).
- Inputs: 1D vibration waveform (2,048 samples) + Hilbert envelope spectrum (1,025 bins).
- Excluded holdouts: 97, 169, 185, 197 permanently quarantined.
- Training schedule: 10 epochs (derived from LOLO inner-validation evidence), AdamW, lr=0.0005.
- Normalization: Waveform global normalizer + envelope spectral normalizer fitted on all 36 development recordings.
- Artifacts: Saved to models/final/ and outputs/experiments/phase11_5/.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.ingestion.loader import VibrationDataLoader
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.preprocessing.signal_cleaner import SignalCleaner
from src.preprocessing.windowing import WindowSegmenter
from src.training.multiload_splitter import PROTECTED_HOLDOUT_IDS
from src.training.normalizer import VibrationNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.FinalTraining")

FINAL_MODELS_DIR = REPO_ROOT / "models" / "final"
EXP_11_5_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_5"
MANIFEST_PATH = REPO_ROOT / "outputs" / "experiments" / "phase10_6" / "cwru_40_recordings_manifest.csv"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "cwru"

FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_11_5_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10  # Selected from LOLO inner-validation evidence (Fold 1: 5, Fold 2: 7, Fold 3: 17; mean ~9.7)
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
FC_UNITS = 64
FS = 12000
WINDOW_SIZE = 2048
STEP_SIZE = 512
SPECTRAL_BINS = 1025


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


class FinalDualBranchDataset(Dataset):
    """Dual-branch dataset providing normalized waveform and envelope spectrum tensors."""

    def __init__(self, waveforms: np.ndarray, spectra: np.ndarray, labels: np.ndarray) -> None:
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


def load_and_preprocess_dev_data(
    manifest_path: Path, raw_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """Load, clean, segment, and compute envelope spectra for all 36 eligible development recordings."""
    logger.info("Loading manifest and filtering out protected historical holdouts...")
    df_manifest = pd.read_csv(manifest_path)
    df_manifest["recording_id"] = df_manifest["recording_id"].astype(str)

    # Exclude protected historical holdouts
    dev_df = df_manifest[~df_manifest["recording_id"].isin(PROTECTED_HOLDOUT_IDS)].copy()
    dev_recording_ids = sorted(dev_df["recording_id"].tolist())

    assert len(dev_recording_ids) == 36, f"Expected 36 development recordings, found {len(dev_recording_ids)}"
    for h_id in PROTECTED_HOLDOUT_IDS:
        assert h_id not in dev_recording_ids, f"Protected holdout {h_id} found in development set!"

    logger.info(f"Verified {len(dev_recording_ids)} eligible development recordings.")

    cleaner = SignalCleaner(
        target_sampling_rate=FS,
        remove_dc=True,
        filter_enabled=True,
        filter_type="bandpass",
        low_cutoff_hz=20.0,
        high_cutoff_hz=5000.0,
        filter_order=4,
    )
    loader = VibrationDataLoader(raw_dir)
    segmenter = WindowSegmenter(window_size=WINDOW_SIZE, step_size=STEP_SIZE, sampling_rate=FS)

    all_windows = []
    all_spectra = []
    all_labels = []
    all_metadata = []

    for _, row in dev_df.iterrows():
        rec_id = str(row["recording_id"])
        fname = f"{rec_id}.mat"
        rec = loader.load_file(fname)
        raw_de = rec.channels["DE_time"]

        orig_fs = 48000 if row["fault_class"] == "Normal" else 12000
        cleaned = cleaner.clean_signal(raw_de, orig_fs=orig_fs)

        source_meta = {
            "recording_id": rec_id,
            "filename": fname,
            "fault_class": row["fault_class"],
            "class_id": CLASS_TO_ID[row["fault_class"]],
            "defect_size_inches": float(row["fault_size_inches"]),
            "load_hp": int(row["load_hp"]),
            "verified_rpm": float(row["verified_rpm"]),
        }
        windows, meta_list = segmenter.segment_signal(cleaned.signal, source_meta)
        labels = np.full(len(windows), CLASS_TO_ID[row["fault_class"]], dtype=np.int64)

        # Compute Hilbert envelope spectrum: shape (N, 1025)
        _, env_spectra, _ = compute_hilbert_envelope_spectrum(windows, sampling_rate_hz=FS)

        all_windows.append(windows)
        all_spectra.append(env_spectra)
        all_labels.append(labels)
        all_metadata.extend(meta_list)

    X_wave = np.concatenate(all_windows, axis=0)
    X_spec = np.concatenate(all_spectra, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(f"Total development windows extracted: {len(X_wave)} from {len(dev_recording_ids)} recordings.")
    return X_wave, X_spec, y, dev_recording_ids, all_metadata


def main():
    logger.info("=== InspectAI Phase 11.5: Final Model Training & Freezing ===")
    t_start = time.time()
    set_seed(SEED)

    # 1. Load and extract features for all 36 development recordings
    X_wave, X_spec, y, dev_recording_ids, metadata = load_and_preprocess_dev_data(MANIFEST_PATH, RAW_DATA_DIR)

    # 2. Fit normalizers strictly on all 36 development recordings
    logger.info("Fitting VibrationNormalizer (waveform scalar statistics) on development set...")
    wave_normalizer = VibrationNormalizer(strategy="global")
    wave_normalizer.fit(X_wave)
    X_wave_norm = wave_normalizer.transform(X_wave)

    logger.info("Fitting SpectralNormalizer (1,025-bin frequency statistics) on development set...")
    spec_normalizer = SpectralNormalizer()
    spec_normalizer.fit(X_spec)
    X_spec_norm = spec_normalizer.transform(X_spec)

    # 3. Compute balanced class weights across the 36 development recordings
    unique_classes = np.unique(y)
    raw_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
    weights = np.ones(len(CLASS_NAMES), dtype=np.float32)
    for c, w in zip(unique_classes, raw_weights):
        weights[c] = float(w)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class distribution in training data: {np.bincount(y)}")
    logger.info(f"Computed balanced class weights: {dict(zip(CLASS_NAMES, [round(float(w), 4) for w in weights]))}")

    # 4. Instantiate final SpectralFusionCNN model
    model = SpectralFusionCNN(
        num_classes=4,
        input_channels=1,
        waveform_length=WINDOW_SIZE,
        spectral_bins=SPECTRAL_BINS,
        fc_units=FC_UNITS,
        dropout_rate=DROPOUT,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Instantiated final SpectralFusionCNN model with {param_count:,} trainable parameters.")
    assert param_count == 89604, f"Expected 89,604 parameters, got {param_count}"

    dataset = FinalDualBranchDataset(X_wave_norm, X_spec_norm, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 5. Execute training for exactly 10 epochs
    history = {"epoch": [], "train_loss": [], "epoch_duration_sec": []}
    logger.info(f"Beginning final model training for {EPOCHS} epochs on CPU...")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        t_epoch_start = time.time()
        running_loss = 0.0

        for b_wave, b_spec, b_y in loader:
            optimizer.zero_grad()
            logits = model(b_wave, spectral_input=b_spec)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_y)

        epoch_loss = running_loss / len(dataset)
        epoch_dur = time.time() - t_epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_loss, 5))
        history["epoch_duration_sec"].append(round(epoch_dur, 2))

        logger.info(f"Epoch {epoch:2d}/{EPOCHS} - Training Loss: {epoch_loss:.5f} ({epoch_dur:.2f}s)")

    training_duration_total = round(time.time() - t_start, 2)
    logger.info(f"Final training completed in {training_duration_total} seconds.")

    # 6. Save frozen final model artifacts to models/final/
    ckpt_path = FINAL_MODELS_DIR / "inspectai_final_model.pth"
    norm_path = FINAL_MODELS_DIR / "inspectai_final_normalizer.json"
    spec_norm_path = FINAL_MODELS_DIR / "inspectai_final_spectral_normalizer.json"
    config_path = FINAL_MODELS_DIR / "inspectai_final_config.json"
    model_card_path = FINAL_MODELS_DIR / "inspectai_model_card.json"

    # Save model checkpoint
    torch.save({
        "model_name": "InspectAI_SpectralFusionCNN_Final",
        "model_class": "SpectralFusionCNN",
        "architecture": "DualBranch_Waveform_HilbertEnvelopeSpectrum",
        "num_classes": 4,
        "class_names": CLASS_NAMES,
        "waveform_length": WINDOW_SIZE,
        "spectral_bins": SPECTRAL_BINS,
        "fc_units": FC_UNITS,
        "dropout_rate": DROPOUT,
        "trainable_parameters": param_count,
        "training_epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "ClassWeightedCrossEntropyLoss",
        "class_weights": weights.tolist(),
        "random_seed": SEED,
        "model_state_dict": model.state_dict(),
        "training_duration_sec": training_duration_total,
    }, ckpt_path)
    logger.info(f"Saved frozen model checkpoint to {ckpt_path}")

    # Save normalizers
    wave_normalizer.save(norm_path)
    spec_normalizer.save(spec_norm_path)
    logger.info(f"Saved waveform normalizer to {norm_path}")
    logger.info(f"Saved spectral normalizer to {spec_norm_path}")

    # Compute checksums
    ckpt_sha256 = calculate_sha256(ckpt_path)
    norm_sha256 = calculate_sha256(norm_path)
    spec_norm_sha256 = calculate_sha256(spec_norm_path)

    # Save configuration
    final_config = {
        "model": {
            "name": "SpectralFusionCNN",
            "class_name": "SpectralFusionCNN",
            "num_classes": 4,
            "class_names": CLASS_NAMES,
            "waveform_length": WINDOW_SIZE,
            "spectral_bins": SPECTRAL_BINS,
            "fc_units": FC_UNITS,
            "dropout_rate": DROPOUT,
            "trainable_parameters": param_count,
            "sha256": ckpt_sha256,
        },
        "preprocessing": {
            "sampling_rate_hz": FS,
            "window_size_samples": WINDOW_SIZE,
            "step_size_samples": STEP_SIZE,
            "low_cutoff_hz": 20.0,
            "high_cutoff_hz": 5000.0,
            "waveform_normalizer": {
                "file": "inspectai_final_normalizer.json",
                "strategy": "global",
                "mean": wave_normalizer.mean,
                "std": wave_normalizer.std,
                "sha256": norm_sha256,
            },
            "spectral_normalizer": {
                "file": "inspectai_final_spectral_normalizer.json",
                "n_features": SPECTRAL_BINS,
                "sha256": spec_norm_sha256,
            },
        },
        "training_provenance": {
            "dataset": "CWRU Multi-Load 12 kHz DE (36 Development Recordings)",
            "excluded_protected_holdout_recordings": sorted(list(PROTECTED_HOLDOUT_IDS)),
            "development_recording_ids": dev_recording_ids,
            "total_training_windows": len(X_wave),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "random_seed": SEED,
            "training_device": "CPU",
            "training_duration_seconds": training_duration_total,
        },
        "expected_tensors": {
            "waveform": {"shape": [1, 1, 2048], "dtype": "float32"},
            "envelope_spectrum": {"shape": [1, 1, 1025], "dtype": "float32"},
            "logits": {"shape": [1, 4], "dtype": "float32"},
        },
    }

    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)
    logger.info(f"Saved final configuration to {config_path}")

    # Generate Model Card
    model_card = {
        "model_details": {
            "name": "InspectAI SpectralFusionCNN",
            "version": "1.0-final",
            "type": "Dual-Branch 1D Waveform & Hilbert Envelope Spectrum Convolutional Neural Network",
            "framework": f"PyTorch {torch.__version__}",
            "developer": "InspectAI Systems Engineering",
            "license": "Proprietary / Industrial Diagnostic Research",
            "checkpoint_sha256": ckpt_sha256,
            "waveform_normalizer_sha256": norm_sha256,
            "spectral_normalizer_sha256": spec_norm_sha256,
        },
        "intended_use": {
            "primary_domain": "Industrial rolling element bearing condition monitoring and fault classification.",
            "target_classes": CLASS_NAMES,
            "operating_context": "Deep groove ball bearings (SKF 6205 series) operating under variable motor load (0 to 3 HP, 1,720 to 1,797 RPM).",
            "sampling_frequency_hz": FS,
            "input_window_duration_ms": round(WINDOW_SIZE / FS * 1000.0, 2),
        },
        "validation_benchmark_summary": {
            "protocol": "3-Fold Leave-One-Load-Out (LOLO) Cross-Validation",
            "evaluated_windows": 7038,
            "evaluated_recordings": 30,
            "cross_load_metrics": {
                "fold_1_1hp_f1": 1.0000,
                "fold_2_2hp_f1": 0.9943,
                "fold_3_3hp_f1": 1.0000,
                "pooled_macro_f1": 0.9981,
                "pooled_accuracy": 0.9977,
                "rec_172_3hp_accuracy": 1.0000,
            },
            "per_class_f1": {
                "Normal": 1.0000,
                "Ball Fault": 0.9965,
                "Inner Race Fault": 0.9970,
                "Outer Race Fault": 0.9989,
            },
            "inference_latency": {
                "single_window_median_ms": 1.2024,
                "single_window_p95_ms": 1.9295,
                "throughput_windows_per_second": 831,
            },
        },
        "training_data": {
            "dataset_name": "Case Western Reserve University (CWRU) Bearing Data Center",
            "channels": ["Drive End (DE) accelerometer"],
            "development_recordings_count": 36,
            "total_training_windows": len(X_wave),
            "permanently_quarantined_recordings": sorted(list(PROTECTED_HOLDOUT_IDS)),
        },
        "ethical_and_operational_limitations": {
            "cross_machine_transfer": "Validated on CWRU motor-driven test rig only. Domain adaptation or fine-tuning is required before transferring to different bearing geometries or gearboxes.",
            "early_failure_prediction": "The model classifies active mechanical defect signatures; it does not forecast remaining useful life (RUL) or incipient metallurgical fatigue prior to surface spalling.",
            "unseen_fault_modes": "Detects Normal, Ball, Inner Race, and Outer Race faults. Unmodeled failure modes (e.g., cage fracture, severe misalignment, electrical fluting) may yield uncertain predictions.",
            "uncalibrated_probabilities": "Softmax scores represent model class attribution and do not equal empirical Bayesian posterior probabilities.",
        },
    }

    with open(model_card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    logger.info(f"Saved model card to {model_card_path}")

    # 7. Save outputs under outputs/experiments/phase11_5/
    with open(EXP_11_5_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(EXP_11_5_DIR / "final_model_summary.json", "w") as f:
        json.dump({
            "model_name": "SpectralFusionCNN",
            "parameters": param_count,
            "total_windows": len(X_wave),
            "final_epoch_loss": history["train_loss"][-1],
            "training_duration_sec": training_duration_total,
            "checkpoint_sha256": ckpt_sha256,
            "waveform_normalizer_sha256": norm_sha256,
            "spectral_normalizer_sha256": spec_norm_sha256,
        }, f, indent=2)

    # Plot final training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", color="#1a5276", lw=2.0)
    ax.set_title("Final SpectralFusionCNN Training Loss (All 36 Development Recordings)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Class-Weighted CrossEntropy Loss", fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout()
    loss_fig_path = EXP_11_5_DIR / "final_training_loss.png"
    plt.savefig(loss_fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved final training loss curve to {loss_fig_path}")

    print("\n================ FINAL MODEL TRAINING COMPLETE ================")
    print(f"Model Checkpoint: {ckpt_path} (SHA256: {ckpt_sha256[:16]}...)")
    print(f"Normalizer:       {norm_path}")
    print(f"Spectral Norm:    {spec_norm_path}")
    print(f"Model Card:       {model_card_path}")
    print(f"Trainable Params: {param_count:,}")
    print(f"Training Epochs:  {EPOCHS} | Final Loss: {history['train_loss'][-1]:.5f}")


if __name__ == "__main__":
    main()

