"""Phase 10.4: Controlled Frequency-Domain Feature Ablation Experiment.

Investigates whether frequency-domain information (one-sided FFT magnitude spectrum
or Hilbert envelope spectrum) improves fault-class discrimination in InspectAI,
with specific focus on inter-raceway confusion between Inner Race (209) and Outer Race (234).

Configurations evaluated:
- Configuration A: Baseline Raw Waveform (VibrationCNN1D, 73,156 parameters).
- Configuration B: Raw Waveform + FFT Spectrum (SpectralFusionCNN, 89,604 parameters).
- Configuration C: Raw Waveform + Hilbert Envelope Spectrum (SpectralFusionCNN, 89,604 parameters).

Strict experimental guarantees:
1. Evaluates strictly on the 9 development recordings (2,110 windows) via 2-fold recording-level CV.
2. The 4 held-out test recordings (97, 169, 185, 197) remain completely untouched.
3. The frozen final model checkpoint and previous experiment artifacts are never overwritten.
4. Normalization (waveform mean/std and spectral mean/std) is fitted strictly on training folds.
5. Identical random seeds, AdamW optimizer, learning rates, early stopping, and CPU hardware.
6. Exports all predictions, checkpoints, and metrics to outputs/experiments/phase10_4/.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.frequency_features import (
    compute_fft_magnitude_spectrum,
    compute_hilbert_envelope_spectrum,
    SpectralNormalizer,
)
from src.models.cnn_1d import VibrationCNN1D
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.training.trainer import DEFAULT_CLASS_NAMES, ModelTrainer
from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.FrequencyAblation")

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_4"


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across numpy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_cpu_latency(model: torch.nn.Module, input_shape: Tuple[int, ...], n_runs: int = 100) -> float:
    """Measure single-window CPU inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(latencies))


class CombinedFeatureDataset(Dataset):
    """PyTorch Dataset wrapping normalized waveform and optional spectral features."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        metadata_df: pd.DataFrame,
    ) -> None:
        self.features = features.float()
        self.labels = labels.long()
        self.metadata_df = metadata_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample_meta = {
            "window_id": str(self.metadata_df.iloc[idx]["window_id"]),
            "recording_id": str(self.metadata_df.iloc[idx]["recording_id"]),
            "fault_type": str(self.metadata_df.iloc[idx].get("fault_type", "")),
            "load_hp": int(self.metadata_df.iloc[idx].get("load_hp", 0)),
        }
        return self.features[idx], self.labels[idx], sample_meta


def precompute_spectral_features(
    raw_windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute FFT and Hilbert envelope spectra for all windows.

    Args:
        raw_windows: Array of shape (N, 2048).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - fft_spectra: Shape (N, 1025)
            - hilbert_spectra: Shape (N, 1025)
    """
    logger.info(f"Precomputing FFT and Hilbert envelope spectra for {len(raw_windows)} windows...")
    t0 = time.time()
    _, fft_spectra = compute_fft_magnitude_spectrum(raw_windows, sampling_rate_hz=12000)
    _, hilbert_spectra, _ = compute_hilbert_envelope_spectrum(raw_windows, sampling_rate_hz=12000)
    elapsed = time.time() - t0
    logger.info(f"Spectral precomputation complete in {elapsed:.2f}s. Shapes: FFT {fft_spectra.shape}, Hilbert {hilbert_spectra.shape}")
    return fft_spectra, hilbert_spectra


def train_and_eval_fold(
    fold_id: int,
    config_name: str,
    spectral_type: Optional[str],  # None for baseline, "fft", or "hilbert"
    model_factory: Callable[[], torch.nn.Module],
    raw_windows: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    fft_spectra: Optional[np.ndarray],
    hilbert_spectra: Optional[np.ndarray],
    exp_config: Dict[str, Any],
    config_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and evaluate a single cross-validation fold for a given feature configuration."""
    logger.info(f"=== Config: '{config_name}' | Fold: {fold_id} ===")
    train_cfg = exp_config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    max_epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("learning_rate", 0.001))
    wd = float(train_cfg.get("weight_decay", 0.0001))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 7))
    min_delta = float(train_cfg.get("early_stopping", {}).get("min_delta", 1e-4))
    device = "cpu"

    set_seed(seed + fold_id)

    # 1. Fit Waveform Normalizer strictly on training fold
    train_wave = raw_windows[train_indices]
    val_wave = raw_windows[val_indices]
    train_recs = metadata_df.iloc[train_indices]["recording_id"].astype(str).unique().tolist()
    val_recs = metadata_df.iloc[val_indices]["recording_id"].astype(str).unique().tolist()

    wave_norm = VibrationNormalizer(epsilon=1e-8, strategy="global")
    wave_norm.fit(train_wave, source_recording_ids=train_recs)
    wave_norm.save(config_dir / f"normalizer_fold_{fold_id}.json")

    norm_train_wave = wave_norm.transform(train_wave)[:, np.newaxis, :]  # (N_train, 1, 2048)
    norm_val_wave = wave_norm.transform(val_wave)[:, np.newaxis, :]      # (N_val, 1, 2048)

    # 2. Handle Spectral Features if applicable
    if spectral_type is not None:
        if spectral_type == "fft":
            raw_train_spec = fft_spectra[train_indices]
            raw_val_spec = fft_spectra[val_indices]
        elif spectral_type == "hilbert":
            raw_train_spec = hilbert_spectra[train_indices]
            raw_val_spec = hilbert_spectra[val_indices]
        else:
            raise ValueError(f"Unknown spectral_type: {spectral_type}")

        # Fit Spectral Normalizer strictly on training fold
        spec_norm = SpectralNormalizer(epsilon=1e-8)
        spec_norm.fit(raw_train_spec)
        spec_norm.save(config_dir / f"spectral_normalizer_fold_{fold_id}.json")

        norm_train_spec = spec_norm.transform(raw_train_spec)[:, np.newaxis, :]  # (N_train, 1, 1025)
        norm_val_spec = spec_norm.transform(raw_val_spec)[:, np.newaxis, :]      # (N_val, 1, 1025)

        # Concatenate into (N, 1, 3073)
        train_features = np.concatenate([norm_train_wave, norm_train_spec], axis=-1)
        val_features = np.concatenate([norm_val_wave, norm_val_spec], axis=-1)
    else:
        train_features = norm_train_wave  # (N_train, 1, 2048)
        val_features = norm_val_wave      # (N_val, 1, 2048)

    # 3. Create Datasets and DataLoaders
    train_ds = CombinedFeatureDataset(
        features=torch.from_numpy(train_features),
        labels=torch.from_numpy(labels[train_indices]),
        metadata_df=metadata_df.iloc[train_indices],
    )
    val_ds = CombinedFeatureDataset(
        features=torch.from_numpy(val_features),
        labels=torch.from_numpy(labels[val_indices]),
        metadata_df=metadata_df.iloc[val_indices],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4. Instantiate Model
    set_seed(seed + fold_id * 10)
    model = model_factory()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config_dir,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        class_names=DEFAULT_CLASS_NAMES,
        config=exp_config,
    )

    checkpoint_path = config_dir / f"fold_{fold_id}_best.pth"

    start_time = time.time()
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        fold_id=fold_id,
        checkpoint_path=checkpoint_path,
        normalizer_meta=wave_norm.to_dict(),
        lr_scheduler=lr_scheduler,
    )
    duration_sec = time.time() - start_time

    # 5. Reload Best Checkpoint and Evaluate
    if checkpoint_path.is_file():
        trainer.load_checkpoint(checkpoint_path)

    val_metrics = trainer.evaluate(val_loader)
    input_shape = (1, 1, 3073) if spectral_type else (1, 1, 2048)
    latency_ms = measure_cpu_latency(model, input_shape=input_shape)

    # 6. Extract Out-of-Fold Predictions
    model.eval()
    metadata_rows = []
    with torch.no_grad():
        for inputs, targets, sample_meta in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for i in range(len(targets)):
                metadata_rows.append({
                    "configuration": config_name,
                    "fold": fold_id,
                    "window_id": sample_meta["window_id"][i],
                    "recording_id": str(sample_meta["recording_id"][i]),
                    "true_class_id": int(targets[i]),
                    "true_class_name": DEFAULT_CLASS_NAMES[int(targets[i])],
                    "pred_class_id": int(preds[i]),
                    "pred_class_name": DEFAULT_CLASS_NAMES[int(preds[i])],
                    "prob_Normal": float(probs[i, 0]),
                    "prob_Ball_Fault": float(probs[i, 1]),
                    "prob_Inner_Race_Fault": float(probs[i, 2]),
                    "prob_Outer_Race_Fault": float(probs[i, 3]),
                })

    df_fold_preds = pd.DataFrame(metadata_rows)

    fold_summary = {
        "configuration": config_name,
        "fold": fold_id,
        "parameters": param_count,
        "duration_sec": round(duration_sec, 2),
        "best_epoch": fit_result["best_epoch"],
        "total_epochs": fit_result.get("total_epochs_trained", len(fit_result.get("history", []))),
        "val_loss": round(val_metrics["loss"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "cpu_latency_ms": round(latency_ms, 3),
        "train_recordings": train_recs,
        "val_recordings": val_recs,
    }

    return fold_summary, df_fold_preds


def run_configuration(
    config_name: str,
    spectral_type: Optional[str],
    model_factory: Callable[[], torch.nn.Module],
    raw_windows: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    fold_splits: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fft_spectra: Optional[np.ndarray],
    hilbert_spectra: Optional[np.ndarray],
    exp_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute complete 2-fold cross validation for a configuration."""
    config_dir = OUTPUT_BASE_DIR / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    fold_dfs = []

    for fold_id in [0, 1]:
        train_idx, val_idx = fold_splits[fold_id]
        f_summary, f_df = train_and_eval_fold(
            fold_id=fold_id,
            config_name=config_name,
            spectral_type=spectral_type,
            model_factory=model_factory,
            raw_windows=raw_windows,
            labels=labels,
            metadata_df=metadata_df,
            train_indices=train_idx,
            val_indices=val_idx,
            fft_spectra=fft_spectra,
            hilbert_spectra=hilbert_spectra,
            exp_config=exp_config,
            config_dir=config_dir,
        )
        fold_summaries.append(f_summary)
        fold_dfs.append(f_df)

    # Combine OOF Predictions across both folds (covers all 2,110 dev windows)
    df_all_oof = pd.concat(fold_dfs, ignore_index=True)
    df_all_oof.to_csv(config_dir / "oof_predictions.csv", index=False)

    # Compute Pooled OOF Metrics
    y_true = df_all_oof["true_class_id"].values
    y_pred = df_all_oof["pred_class_id"].values
    oof_accuracy = float(accuracy_score(y_true, y_pred))
    oof_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=DEFAULT_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Per-recording breakdown
    rec_breakdown = {}
    for rec_id, group in df_all_oof.groupby("recording_id"):
        rec_acc = float(accuracy_score(group["true_class_id"], group["pred_class_id"]))
        rec_breakdown[rec_id] = {
            "true_class": group["true_class_name"].iloc[0],
            "count": len(group),
            "accuracy": round(rec_acc, 4),
            "pred_distribution": group["pred_class_name"].value_counts().to_dict(),
        }

    summary = {
        "configuration": config_name,
        "spectral_type": spectral_type,
        "parameters": fold_summaries[0]["parameters"],
        "oof_accuracy": round(oof_accuracy, 4),
        "oof_macro_f1": round(oof_macro_f1, 4),
        "mean_latency_ms": round(float(np.mean([s["cpu_latency_ms"] for s in fold_summaries])), 3),
        "fold_metrics": fold_summaries,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "recording_breakdown": rec_breakdown,
    }

    with open(config_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Config '{config_name}' Complete -> OOF Accuracy: {oof_accuracy*100:.2f}%, "
        f"OOF Macro-F1: {oof_macro_f1*100:.2f}%, Latency: {summary['mean_latency_ms']} ms"
    )
    return summary, df_all_oof


def main() -> None:
    """Main execution orchestrating the Phase 10.4 frequency ablation."""
    logger.info("Starting Phase 10.4 Frequency-Domain Feature Ablation Experiment...")
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()

    # 1. Load full dataset archive and metadata
    full_ds = CWRUVibrationDataset(split=None, config=config)
    raw_windows = full_ds.raw_windows
    labels = full_ds.raw_labels
    metadata_df = full_ds.metadata_df.copy()

    # 2. Extract 2-fold cross-validation partition indices (strictly development recordings)
    fold_splits = {}
    for fid in [0, 1]:
        train_ds = CWRUVibrationDataset(split="train", fold=fid, config=config)
        val_ds = CWRUVibrationDataset(split="val", fold=fid, config=config)
        fold_splits[fid] = (train_ds.subset_indices, val_ds.subset_indices)
        logger.info(f"Fold {fid}: Train samples = {len(train_ds.subset_indices)}, Val samples = {len(val_ds.subset_indices)}")

    # Verify dev pool size (2,110 windows) and strict absence of held-out recordings
    dev_indices = np.concatenate([fold_splits[0][0], fold_splits[0][1]])
    if len(dev_indices) != 2110:
        raise RuntimeError(f"Expected 2,110 development windows, got {len(dev_indices)}.")

    held_out_ids = {"97", "169", "185", "197"}
    present_held_out = set(metadata_df.iloc[dev_indices]["recording_id"].astype(str)).intersection(held_out_ids)
    if present_held_out:
        raise RuntimeError(f"FATAL: Held-out recordings detected in development dataset: {present_held_out}")

    logger.info(f"Confirmed 2,110 development windows across {metadata_df.iloc[dev_indices]['recording_id'].nunique()} recordings. Zero held-out test leakage.")

    # 3. Precompute Spectral Representations (FFT and Hilbert)
    fft_spectra, hilbert_spectra = precompute_spectral_features(raw_windows)

    # 4. Define Model Factories
    def baseline_factory() -> VibrationCNN1D:
        return VibrationCNN1D(
            num_classes=4,
            input_channels=1,
            conv_filters=[32, 64, 128],
            kernel_sizes=[15, 7, 3],
            pool_sizes=[4, 4, 2],
            dropout_rate=0.3,
            fc_units=64,
        )

    def fft_fusion_factory() -> SpectralFusionCNN:
        return SpectralFusionCNN(
            num_classes=4,
            input_channels=1,
            waveform_length=2048,
            spectral_bins=1025,
            spectral_type="fft",
            fc_units=64,
            dropout_rate=0.3,
        )

    def hilbert_fusion_factory() -> SpectralFusionCNN:
        return SpectralFusionCNN(
            num_classes=4,
            input_channels=1,
            waveform_length=2048,
            spectral_bins=1025,
            spectral_type="hilbert",
            fc_units=64,
            dropout_rate=0.3,
        )

    configurations = [
        ("baseline_waveform", None, baseline_factory),
        ("waveform_plus_fft", "fft", fft_fusion_factory),
        ("waveform_plus_hilbert", "hilbert", hilbert_fusion_factory),
    ]

    all_summaries = {}
    for cfg_name, spec_type, factory in configurations:
        summary, _ = run_configuration(
            config_name=cfg_name,
            spectral_type=spec_type,
            model_factory=factory,
            raw_windows=raw_windows,
            labels=labels,
            metadata_df=metadata_df,
            fold_splits=fold_splits,
            fft_spectra=fft_spectra,
            hilbert_spectra=hilbert_spectra,
            exp_config=config,
        )
        all_summaries[cfg_name] = summary

    # Export Master Experiment Summary
    master_summary_file = OUTPUT_BASE_DIR / "frequency_ablation_summary.json"
    with open(master_summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"Phase 10.4 Experiment Complete! Master summary saved to {master_summary_file}")


if __name__ == "__main__":
    main()
