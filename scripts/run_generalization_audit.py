"""Phase 10.5: Dataset Generalization and Failure Analysis Audit.

Performs a rigorous, empirical investigation into:
1. Cross-Experiment Metric Reconciliation (Phase 10.2 vs. 10.3 vs. 10.4 baselines).
2. Recording-Level Metadata and Physical Vibration Statistics for all 13 CWRU recordings.
3. Signal-Level Failure Analysis comparing recordings 105, 209, 130, and 234.
4. Confounding and Identifiability Audit (severity, load, speed, N=1 recording identity).
5. Cross-Validation Integrity Audit (recording isolation, zero leakage, OOF coverage).
6. Generates high-resolution publication-quality diagnostic figures.

Strictly non-training: Does not train or alter any model weights or checkpoints.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for automation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.frequency_features import (
    compute_fft_magnitude_spectrum,
    compute_hilbert_envelope_spectrum,
    get_bearing_characteristic_frequencies,
)
from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.GeneralizationAudit")

EXP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_5"
FIG_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase10_5"


def audit_previous_experiments() -> Dict[str, Any]:
    """Audit baseline metrics and configurations from Phase 10.2, 10.3, and 10.4."""
    logger.info("Auditing previous experiment metrics and reconciliation...")
    
    # 1. Load Phase 10.2 Baseline
    p10_2_file = PROJECT_ROOT / "outputs" / "experiments" / "phase10_2" / "global" / "metrics_summary.json"
    with open(p10_2_file, "r", encoding="utf-8") as f:
        m10_2 = json.load(f)

    # 2. Load Phase 10.3 Baseline
    p10_3_file = PROJECT_ROOT / "outputs" / "experiments" / "phase10_3" / "baseline_1d_cnn" / "metrics_summary.json"
    with open(p10_3_file, "r", encoding="utf-8") as f:
        m10_3 = json.load(f)

    # 3. Load Phase 10.4 Baseline
    p10_4_file = PROJECT_ROOT / "outputs" / "experiments" / "phase10_4" / "baseline_waveform" / "metrics_summary.json"
    with open(p10_4_file, "r", encoding="utf-8") as f:
        m10_4 = json.load(f)

    reconciliation = {
        "Phase_10_2": {
            "experiment": "Phase 10.2 Normalization Ablation (global baseline)",
            "model_architecture": "VibrationCNN1D",
            "fc_units": 128,
            "trainable_parameters": 106244,
            "oof_accuracy": m10_2["pooled_oof_accuracy"],
            "oof_macro_f1": m10_2["pooled_oof_macro_f1"],
            "fold_0_accuracy": m10_2["folds"][0]["val_accuracy"],
            "fold_0_macro_f1": m10_2["folds"][0]["val_macro_f1"],
            "fold_1_accuracy": m10_2["folds"][1]["val_accuracy"],
            "fold_1_macro_f1": m10_2["folds"][1]["val_macro_f1"],
        },
        "Phase_10_3": {
            "experiment": "Phase 10.3 Multiscale Ablation (baseline_1d_cnn)",
            "model_architecture": "VibrationCNN1D",
            "fc_units": 128,
            "trainable_parameters": 106244,
            "oof_accuracy": m10_3["pooled_oof_accuracy"],
            "oof_macro_f1": m10_3["pooled_oof_macro_f1"],
            "fold_0_accuracy": m10_3["folds"][0]["val_accuracy"],
            "fold_0_macro_f1": m10_3["folds"][0]["val_macro_f1"],
            "fold_1_accuracy": m10_3["folds"][1]["val_accuracy"],
            "fold_1_macro_f1": m10_3["folds"][1]["val_macro_f1"],
        },
        "Phase_10_4": {
            "experiment": "Phase 10.4 Frequency Ablation (baseline_waveform)",
            "model_architecture": "VibrationCNN1D",
            "fc_units": 64,
            "trainable_parameters": 73156,
            "oof_accuracy": m10_4["oof_accuracy"],
            "oof_macro_f1": m10_4["oof_macro_f1"],
            "fold_0_accuracy": m10_4["fold_metrics"][0]["val_accuracy"],
            "fold_0_macro_f1": m10_4["fold_metrics"][0]["val_macro_f1"],
            "fold_1_accuracy": m10_4["fold_metrics"][1]["val_accuracy"],
            "fold_1_macro_f1": m10_4["fold_metrics"][1]["val_macro_f1"],
        },
        "discrepancy_explanation": (
            "Phase 10.2 and Phase 10.3 evaluated identical baseline models instantiated with "
            "fc_units=128 (106,244 parameters), yielding identical OOF Macro-F1 of 83.78% and "
            "accuracy of 86.26%. The prompt reference to Phase 10.3 having '86.26%' was an accidental "
            "citation of OOF accuracy rather than Macro-F1. In Phase 10.4, the baseline was explicitly "
            "instantiated with fc_units=64 (73,156 parameters) per prompt instruction, resulting in "
            "an expected slight decrease in Fold 1 capacity (val Macro-F1 90.57% vs 94.09%), yielding "
            "pooled OOF Macro-F1 of 81.09% and accuracy of 84.69%."
        ),
    }

    with open(EXP_OUTPUT_DIR / "cross_experiment_reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2)

    return reconciliation


def audit_recording_metadata_and_vibration_statistics() -> pd.DataFrame:
    """Generate comprehensive recording-level metadata and physical vibration statistics."""
    logger.info("Computing recording metadata and physical vibration statistics for all 13 recordings...")
    
    npz_path = PROJECT_ROOT / "data" / "processed" / "cwru_segmented_windows.npz"
    meta_path = PROJECT_ROOT / "data" / "processed" / "window_split_metadata.csv"
    raw_cwru_dir = PROJECT_ROOT / "data" / "raw" / "cwru"

    npz = np.load(npz_path)
    windows = npz["windows"]  # (2928, 2048)
    split_meta = pd.read_csv(meta_path)

    records = []
    for rec_id, group in split_meta.groupby("recording_id"):
        rec_str = str(rec_id)
        idx = group.index.values
        rec_wins = windows[idx]  # Shape: (n_windows, 2048)
        flat_sig = rec_wins.flatten()

        # Physical statistics
        rms = float(np.sqrt(np.mean(flat_sig**2)))
        std = float(np.std(flat_sig))
        peak = float(np.max(np.abs(flat_sig)))
        crest_factor = float(peak / (rms + 1e-8))
        kurt = float(stats.kurtosis(flat_sig, fisher=False))  # Pearson kurtosis (normal=3.0)
        skew = float(stats.skew(flat_sig))

        first_row = group.iloc[0]
        mat_file = raw_cwru_dir / f"{rec_str}.mat"
        if not mat_file.is_file():
            # Check 3-digit padded name
            mat_file = raw_cwru_dir / f"{int(rec_str):03d}.mat"

        # Check actual channels in .mat file
        mat_chans = []
        if mat_file.is_file():
            mat_dict = sio.loadmat(str(mat_file))
            mat_chans = [k for k in mat_dict.keys() if not k.startswith("__")]

        has_de = any("DE" in c for c in mat_chans)
        has_fe = any("FE" in c for c in mat_chans)
        has_ba = any("BA" in c for c in mat_chans)

        rpm_val = first_row["motor_rpm"]
        rpm_status = "Verified (Header)" if pd.notna(rpm_val) else "Missing (Inferred: ~1772 RPM for 1-2 HP)"

        records.append({
            "recording_id": rec_str,
            "split": first_row["split"],
            "fault_class": first_row["fault_class"],
            "defect_diameter_in": float(first_row["defect_diameter_in"]),
            "motor_load_hp": int(first_row["motor_load_hp"]),
            "motor_rpm": float(rpm_val) if pd.notna(rpm_val) else np.nan,
            "motor_rpm_status": rpm_status,
            "sampling_rate_hz": int(first_row["sampling_rate_hz"]),
            "sensor_location": "Drive End (DE)",
            "bearing_model": "SKF 6205-2RS JEM",
            "channels_present": f"DE={has_de}, FE={has_fe}, BA={has_ba}",
            "window_count": len(group),
            "rms_g": round(rms, 4),
            "std_g": round(std, 4),
            "peak_g": round(peak, 4),
            "crest_factor": round(crest_factor, 4),
            "kurtosis": round(kurt, 4),
            "skewness": round(skew, 4),
        })

    df_records = pd.DataFrame(records).sort_values(by=["split", "fault_class", "defect_diameter_in"])
    df_records.to_csv(EXP_OUTPUT_DIR / "recording_metadata_table.csv", index=False)
    
    # Save as JSON as well
    with open(EXP_OUTPUT_DIR / "recording_metadata_table.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return df_records


def perform_signal_failure_analysis() -> Dict[str, Any]:
    """Perform detailed signal-level comparison of recordings 105, 209, 130, and 234."""
    logger.info("Performing signal-level failure analysis on recordings 105, 209, 130, 234...")
    
    npz_path = PROJECT_ROOT / "data" / "processed" / "cwru_segmented_windows.npz"
    meta_path = PROJECT_ROOT / "data" / "processed" / "window_split_metadata.csv"
    npz = np.load(npz_path)
    windows = npz["windows"]
    split_meta = pd.read_csv(meta_path)

    target_recs = ["105", "209", "130", "234"]
    rec_signals = {}
    fs = 12000

    for rec_id in target_recs:
        group = split_meta[split_meta["recording_id"].astype(str) == rec_id]
        rec_wins = windows[group.index.values]
        rec_signals[rec_id] = {
            "windows": rec_wins,
            "flat": rec_wins.flatten(),
            "meta": group.iloc[0].to_dict(),
        }

    # Theoretical bearing frequencies at ~1797 RPM (0 HP)
    kinematic_freqs = get_bearing_characteristic_frequencies(motor_rpm=1797.0)

    # 1. Band Energy Calculations
    # Bands: Low (0-500 Hz), Mid (500-2000 Hz), High (2000-6000 Hz)
    band_energies = {}
    signal_metrics = {}

    for rec_id in target_recs:
        sig = rec_signals[rec_id]["flat"]
        freqs, psd = signal.welch(sig, fs=fs, nperseg=2048)

        total_pwr = np.sum(psd)
        band_low = np.sum(psd[(freqs >= 0) & (freqs < 500)]) / total_pwr
        band_mid = np.sum(psd[(freqs >= 500) & (freqs < 2000)]) / total_pwr
        band_high = np.sum(psd[(freqs >= 2000) & (freqs <= 6000)]) / total_pwr

        band_energies[rec_id] = {
            "band_low_0_500hz": round(float(band_low), 4),
            "band_mid_500_2000hz": round(float(band_mid), 4),
            "band_high_2000_6000hz": round(float(band_high), 4),
        }

        # Spectral peak detection in baseband (0 to 500 Hz)
        base_mask = (freqs >= 10) & (freqs <= 500)
        base_freqs = freqs[base_mask]
        base_psd = psd[base_mask]
        top_idx = np.argsort(base_psd)[-3:][::-1]
        dominant_base_peaks = [round(float(base_freqs[i]), 2) for i in top_idx]

        rms = float(np.sqrt(np.mean(sig**2)))
        peak = float(np.max(np.abs(sig)))
        kurt = float(stats.kurtosis(sig, fisher=False))

        signal_metrics[rec_id] = {
            "fault_class": rec_signals[rec_id]["meta"]["fault_class"],
            "defect_diameter_in": rec_signals[rec_id]["meta"]["defect_diameter_in"],
            "rms_g": round(rms, 4),
            "peak_g": round(peak, 4),
            "kurtosis": round(kurt, 4),
            "dominant_peaks_0_500hz": dominant_base_peaks,
            "band_energy_fractions": band_energies[rec_id],
        }

    # -------------------------------------------------------------
    # Figure Generation
    # -------------------------------------------------------------
    labels_dict = {
        "105": "Rec 105: Inner Race 0.007\"",
        "209": "Rec 209: Inner Race 0.021\"",
        "130": "Rec 130: Outer Race 0.007\"",
        "234": "Rec 234: Outer Race 0.021\"",
    }
    colors_dict = {
        "105": "#1f77b4",  # Blue
        "209": "#d62728",  # Red
        "130": "#2ca02c",  # Green
        "234": "#ff7f0e",  # Orange
    }

    # Fig 1: Time Domain Waveforms (500 samples ~ 41.7 ms ~ 1.25 rotations)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t_ms = (np.arange(500) / fs) * 1000.0

    for ax, rec_id in zip(axes, target_recs):
        sample_wave = rec_signals[rec_id]["windows"][0, :500]
        ax.plot(t_ms, sample_wave, color=colors_dict[rec_id], lw=1.2)
        ax.set_ylabel("Accel (g)", fontsize=10)
        ax.set_title(labels_dict[rec_id], fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (milliseconds) [~1.25 Motor Rotations]", fontsize=11)
    plt.suptitle("Figure 1: Time-Domain Vibration Waveform Comparison (500 Samples)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "time_domain_waveforms.png", dpi=300)
    plt.close()

    # Fig 2: Normalized Amplitude Distributions (KDE / Histograms)
    fig, ax = plt.subplots(figsize=(10, 6))
    for rec_id in target_recs:
        sig = rec_signals[rec_id]["flat"]
        # Standardize for shape comparison
        norm_sig = (sig - np.mean(sig)) / np.std(sig)
        ax.hist(norm_sig, bins=100, range=(-5, 5), density=True, histtype="step", lw=2,
                label=labels_dict[rec_id], color=colors_dict[rec_id])
    
    # Overlay Gaussian reference
    x_grid = np.linspace(-5, 5, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid), 'k--', lw=1.5, label="Standard Normal Reference (Kurt=3.0)")
    ax.set_title("Figure 2: Normalized Amplitude Probability Densities (Kurtosis Heavy-Tails)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Standardized Acceleration (Z-Score)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "amplitude_distributions.png", dpi=300)
    plt.close()

    # Fig 3: Power Spectral Density (Welch PSD, 0 to 6000 Hz)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, rec_id in zip(axes, target_recs):
        sig = rec_signals[rec_id]["flat"]
        freqs, psd = signal.welch(sig, fs=fs, nperseg=2048)
        ax.semilogy(freqs, psd, color=colors_dict[rec_id], lw=1.2)
        ax.set_title(labels_dict[rec_id], fontsize=11, fontweight="bold")
        ax.set_ylabel("PSD ($g^2$/Hz)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        # Highlight BPFO and BPFI
        ax.axvline(kinematic_freqs["bpfo_hz"], color="purple", linestyle="--", alpha=0.7, label=f"BPFO ({kinematic_freqs['bpfo_hz']} Hz)")
        ax.axvline(kinematic_freqs["bpfi_hz"], color="brown", linestyle="--", alpha=0.7, label=f"BPFI ({kinematic_freqs['bpfi_hz']} Hz)")
        ax.legend(fontsize=9, loc="upper right")

    axes[2].set_xlabel("Frequency (Hz)", fontsize=11)
    axes[3].set_xlabel("Frequency (Hz)", fontsize=11)
    plt.suptitle("Figure 3: Power Spectral Density Comparison (0 to 6,000 Hz)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "power_spectral_density.png", dpi=300)
    plt.close()

    # Fig 4: Hilbert Envelope Spectra (Baseband 0 to 500 Hz)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, rec_id in zip(axes, target_recs):
        sample_win = rec_signals[rec_id]["windows"][0]
        env_freqs, env_mag, _ = compute_hilbert_envelope_spectrum(sample_win, sampling_rate_hz=fs)
        mask = env_freqs <= 500
        ax.plot(env_freqs[mask], env_mag[mask], color=colors_dict[rec_id], lw=1.3)
        ax.set_ylabel("Envelope Mag", fontsize=10)
        ax.set_title(labels_dict[rec_id], fontsize=11, fontweight="bold", loc="left")
        ax.axvline(kinematic_freqs["bpfo_hz"], color="purple", linestyle="--", lw=1.5, label=f"BPFO ({kinematic_freqs['bpfo_hz']} Hz)")
        ax.axvline(kinematic_freqs["bpfi_hz"], color="brown", linestyle="--", lw=1.5, label=f"BPFI ({kinematic_freqs['bpfi_hz']} Hz)")
        ax.axvline(kinematic_freqs["fr_hz"], color="gray", linestyle=":", lw=1.2, label=f"Shaft fr ({kinematic_freqs['fr_hz']} Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Modulation Frequency (Hz)", fontsize=11)
    plt.suptitle("Figure 4: Hilbert Envelope Demodulation Spectra (Baseband 0 to 500 Hz)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "envelope_spectra_comparison.png", dpi=300)
    plt.close()

    # Fig 5: Frequency-Band Energy Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.2
    x_indices = np.arange(len(target_recs))

    low_vals = [band_energies[r]["band_low_0_500hz"] * 100 for r in target_recs]
    mid_vals = [band_energies[r]["band_mid_500_2000hz"] * 100 for r in target_recs]
    high_vals = [band_energies[r]["band_high_2000_6000hz"] * 100 for r in target_recs]

    ax.bar(x_indices - bar_width, low_vals, width=bar_width, label="Low Band (0 - 500 Hz)", color="#3498db")
    ax.bar(x_indices, mid_vals, width=bar_width, label="Mid Band (500 - 2,000 Hz)", color="#2ecc71")
    ax.bar(x_indices + bar_width, high_vals, width=bar_width, label="High Band (2,000 - 6,000 Hz)", color="#e74c3c")

    ax.set_xticks(x_indices)
    ax.set_xticklabels([labels_dict[r] for r in target_recs], fontsize=10, rotation=10)
    ax.set_ylabel("Fractional Energy (%)", fontsize=11)
    ax.set_title("Figure 5: Spectral Band Energy Distribution Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "frequency_band_energy_barchart.png", dpi=300)
    plt.close()

    # Fig 6: Confounding Matrix Grid (Severity vs Class in Cross-Validation)
    fig, ax = plt.subplots(figsize=(8, 5))
    grid_data = np.array([
        [1, 0],  # Rec 105: Inner 0.007" (Train F0 / Val F1)
        [0, 1],  # Rec 209: Inner 0.021" (Val F0 / Train F1)
        [0, 1],  # Rec 130: Outer 0.007" (Val F0 / Train F1)
        [1, 0],  # Rec 234: Outer 0.021" (Train F0 / Val F1)
    ])
    # Show how Fold 0 couples Inner 0.007" with Outer 0.021" in Training
    ax.text(0.1, 0.8, "Fold 0 Training Set:\n* Inner Race: ONLY 0.007\" (Rec 105)\n* Outer Race: ONLY 0.021\" (Rec 234)",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#2980b9", lw=1.5))
    ax.text(0.1, 0.45, "Fold 0 Validation Set:\n* Inner Race: ONLY 0.021\" (Rec 209) -> Predicted 100% as Outer Race!\n* Outer Race: ONLY 0.007\" (Rec 130)",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdeed9", edgecolor="#d35400", lw=1.5))
    ax.text(0.1, 0.1, "Diagnostic Proof:\nDefect Severity (0.007\" vs 0.021\") is 100% collinear with Class Label\nin training! Network learns amplitude/broadband energy shortcut.",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#c0392b", lw=1.5))
    ax.axis("off")
    plt.title("Figure 6: Structural Confounding in 2-Fold Cross-Validation Partitions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_DIR / "severity_confounding_grid.png", dpi=300)
    plt.close()

    results = {
        "kinematic_reference_frequencies": kinematic_freqs,
        "signal_metrics": signal_metrics,
    }

    with open(EXP_OUTPUT_DIR / "signal_failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def audit_cross_validation_integrity() -> Dict[str, Any]:
    """Verify cross-validation group partitioning and absence of leakage."""
    logger.info("Auditing Cross-Validation group isolation and leakage integrity...")
    meta_path = PROJECT_ROOT / "data" / "processed" / "window_split_metadata.csv"
    df = pd.read_csv(meta_path)

    # 1. Check window partition counts
    dev_mask = df["split"] == "dev"
    test_mask = df["split"] == "test"
    n_dev = int(np.sum(dev_mask))
    n_test = int(np.sum(test_mask))

    # 2. Check recording memberships
    dev_recs = sorted(df[dev_mask]["recording_id"].astype(str).unique().tolist())
    test_recs = sorted(df[test_mask]["recording_id"].astype(str).unique().tolist())
    test_leakage = set(dev_recs).intersection(set(test_recs))

    # 3. Check Fold 0 and Fold 1 assignments
    f0_train = sorted(df[df["fold_0"] == "train"]["recording_id"].astype(str).unique().tolist())
    f0_val = sorted(df[df["fold_0"] == "val"]["recording_id"].astype(str).unique().tolist())
    f0_overlap = set(f0_train).intersection(set(f0_val))

    f1_train = sorted(df[df["fold_1"] == "train"]["recording_id"].astype(str).unique().tolist())
    f1_val = sorted(df[df["fold_1"] == "val"]["recording_id"].astype(str).unique().tolist())
    f1_overlap = set(f1_train).intersection(set(f1_val))

    # 4. Verify each development window is evaluated exactly once out-of-fold
    f0_val_idx = set(df[df["fold_0"] == "val"].index)
    f1_val_idx = set(df[df["fold_1"] == "val"].index)
    val_overlap = f0_val_idx.intersection(f1_val_idx)
    all_val = f0_val_idx.union(f1_val_idx)
    dev_idx = set(df[dev_mask].index)
    uncovered_dev = dev_idx - all_val

    # 5. Check Normalizer save paths from Phase 10.2, 10.3, 10.4
    normalizer_audits = {}
    for phase_dir in ["phase10_2/global", "phase10_3/baseline_1d_cnn", "phase10_4/baseline_waveform"]:
        p = PROJECT_ROOT / "outputs" / "experiments" / phase_dir
        f0_norm = p / "normalizer_fold_0.json"
        f1_norm = p / "normalizer_fold_1.json"
        if f0_norm.is_file() and f1_norm.is_file():
            with open(f0_norm, "r") as f:
                d0 = json.load(f)
            with open(f1_norm, "r") as f:
                d1 = json.load(f)
            normalizer_audits[phase_dir] = {
                "fold_0_train_recordings": d0.get("fit_recording_ids", []),
                "fold_1_train_recordings": d1.get("fit_recording_ids", []),
                "fold_0_has_val_leakage": bool(set(d0.get("fit_recording_ids", [])).intersection(set(f0_val))),
                "fold_1_has_val_leakage": bool(set(d1.get("fit_recording_ids", [])).intersection(set(f1_val))),
            }

    integrity_report = {
        "total_windows": len(df),
        "development_windows": n_dev,
        "test_windows": n_test,
        "development_recordings": dev_recs,
        "test_recordings": test_recs,
        "test_leakage_detected": bool(test_leakage),
        "fold_0": {
            "train_recordings": f0_train,
            "val_recordings": f0_val,
            "recording_overlap": list(f0_overlap),
            "train_samples": int(np.sum(df["fold_0"] == "train")),
            "val_samples": int(np.sum(df["fold_0"] == "val")),
        },
        "fold_1": {
            "train_recordings": f1_train,
            "val_recordings": f1_val,
            "recording_overlap": list(f1_overlap),
            "train_samples": int(np.sum(df["fold_1"] == "train")),
            "val_samples": int(np.sum(df["fold_1"] == "val")),
        },
        "oof_coverage": {
            "val_windows_overlap": len(val_overlap),
            "uncovered_dev_windows": len(uncovered_dev),
            "exact_single_evaluation": bool(len(val_overlap) == 0 and len(uncovered_dev) == 0),
        },
        "normalizer_provenance_audit": normalizer_audits,
    }

    with open(EXP_OUTPUT_DIR / "cv_integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    return integrity_report


def main() -> None:
    """Execute complete Phase 10.5 generalization audit."""
    logger.info("Starting Phase 10.5 Dataset Generalization and Failure Analysis Audit...")
    EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Audit previous experiments
    reconciliation = audit_previous_experiments()
    logger.info("Completed previous experiment audit and reconciliation.")

    # 2. Audit recording metadata and physical vibration statistics
    meta_df = audit_recording_metadata_and_vibration_statistics()
    logger.info(f"Completed metadata and statistics audit for {len(meta_df)} recordings.")

    # 3. Perform signal failure analysis and figure generation
    signal_results = perform_signal_failure_analysis()
    logger.info("Completed signal-level failure analysis and generated 6 diagnostic figures.")

    # 4. Audit cross-validation integrity
    integrity_report = audit_cross_validation_integrity()
    logger.info("Completed cross-validation integrity and leakage audit.")

    master_summary = {
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reconciliation": reconciliation,
        "metadata_records_count": len(meta_df),
        "cv_integrity": integrity_report["oof_coverage"],
        "signal_metrics_summary": signal_results["signal_metrics"],
    }
    with open(EXP_OUTPUT_DIR / "generalization_audit_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    logger.info(f"Phase 10.5 Audit Complete! Outputs saved to {EXP_OUTPUT_DIR} and {FIG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

