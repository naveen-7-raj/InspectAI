"""InspectAI - Task 3.1: Preprocessing & Sampling Rate Normalization Validation.

Compares raw vibration waveforms vs. processed signals (sampling-rate normalization,
DC offset removal, and zero-phase Butterworth filtering) on representative recordings.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ingestion.loader import VibrationDataLoader
from src.preprocessing.signal_cleaner import SignalCleaner
from src.utils.config_parser import load_config, get_resolved_paths


def run_preprocessing_validation() -> pd.DataFrame:
    """Execute validation of signal cleaning pipeline on representative recordings."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    raw_dir = resolved_paths["raw_data_dir"] / "cwru"
    figures_dir = resolved_paths["figures_dir"]
    reports_dir = resolved_paths["reports_dir"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg = config["preprocessing"]
    cleaner = SignalCleaner(
        target_sampling_rate=prep_cfg["target_sampling_rate_hz"],
        resample_method=prep_cfg["resample_method"],
        remove_dc=prep_cfg["remove_dc"],
        filter_enabled=prep_cfg["filtering"]["enabled"],
        filter_type=prep_cfg["filtering"]["type"],
        low_cutoff_hz=prep_cfg["filtering"]["low_cutoff_hz"],
        high_cutoff_hz=prep_cfg["filtering"]["high_cutoff_hz"],
        filter_order=prep_cfg["filtering"]["order"],
    )

    loader = VibrationDataLoader(raw_dir)

    # Representative recordings: 97.mat (Normal 48 kHz), 105.mat (Inner Race 12 kHz), 130.mat (Outer Race 12 kHz)
    eval_cases = [
        {"filename": "97.mat", "category": "Normal", "orig_fs": 48000, "label": "Normal Baseline (97.mat)"},
        {"filename": "105.mat", "category": "Inner_Race_Fault", "orig_fs": 12000, "label": "Inner Race Fault 0.007\" (105.mat)"},
        {"filename": "130.mat", "category": "Outer_Race_Fault", "orig_fs": 12000, "label": "Outer Race Fault 0.007\" (130.mat)"},
    ]

    print("=================================================================")
    print("InspectAI - Task 3.1: Preprocessing Validation")
    print(f"Target Sampling Rate: {cleaner.target_sampling_rate} Hz")
    print(f"DC Removal: {cleaner.remove_dc}")
    print(f"Filter: {cleaner.filter_type} [{cleaner.low_cutoff_hz} Hz - {cleaner.high_cutoff_hz} Hz] (Order {cleaner.filter_order})")
    print("=================================================================\n")

    records_summary: List[Dict[str, Any]] = []
    plot_data: Dict[str, Dict[str, Any]] = {}

    for item in eval_cases:
        fname = item["filename"]
        orig_fs = item["orig_fs"]
        print(f"Processing {fname} (Nominal fs={orig_fs} Hz)...")

        rec = loader.load_file(fname)
        raw_signal = rec.channels["DE_time"]

        processed = cleaner.clean_signal(raw_signal, orig_fs=orig_fs, metadata={"filename": fname})

        raw_mean = float(np.mean(raw_signal))
        proc_mean = float(np.mean(processed.signal))
        raw_rms = float(np.sqrt(np.mean(raw_signal**2)))
        proc_rms = float(np.sqrt(np.mean(processed.signal**2)))
        raw_peak = float(np.max(np.abs(raw_signal)))
        proc_peak = float(np.max(np.abs(processed.signal)))

        summary_row = {
            "filename": fname,
            "category": item["category"],
            "original_fs_hz": orig_fs,
            "target_fs_hz": processed.target_fs,
            "original_sample_count": len(raw_signal),
            "processed_sample_count": len(processed.signal),
            "resampling_applied": processed.applied_resampling,
            "raw_mean_g": raw_mean,
            "processed_mean_g": proc_mean,
            "raw_rms_g": raw_rms,
            "processed_rms_g": proc_rms,
            "raw_peak_g": raw_peak,
            "processed_peak_g": proc_peak,
            "filtering_applied": processed.applied_filtering,
        }
        records_summary.append(summary_row)

        # Store 50 ms window for visualization
        win_sec = 0.05
        raw_n = int(win_sec * orig_fs)
        proc_n = int(win_sec * processed.target_fs)

        plot_data[fname] = {
            "label": item["label"],
            "raw_time_ms": np.linspace(0, win_sec * 1000, raw_n, endpoint=False),
            "raw_sig": raw_signal[:raw_n],
            "proc_time_ms": np.linspace(0, win_sec * 1000, proc_n, endpoint=False),
            "proc_sig": processed.signal[:proc_n],
            "orig_fs": orig_fs,
            "target_fs": processed.target_fs,
        }

    # Save CSV summary
    summary_df = pd.DataFrame(records_summary)
    csv_out = reports_dir / "preprocessing_validation_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\n[Artifact Created] Preprocessing report: {csv_out.resolve()}")

    # Generate before/after comparison plots
    plot_path = figures_dir / "preprocessing_comparison.png"
    plot_preprocessing_comparison(plot_data, plot_path)

    return summary_df


def plot_preprocessing_comparison(plot_data: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Generate side-by-side time-domain and spectral comparison of raw vs. cleaned signals."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    fig.suptitle(
        "InspectAI - Task 3.1: Raw vs. Preprocessed Signals (50 ms Physical Window)\n"
        "Sampling Rate Normalization (12 kHz) + DC Bias Removal + Zero-Phase Butterworth Bandpass (20 - 5000 Hz)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    filenames = ["97.mat", "105.mat", "130.mat"]

    for idx, fname in enumerate(filenames):
        item = plot_data[fname]
        raw_t = item["raw_time_ms"]
        raw_sig = item["raw_sig"]
        proc_t = item["proc_time_ms"]
        proc_sig = item["proc_sig"]

        # Left Column: Raw Signals
        ax_raw = axes[idx, 0]
        ax_raw.plot(raw_t, raw_sig, color="#1f77b4", linewidth=1.1, alpha=0.9)
        ax_raw.set_title(f"Raw: {item['label']} (fs = {item['orig_fs']:,} Hz)", fontsize=10, fontweight="bold")
        ax_raw.set_ylabel("Accel (g)", fontsize=10)
        ax_raw.grid(True, linestyle="--", alpha=0.5)

        raw_rms = np.sqrt(np.mean(raw_sig**2))
        raw_mean = np.mean(raw_sig)
        ax_raw.text(
            0.02, 0.85, f"Mean: {raw_mean:+.4f} g | RMS: {raw_rms:.4f} g",
            transform=ax_raw.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

        # Right Column: Preprocessed Signals
        ax_proc = axes[idx, 1]
        ax_proc.plot(proc_t, proc_sig, color="#2ca02c" if "Normal" in item["label"] else "#d62728", linewidth=1.1, alpha=0.9)
        ax_proc.set_title(f"Cleaned: {item['label']} (fs = {item['target_fs']:,} Hz, DC Removed, Filtered)", fontsize=10, fontweight="bold")
        ax_proc.set_ylabel("Accel (g)", fontsize=10)
        ax_proc.grid(True, linestyle="--", alpha=0.5)

        proc_rms = np.sqrt(np.mean(proc_sig**2))
        proc_mean = np.mean(proc_sig)
        ax_proc.text(
            0.02, 0.85, f"Mean: {proc_mean:+.4f} g | RMS: {proc_rms:.4f} g",
            transform=ax_proc.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    axes[-1, 0].set_xlabel("Time (milliseconds)", fontsize=10, fontweight="semibold")
    axes[-1, 1].set_xlabel("Time (milliseconds)", fontsize=10, fontweight="semibold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Preprocessing figure: {save_path.resolve()}")


if __name__ == "__main__":
    run_preprocessing_validation()

