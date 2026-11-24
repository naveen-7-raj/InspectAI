"""InspectAI - Task 2.3: Initial Vibration Waveform Exploration (EDA).

Loads authentic CWRU recordings one at a time using VibrationDataLoader,
extracts Drive-End (DE_time) acceleration signals, computes summary statistics,
and generates publication-quality waveform comparisons across healthy and fault states.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")  # Headless backend for automation and script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ingestion.loader import VibrationDataLoader
from src.utils.config_parser import load_config, get_resolved_paths

# Experiment specifications mapping CWRU test files to physical metadata
CWRU_FILE_CATALOG = {
    "97.mat": {"category": "Normal", "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 0, "nominal_fs": 48000},
    "98.mat": {"category": "Normal", "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 1, "nominal_fs": 48000},
    "99.mat": {"category": "Normal", "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 2, "nominal_fs": 48000},
    "100.mat": {"category": "Normal", "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 3, "nominal_fs": 48000},
    "118.mat": {"category": "Ball_Fault", "defect_type": "Ball", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "185.mat": {"category": "Ball_Fault", "defect_type": "Ball", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "222.mat": {"category": "Ball_Fault", "defect_type": "Ball", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
    "105.mat": {"category": "Inner_Race_Fault", "defect_type": "Inner_Race", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "169.mat": {"category": "Inner_Race_Fault", "defect_type": "Inner_Race", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "209.mat": {"category": "Inner_Race_Fault", "defect_type": "Inner_Race", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
    "130.mat": {"category": "Outer_Race_Fault", "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "197.mat": {"category": "Outer_Race_Fault", "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "234.mat": {"category": "Outer_Race_Fault", "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
}


def compute_signal_statistics(signal: np.ndarray) -> Dict[str, float]:
    """Calculate statistical indicators for a 1D vibration signal array."""
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    rms_val = float(np.sqrt(np.mean(signal**2)))
    peak_to_peak = float(max_val - min_val)
    crest_factor = float(np.max(np.abs(signal)) / rms_val) if rms_val > 0 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "rms": rms_val,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
    }


def run_waveform_eda() -> pd.DataFrame:
    """Execute Task 2.3 exploratory waveform analysis and persist artifacts."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    raw_dir = resolved_paths["raw_data_dir"] / "cwru"
    figures_dir = resolved_paths["figures_dir"]
    reports_dir = resolved_paths["reports_dir"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_fs = config["dataset"]["sampling_rate_hz"]
    loader = VibrationDataLoader(raw_dir)

    print("=================================================================")
    print("InspectAI - Task 2.3: Vibration Waveform Exploration")
    print(f"Data Source Directory: {raw_dir}")
    print(f"Project Config Default Sampling Rate: {config_fs} Hz")
    print("=================================================================\n")

    stats_records: List[Dict[str, Any]] = []
    # Dictionary to hold the 4 key representative recordings for plotting
    representative_files = {
        "Normal": "97.mat",
        "Ball_Fault": "118.mat",
        "Inner_Race_Fault": "105.mat",
        "Outer_Race_Fault": "130.mat",
    }
    plot_segments: Dict[str, Dict[str, Any]] = {}

    # Load one file at a time to preserve RAM
    for fname, meta in CWRU_FILE_CATALOG.items():
        print(f"Processing {fname} ({meta['category']})...")
        record = loader.load_file(fname)
        de_signal = record.channels["DE_time"]

        nominal_fs = meta["nominal_fs"]
        sample_count = len(de_signal)
        duration_sec = sample_count / nominal_fs

        # Compute summary statistics
        stats = compute_signal_statistics(de_signal)

        record_info = {
            "file_name": fname,
            "category": meta["category"],
            "defect_type": meta["defect_type"],
            "defect_diameter_in": meta["defect_diameter_in"],
            "motor_load_hp": meta["load_hp"],
            "rpm": record.rpm,
            "nominal_sampling_rate_hz": nominal_fs,
            "total_samples": sample_count,
            "duration_seconds": round(duration_sec, 3),
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "rms": stats["rms"],
            "peak_to_peak": stats["peak_to_peak"],
            "crest_factor": stats["crest_factor"],
        }
        stats_records.append(record_info)

        # Retain short time window for the 4 representative files
        for cat_name, rep_fname in representative_files.items():
            if fname == rep_fname:
                # Time window of 0.05 seconds (50 ms)
                window_duration_sec = 0.05
                n_samples_window = int(window_duration_sec * nominal_fs)
                segment = de_signal[:n_samples_window]
                t_vec = np.linspace(0.0, window_duration_sec * 1000.0, n_samples_window, endpoint=False)  # in ms
                plot_segments[cat_name] = {
                    "file_name": fname,
                    "time_ms": t_vec,
                    "signal_g": segment,
                    "fs": nominal_fs,
                    "defect_diameter_in": meta["defect_diameter_in"],
                    "rpm": record.rpm,
                }

    # 1. Save summary statistics CSV
    summary_df = pd.DataFrame(stats_records)
    csv_out_path = reports_dir / "waveform_eda_summary.csv"
    summary_df.to_csv(csv_out_path, index=False)
    print(f"\n[Artifact Created] Summary statistics saved to: {csv_out_path.resolve()}")

    # 2. Plot: 4-Subplot Stacked Comparison
    plot_four_class_subplots(plot_segments, figures_dir / "waveform_comparison_subplots.png")

    # 3. Plot: Overlay Comparison (Normal vs Faults)
    plot_waveform_overlay(plot_segments, figures_dir / "waveform_comparison_overlay.png")

    # 4. Plot: Statistical Energy Comparison across all 13 recordings
    plot_all_recordings_stats(summary_df, figures_dir / "waveform_statistics_summary.png")

    print("\nEDA Execution Complete. All figures and reports generated successfully.")
    return summary_df


def plot_four_class_subplots(plot_segments: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Generate 4-subplot stacked visualization comparing Normal vs 3 fault types."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=False)
    fig.suptitle(
        "InspectAI - Drive-End (DE) Vibration Waveforms (50 ms Physical Window)\n"
        "Comparison of Healthy Baseline vs. 0.007\" Bearing Faults (CWRU Benchmark)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plot_configs = [
        ("Normal", "#2ca02c", "Normal Baseline (97.mat, 0 HP, 1797 RPM, fs=48 kHz)"),
        ("Ball_Fault", "#ff7f0e", "Ball Fault 0.007\" (118.mat, 0 HP, 1796 RPM, fs=12 kHz)"),
        ("Inner_Race_Fault", "#d62728", "Inner Race Fault 0.007\" (105.mat, 0 HP, 1797 RPM, fs=12 kHz)"),
        ("Outer_Race_Fault", "#9467bd", "Outer Race Fault 0.007\" @ 6:00 (130.mat, 0 HP, 1796 RPM, fs=12 kHz)"),
    ]

    for ax, (cat, color, label_text) in zip(axes, plot_configs):
        data = plot_segments[cat]
        t = data["time_ms"]
        sig = data["signal_g"]
        rms = np.sqrt(np.mean(sig**2))
        peak = np.max(np.abs(sig))

        ax.plot(t, sig, color=color, linewidth=1.1, alpha=0.9, label=label_text)
        ax.set_ylabel("Radial Accel (g)", fontsize=10, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Annotate peak and RMS
        stats_txt = f"RMS: {rms:.3f} g | Peak: {peak:.3f} g"
        ax.text(
            0.015,
            0.85,
            stats_txt,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    axes[-1].set_xlabel("Time (milliseconds)", fontsize=11, fontweight="semibold")
    axes[-1].set_xlim(0, 50.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Subplots figure saved to: {save_path.resolve()}")


def plot_waveform_overlay(plot_segments: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Generate direct waveform overlay showing amplitude contrast between healthy and fault states."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot normal baseline as background reference
    norm_data = plot_segments["Normal"]
    ax.plot(
        norm_data["time_ms"],
        norm_data["signal_g"],
        color="#2ca02c",
        linewidth=1.2,
        alpha=0.85,
        label=f"Normal (97.mat, RMS={np.sqrt(np.mean(norm_data['signal_g']**2)):.3f} g)",
    )

    # Overlay Inner Race Fault
    ir_data = plot_segments["Inner_Race_Fault"]
    ax.plot(
        ir_data["time_ms"],
        ir_data["signal_g"],
        color="#d62728",
        linewidth=1.0,
        alpha=0.8,
        label=f"Inner Race Fault 0.007\" (105.mat, RMS={np.sqrt(np.mean(ir_data['signal_g']**2)):.3f} g)",
    )

    # Overlay Outer Race Fault
    or_data = plot_segments["Outer_Race_Fault"]
    ax.plot(
        or_data["time_ms"],
        or_data["signal_g"],
        color="#9467bd",
        linewidth=0.9,
        alpha=0.75,
        label=f"Outer Race Fault 0.007\" (130.mat, RMS={np.sqrt(np.mean(or_data['signal_g']**2)):.3f} g)",
    )

    ax.set_title(
        "InspectAI - Vibration Amplitude Overlay: Healthy Baseline vs. Localized Fault Impacts",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (milliseconds)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Radial Acceleration (g)", fontsize=11, fontweight="semibold")
    ax.set_xlim(0, 50.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Overlay figure saved to: {save_path.resolve()}")


def plot_all_recordings_stats(df: pd.DataFrame, save_path: Path) -> None:
    """Plot summary comparison of RMS and Standard Deviation across all 13 recordings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color mapping by category
    palette = {
        "Normal": "#2ca02c",
        "Ball_Fault": "#ff7f0e",
        "Inner_Race_Fault": "#d62728",
        "Outer_Race_Fault": "#9467bd",
    }
    bar_colors = [palette.get(c, "#1f77b4") for c in df["category"]]

    x_labels = [f"{row['file_name']}\n({row['category'][:6]})" for _, row in df.iterrows()]
    x_indices = np.arange(len(df))

    # RMS Bar Chart
    ax1.bar(x_indices, df["rms"], color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title("Root Mean Square (RMS) Acceleration per Recording", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMS (g)", fontsize=10, fontweight="semibold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Peak-to-Peak Bar Chart
    ax2.bar(x_indices, df["peak_to_peak"], color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title("Peak-to-Peak Amplitude per Recording", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Peak-to-Peak (g)", fontsize=10, fontweight="semibold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in palette.items()]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Statistics summary figure saved to: {save_path.resolve()}")


if __name__ == "__main__":
    run_waveform_eda()
