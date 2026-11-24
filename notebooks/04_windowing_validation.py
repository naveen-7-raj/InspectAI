"""InspectAI - Task 3.2: Sliding Window Segmentation Validation.

Executes dataset segmentation across the 13 authentic CWRU recordings, exports
the processed dataset, and generates validation plots and summary reports.
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
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.ingestion.loader import VibrationDataLoader
from src.preprocessing.signal_cleaner import SignalCleaner
from src.preprocessing.windowing import WindowSegmenter, CWRU_CATALOG
from src.utils.config_parser import load_config, get_resolved_paths


def run_windowing_validation() -> Dict[str, Any]:
    """Execute complete segmentation pipeline, generate reports and visualization figures."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    figures_dir = resolved_paths["figures_dir"]
    reports_dir = resolved_paths["reports_dir"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("InspectAI - Task 3.2: Sliding Window Segmentation Validation")
    print(f"Window Size: {config['windowing']['window_size']} samples")
    print(f"Step Size: {config['windowing']['step_size']} samples")
    print(f"Overlap Ratio: {config['windowing']['overlap_ratio'] * 100:.1f}%")
    print(f"Sampling Rate: {config['windowing']['sampling_rate_hz']} Hz")
    print("=================================================================\n")

    # 1. Execute dataset segmentation and persist arrays to data/processed/
    results = WindowSegmenter.segment_dataset(config=config, save_dataset=True)

    print(f"Total Windows Generated: {results['total_windows']:,}")
    print(f"Array Shape: {results['window_shape']}")
    print(f"Class Distribution: {results['class_counts']}")

    # 2. Save windowing summary report CSV
    summary_df = pd.DataFrame(results["recording_summaries"])
    summary_csv_path = reports_dir / "windowing_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Artifact Created] Summary report: {summary_csv_path.resolve()}")

    # 3. Plot: Demonstration of 75% Overlapping Windows on Inner Race Fault (105.mat)
    demo_fig_path = figures_dir / "sliding_window_demonstration.png"
    plot_sliding_window_demonstration(config, demo_fig_path)

    # 4. Plot: Window Distribution Across Classes & Severities
    dist_fig_path = figures_dir / "window_distribution_per_class.png"
    plot_window_class_distribution(results["class_counts"], summary_df, dist_fig_path)

    print("\nWindowing Validation Complete. All datasets and figures generated.")
    return results


def plot_sliding_window_demonstration(config: Dict[str, Any], save_path: Path) -> None:
    """Generate visual demonstration of overlapping sliding windows on a vibration waveform."""
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"
    loader = VibrationDataLoader(raw_dir)
    rec = loader.load_file("105.mat")

    cleaner = SignalCleaner(target_sampling_rate=12000, remove_dc=True, filter_enabled=True)
    cleaned = cleaner.clean_signal(rec.channels["DE_time"], orig_fs=12000)

    fs = 12000
    win_size = 2048
    step_size = 512
    overlap_samples = win_size - step_size  # 1536 samples (75%)

    # Display first 4000 samples (~0.33 seconds) showing first 4 overlapping windows
    display_len = 4000
    t_ms = np.linspace(0, (display_len / fs) * 1000, display_len, endpoint=False)
    sig_slice = cleaned.signal[:display_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1.2]}, sharex=True)
    fig.suptitle(
        "InspectAI - Sliding Window Segmentation Architecture\n"
        "Window Size = 2,048 Samples (~170.7 ms) | Step = 512 Samples (~42.7 ms) | Overlap = 75% (1,536 Samples)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Top: Vibration signal
    ax1.plot(t_ms, sig_slice, color="#d62728", linewidth=1.0, alpha=0.9, label="DE Vibration Signal (105.mat - Inner Race 0.007\")")
    ax1.set_ylabel("Radial Accel (g)", fontsize=10, fontweight="semibold")
    ax1.set_title("Continuous Cleaned Vibration Waveform with Extracting Windows", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Bottom: Staggered window coverage spans
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    num_demo_windows = 4

    for w_idx in range(num_demo_windows):
        start_samp = w_idx * step_size
        end_samp = start_samp + win_size
        start_t = (start_samp / fs) * 1000
        end_t = (end_samp / fs) * 1000
        y_pos = num_demo_windows - 1 - w_idx

        # Plot rectangle span in bottom axis
        rect = patches.Rectangle(
            (start_t, y_pos - 0.3),
            end_t - start_t,
            0.6,
            linewidth=1.5,
            edgecolor=colors[w_idx % len(colors)],
            facecolor=colors[w_idx % len(colors)],
            alpha=0.35,
        )
        ax2.add_patch(rect)
        ax2.text(
            start_t + (end_t - start_t) / 2,
            y_pos,
            f"Window {w_idx}: [{start_samp} : {end_samp}] ({end_t - start_t:.1f} ms)",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

        # Highlight span on top plot
        ax1.axvspan(start_t, end_t, color=colors[w_idx % len(colors)], alpha=0.08)
        ax1.axvline(start_t, color=colors[w_idx % len(colors)], linestyle=":", alpha=0.6)

    ax2.set_ylim(-0.8, num_demo_windows - 0.2)
    ax2.set_yticks(range(num_demo_windows))
    ax2.set_yticklabels([f"Window {num_demo_windows - 1 - i}" for i in range(num_demo_windows)], fontsize=9)
    ax2.set_xlabel("Time (milliseconds)", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Segmentation Steps", fontsize=10, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xlim(0, (display_len / fs) * 1000)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Demonstration figure: {save_path.resolve()}")


def plot_window_class_distribution(class_counts: Dict[str, int], summary_df: pd.DataFrame, save_path: Path) -> None:
    """Plot bar chart of extracted window distribution across all 4 machine health classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    palette = {
        "Normal": "#2ca02c",
        "Ball_Fault": "#ff7f0e",
        "Inner_Race_Fault": "#d62728",
        "Outer_Race_Fault": "#9467bd",
    }

    # Left: Total windows per class
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    bar_colors = [palette.get(c, "#333333") for c in classes]

    bars = ax1.bar(classes, counts, color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax1.set_title("Total Windows per Machine Health Class (N_total = 2,928)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Complete Windows", fontsize=10, fontweight="semibold")
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{int(h):,}", ha="center", fontsize=9, fontweight="bold")

    # Right: Windows per recording
    rec_colors = [palette.get(c, "#333333") for c in summary_df["category"]]
    x_pos = np.arange(len(summary_df))
    bars2 = ax2.bar(x_pos, summary_df["num_windows"], color=rec_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.set_title("Windows Generated per Source Recording", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Windows", fontsize=10, fontweight="semibold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary_df["file_name"], rotation=45, ha="right", fontsize=8.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Artifact Created] Distribution figure: {save_path.resolve()}")


if __name__ == "__main__":
    run_windowing_validation()
