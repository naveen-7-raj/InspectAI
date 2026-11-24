"""Phase 5: 1D Grad-CAM Explainability Generation Script for InspectAI.

Generates 1D Grad-CAM attributions for representative test vibration recordings
using the frozen InspectAI final model and normalizer.

Produces:
- Single-window waveform & Grad-CAM heatmaps for:
  * 97_w0000 (Normal - True Positive)
  * 185_w0000 (Ball Fault - True Positive)
  * 169_w0000 (Inner Race Fault - Misclassified as Ball Fault):
      - Attribution for Predicted Class (Ball Fault)
      - Attribution for True Class (Inner Race Fault)
  * 197_w0000 (Outer Race Fault - Misclassified as Ball Fault):
      - Attribution for Predicted Class (Ball Fault)
      - Attribution for True Class (Outer Race Fault)
- Multi-case comparison figures:
  * gradcam_all_classes_comparison.png
  * gradcam_misclassification_comparison.png
- Quantitative explainability metrics exported to outputs/reports/gradcam_analysis_summary.json.

Strict Integrity Rules:
- The model checkpoint is loaded in evaluation mode only.
- No model weights or normalizer parameters are modified.
- No retraining or tuning is performed.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.explainability.explainer import VibrationExplainer, DEFAULT_CLASS_NAMES
from src.models.cnn_1d import VibrationCNN1D
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import get_resolved_paths, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.GradCAMGen")


def plot_single_explanation(
    explanation: Dict[str, Any],
    metadata_row: pd.Series,
    save_path: Path,
    title_suffix: str = "",
) -> None:
    """Render aligned 2-panel figure: Raw Vibration Waveform (top) and 1D Grad-CAM Heatmap (bottom)."""
    waveform_raw = explanation["waveform_raw"]
    heatmap = explanation["gradcam_heatmap"]
    timestamps = explanation["timestamps_sec"] * 1000.0  # Convert to ms
    fs = explanation["sampling_rate_hz"]

    pred_name = explanation["predicted_class_name"]
    target_name = explanation["target_class_name"]
    true_class = metadata_row["fault_class"]
    window_id = metadata_row["window_id"]
    source_file = metadata_row["source_file"]
    pred_prob = explanation["probabilities"][explanation["predicted_class_id"]]
    target_prob = explanation["probabilities"][explanation["target_class_id"]]

    fig, (ax_wave, ax_cam) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]}
    )

    # 1. Waveform Plot
    ax_wave.plot(timestamps, waveform_raw, color="#1f77b4", linewidth=0.8, alpha=0.9, label="Drive End Acceleration (g)")
    ax_wave.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)

    # Highlight important regions (> 0.6 attribution) on waveform
    for reg in explanation["important_regions"]:
        start_ms = reg["start_time_sec"] * 1000.0
        end_ms = reg["end_time_sec"] * 1000.0
        ax_wave.axvspan(start_ms, end_ms, color="#ff7f0e", alpha=0.25, label="High Attribution Region" if reg == explanation["important_regions"][0] else "")

    ax_wave.set_ylabel("Amplitude (g)", fontsize=11, fontweight="bold")
    ax_wave.set_title(
        f"InspectAI 1D Grad-CAM: {window_id} ({source_file}) {title_suffix}\n"
        f"True Class: {true_class} | Predicted: {pred_name} (Uncalibrated Softmax: {pred_prob:.1%}) | Explaining: {target_name} ({target_prob:.1%})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_wave.grid(True, linestyle=":", alpha=0.6)
    ax_wave.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 2. Grad-CAM Attribution Heatmap
    # Display as filled curve and colored gradient
    ax_cam.plot(timestamps, heatmap, color="#d62728", linewidth=1.2, label=f"Grad-CAM Attribution (Layer: {explanation['target_layer_name']})")
    ax_cam.fill_between(timestamps, 0, heatmap, color="#d62728", alpha=0.3)
    ax_cam.axhline(0.6, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Salience Threshold (0.6)")

    ax_cam.set_ylabel("CAM Attribution", fontsize=11, fontweight="bold")
    ax_cam.set_xlabel("Time (milliseconds)", fontsize=11, fontweight="bold")
    ax_cam.set_ylim(-0.05, 1.05)
    ax_cam.grid(True, linestyle=":", alpha=0.6)
    ax_cam.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Annotation box for scientific disclaimer
    disclaimer = "Attribution reflects CNN activation sensitivity, not direct physical proof of defect dynamics."
    fig.text(0.5, 0.01, disclaimer, ha="center", fontsize=8.5, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved explanation plot: {save_path.name}")


def plot_misclassification_comparison(
    exp_inner_pred: Dict[str, Any],
    exp_inner_true: Dict[str, Any],
    meta_inner: pd.Series,
    exp_outer_pred: Dict[str, Any],
    exp_outer_true: Dict[str, Any],
    meta_outer: pd.Series,
    save_path: Path,
) -> None:
    """Render 4-panel comparison contrasting Predicted (Ball Fault) vs True Class Grad-CAM attributions."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    timestamps = exp_inner_pred["timestamps_sec"] * 1000.0

    cases = [
        (
            exp_inner_pred,
            meta_inner,
            "169_w0000 (0.014\" Inner Race Fault)",
            "Predicted: Ball Fault (Softmax: 96.1%) - Spurious Attribution",
            "#d62728",
        ),
        (
            exp_inner_true,
            meta_inner,
            "169_w0000 (0.014\" Inner Race Fault)",
            "True Class: Inner Race Fault (Softmax: 0.8%) - True Class Attribution",
            "#1f77b4",
        ),
        (
            exp_outer_pred,
            meta_outer,
            "197_w0000 (0.014\" Outer Race Fault)",
            "Predicted: Ball Fault (Softmax: 99.2%) - Spurious Attribution",
            "#d62728",
        ),
        (
            exp_outer_true,
            meta_outer,
            "197_w0000 (0.014\" Outer Race Fault)",
            "True Class: Outer Race Fault (Softmax: 0.1%) - True Class Attribution",
            "#2ca02c",
        ),
    ]

    for idx, (exp, meta, case_label, sub_label, color) in enumerate(cases):
        ax = axes[idx]
        cam = exp["gradcam_heatmap"]
        ax.plot(timestamps, cam, color=color, linewidth=1.2, label=sub_label)
        ax.fill_between(timestamps, 0, cam, color=color, alpha=0.25)
        ax.axhline(0.6, color="orange", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("CAM Score", fontsize=9.5, fontweight="bold")
        ax.set_title(f"{case_label} | {sub_label}", fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    axes[-1].set_xlabel("Time (milliseconds)", fontsize=11, fontweight="bold")
    fig.suptitle(
        "InspectAI Diagnostic Analysis: Predicted (Ball Fault) vs True Class Attributions\n"
        "Demonstrating Model Confusion on Intermediate Defect Sizes (0.014\" vs 0.007\"/0.021\" Training Set)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    disclaimer = "Scientific Limitation: 1D Grad-CAM explains convolutional feature weighting; it does not replace kinematic physics or provide RUL prognosis."
    fig.text(0.5, 0.005, disclaimer, ha="center", fontsize=8.5, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved misclassification comparison plot: {save_path.name}")


def plot_all_classes_comparison(
    explanations: List[Tuple[Dict[str, Any], pd.Series, str]],
    save_path: Path,
) -> None:
    """Render comprehensive 4-class multi-panel figure comparing waveform and CAM across Normal, Ball, Inner, Outer."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

    for i, (exp, meta, desc) in enumerate(explanations):
        ax_wave = axes[i, 0]
        ax_cam = axes[i, 1]
        timestamps = exp["timestamps_sec"] * 1000.0
        wave = exp["waveform_raw"]
        cam = exp["gradcam_heatmap"]

        # Left: Waveform
        ax_wave.plot(timestamps, wave, color="#1f77b4", linewidth=0.7)
        ax_wave.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_wave.set_ylabel("Amp (g)", fontsize=9, fontweight="bold")
        ax_wave.set_title(f"{meta['window_id']} ({meta['fault_class']}) - {desc}", fontsize=10, fontweight="bold")
        ax_wave.grid(True, linestyle=":", alpha=0.5)

        # Right: Grad-CAM
        ax_cam.plot(timestamps, cam, color="#d62728", linewidth=1.0)
        ax_cam.fill_between(timestamps, 0, cam, color="#d62728", alpha=0.3)
        ax_cam.axhline(0.6, color="orange", linestyle="--", linewidth=0.7)
        ax_cam.set_ylim(-0.05, 1.05)
        ax_cam.set_ylabel("CAM", fontsize=9, fontweight="bold")
        ax_cam.set_title(f"Target: {exp['target_class_name']} | Softmax: {exp['probabilities'][exp['target_class_id']]:.1%}", fontsize=10)
        ax_cam.grid(True, linestyle=":", alpha=0.5)

    axes[-1, 0].set_xlabel("Time (milliseconds)", fontsize=10, fontweight="bold")
    axes[-1, 1].set_xlabel("Time (milliseconds)", fontsize=10, fontweight="bold")

    fig.suptitle("InspectAI 1D Grad-CAM: Four-Class Vibration Waveform and Attribution Comparison", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved all-classes comparison plot: {save_path.name}")


def main():
    config = load_config()
    paths = get_resolved_paths(config)
    reports_dir = paths["reports_dir"]
    figures_dir = paths["figures_dir"]
    gradcam_figures_dir = figures_dir / "gradcam"
    gradcam_figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = paths["model_checkpoint_dir"] / "inspectai_final_cnn.pth"
    norm_path = paths["model_checkpoint_dir"] / "inspectai_final_normalizer.json"
    pred_path = reports_dir / "final_test_predictions.csv"
    npz_path = paths["processed_data_dir"] / "cwru_segmented_windows.npz"
    meta_path = paths["processed_data_dir"] / "window_split_metadata.csv"

    # 1. Load frozen model & normalizer
    logger.info("Loading frozen final model checkpoint and normalizer...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = VibrationCNN1D.from_config(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = VibrationNormalizer.load(norm_path)
    explainer = VibrationExplainer(model=model, normalizer=normalizer, device="cpu")

    # 2. Load dataset and prediction tables
    pred_df = pd.read_csv(pred_path)
    meta_df = pd.read_csv(meta_path)
    npz = np.load(npz_path)
    all_windows = npz["windows"]

    # Target representative windows
    target_window_ids = ["97_w0000", "185_w0000", "169_w0000", "197_w0000"]
    logger.info(f"Locating representative test windows: {target_window_ids}")

    analysis_records = []

    # Map window_id to row index in cwru_segmented_windows.npz
    meta_df["global_index"] = meta_df.index
    target_meta = meta_df[meta_df["window_id"].isin(target_window_ids)].copy()

    computed_explanations = {}

    for _, row in target_meta.iterrows():
        wid = row["window_id"]
        g_idx = row["global_index"]
        raw_waveform = all_windows[g_idx]

        # Case 1: Normal (97_w0000)
        if wid == "97_w0000":
            exp = explainer.explain_window(raw_waveform, target_class=None, is_normalized=False)
            computed_explanations["97_normal"] = (exp, row)
            plot_single_explanation(
                exp, row, gradcam_figures_dir / "gradcam_97_normal.png", title_suffix="(True Positive Normal)"
            )
            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": exp["target_class_name"],
                "predicted_class": exp["predicted_class_name"],
                "uncalibrated_prob": exp["probabilities"][exp["predicted_class_id"]],
                "num_important_regions": len(exp["important_regions"]),
                "important_regions": exp["important_regions"],
            })

        # Case 2: Ball Fault (185_w0000)
        elif wid == "185_w0000":
            exp = explainer.explain_window(raw_waveform, target_class=None, is_normalized=False)
            computed_explanations["185_ball"] = (exp, row)
            plot_single_explanation(
                exp, row, gradcam_figures_dir / "gradcam_185_ball_fault.png", title_suffix="(True Positive Ball Fault)"
            )
            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": exp["target_class_name"],
                "predicted_class": exp["predicted_class_name"],
                "uncalibrated_prob": exp["probabilities"][exp["predicted_class_id"]],
                "num_important_regions": len(exp["important_regions"]),
                "important_regions": exp["important_regions"],
            })

        # Case 3: Inner Race Fault (169_w0000) -> Pred (Ball Fault) & True (Inner Race Fault)
        elif wid == "169_w0000":
            # Predicted Class (Ball Fault)
            exp_pred = explainer.explain_window(raw_waveform, target_class="Ball Fault", is_normalized=False)
            plot_single_explanation(
                exp_pred, row, gradcam_figures_dir / "gradcam_169_inner_race_pred_ball.png",
                title_suffix="(Predicted Class Attribution: Ball Fault)"
            )
            # True Class (Inner Race Fault)
            exp_true = explainer.explain_window(raw_waveform, target_class="Inner Race Fault", is_normalized=False)
            plot_single_explanation(
                exp_true, row, gradcam_figures_dir / "gradcam_169_inner_race_true_class.png",
                title_suffix="(True Class Attribution: Inner Race Fault)"
            )
            computed_explanations["169_pred"] = (exp_pred, row)
            computed_explanations["169_true"] = (exp_true, row)

            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": "Ball Fault (Predicted)",
                "predicted_class": exp_pred["predicted_class_name"],
                "uncalibrated_prob": exp_pred["probabilities"][1],
                "num_important_regions": len(exp_pred["important_regions"]),
                "important_regions": exp_pred["important_regions"],
            })
            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": "Inner Race Fault (True)",
                "predicted_class": exp_true["predicted_class_name"],
                "uncalibrated_prob": exp_true["probabilities"][2],
                "num_important_regions": len(exp_true["important_regions"]),
                "important_regions": exp_true["important_regions"],
            })

        # Case 4: Outer Race Fault (197_w0000) -> Pred (Ball Fault) & True (Outer Race Fault)
        elif wid == "197_w0000":
            # Predicted Class (Ball Fault)
            exp_pred = explainer.explain_window(raw_waveform, target_class="Ball Fault", is_normalized=False)
            plot_single_explanation(
                exp_pred, row, gradcam_figures_dir / "gradcam_197_outer_race_pred_ball.png",
                title_suffix="(Predicted Class Attribution: Ball Fault)"
            )
            # True Class (Outer Race Fault)
            exp_true = explainer.explain_window(raw_waveform, target_class="Outer Race Fault", is_normalized=False)
            plot_single_explanation(
                exp_true, row, gradcam_figures_dir / "gradcam_197_outer_race_true_class.png",
                title_suffix="(True Class Attribution: Outer Race Fault)"
            )
            computed_explanations["197_pred"] = (exp_pred, row)
            computed_explanations["197_true"] = (exp_true, row)

            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": "Ball Fault (Predicted)",
                "predicted_class": exp_pred["predicted_class_name"],
                "uncalibrated_prob": exp_pred["probabilities"][1],
                "num_important_regions": len(exp_pred["important_regions"]),
                "important_regions": exp_pred["important_regions"],
            })
            analysis_records.append({
                "window_id": wid,
                "source_file": row["source_file"],
                "fault_class": row["fault_class"],
                "explanation_target": "Outer Race Fault (True)",
                "predicted_class": exp_true["predicted_class_name"],
                "uncalibrated_prob": exp_true["probabilities"][3],
                "num_important_regions": len(exp_true["important_regions"]),
                "important_regions": exp_true["important_regions"],
            })

    # 3. Multi-case comparison figures
    logger.info("Generating multi-case comparative figures...")
    plot_misclassification_comparison(
        exp_inner_pred=computed_explanations["169_pred"][0],
        exp_inner_true=computed_explanations["169_true"][0],
        meta_inner=computed_explanations["169_pred"][1],
        exp_outer_pred=computed_explanations["197_pred"][0],
        exp_outer_true=computed_explanations["197_true"][0],
        meta_outer=computed_explanations["197_pred"][1],
        save_path=gradcam_figures_dir / "gradcam_misclassification_comparison.png",
    )

    all_cases = [
        (computed_explanations["97_normal"][0], computed_explanations["97_normal"][1], "True Normal (Correct)"),
        (computed_explanations["185_ball"][0], computed_explanations["185_ball"][1], "True Ball (Correct)"),
        (computed_explanations["169_true"][0], computed_explanations["169_true"][1], "True Inner Race (Target: Inner)"),
        (computed_explanations["197_true"][0], computed_explanations["197_true"][1], "True Outer Race (Target: Outer)"),
    ]
    plot_all_classes_comparison(all_cases, gradcam_figures_dir / "gradcam_all_classes_comparison.png")

    # 4. Save JSON summary
    summary_path = reports_dir / "gradcam_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis_target": "InspectAI 1D Grad-CAM Diagnostic Evaluation",
                "frozen_model": str(model_path.name),
                "frozen_normalizer": str(norm_path.name),
                "target_layer": "conv3",
                "records": analysis_records,
                "scientific_limitation_statement": (
                    "Attribution indicates where convolutional filters in conv3 exhibit high activation "
                    "gradient sensitivity for the selected logit. It represents model decision weighting, "
                    "not physical proof of bearing mechanics, kinematic fault frequency alignment, or RUL prognosis."
                ),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved explainability summary JSON: {summary_path}")
    print("Grad-CAM generation complete.")


if __name__ == "__main__":
    main()

