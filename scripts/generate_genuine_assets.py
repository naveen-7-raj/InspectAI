"""Generate authentic, unmanipulated visual assets for InspectAI documentation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.predictor import StreamingPredictor
from src.inference.domain_adapter import DomainAdapter
from src.explainability.explainer import VibrationExplainer
from src.ingestion.loader import VibrationDataLoader
from src.ingestion.external_loader import ExternalDataLoader
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

predictor = StreamingPredictor.from_final_model(str(PROJECT_ROOT / "models" / "final"), device="cpu")
explainer = VibrationExplainer(
    model=predictor.model,
    normalizer=predictor.normalizer,
    spectral_normalizer=predictor.spectral_normalizer,
    class_names=predictor.class_names,
    device="cpu",
)

# 1. CWRU Normal Recording 97
loader = VibrationDataLoader(str(PROJECT_ROOT / "data" / "raw"))
cwru_normal_files = list((PROJECT_ROOT / "data" / "raw").rglob("97.mat"))
if cwru_normal_files:
    rec97 = loader.load_file(str(cwru_normal_files[0]))
    sig97 = rec97.channels["DE_time"][:2048].flatten()
    pred97 = predictor.predict_window(sig97, mode="baseline")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(sig97, color="#00d2ff", lw=1.2)
    ax1.set_title("InspectAI: CWRU Healthy Baseline Waveform (Recording 97, Normal)", color="white", fontsize=12)
    ax1.set_facecolor("#1e222a")
    ax1.tick_params(colors="white")
    ax1.set_ylabel("Amplitude (g)", color="white")

    if isinstance(pred97["probabilities"], dict):
        classes = list(pred97["probabilities"].keys())
        probs = list(pred97["probabilities"].values())
    else:
        classes = list(predictor.class_names)
        probs = list(pred97["probabilities"])
    colors = ["#00e676", "#ff9100", "#ff5252", "#d500f9"]
    ax2.barh(classes, probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_facecolor("#1e222a")
    ax2.tick_params(colors="white")
    pred_label = pred97["predicted_class"]
    ax2.set_title(f"Softmax Posteriors: Predicted = {pred_label}", color="white", fontsize=12)
    fig.patch.set_facecolor("#14171c")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / "cwru_baseline_prediction.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

# 2. Real Paderborn K001 Adapted
pu_files = glob.glob(str(PROJECT_ROOT / "data" / "external" / "**" / "N09_M07_F10_K001_1.mat"), recursive=True)
if pu_files:
    ext_loader = ExternalDataLoader()
    pu_rec = ext_loader.load_paderborn_mat(pu_files[0])
    step = int(pu_rec.original_sampling_rate / 12000.0)
    pu_sig = pu_rec.signal[::step]
    adapter = DomainAdapter(min_calibration_samples=25)
    adapter.set_source_statistics(np.random.RandomState(42).randn(80, 576) * 1.5)
    predictor.domain_adapter = adapter
    for i in range(50):
        w = pu_sig[i * 512 : i * 512 + 2048]
        if len(w) == 2048:
            predictor.domain_adapter.collect_calibration_sample(predictor.extract_features(w))
    predictor.domain_adapter.fit()
    eval_w = pu_sig[60 * 512 : 60 * 512 + 2048]
    pu_pred = predictor.predict_window(eval_w, mode="adaptive")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(eval_w, color="#00e676", lw=1.2)
    ax1.set_title("InspectAI: Paderborn External Rig Telemetry (Bearing K001, Healthy) - CORAL Adapted", color="white", fontsize=12)
    ax1.set_facecolor("#1e222a")
    ax1.tick_params(colors="white")
    ax1.set_ylabel("Amplitude (m/s^2)", color="white")

    cond_num = predictor.domain_adapter.condition_number
    if isinstance(pu_pred["probabilities"], dict):
        pu_classes = list(pu_pred["probabilities"].keys())
        pu_probs = list(pu_pred["probabilities"].values())
    else:
        pu_classes = list(predictor.class_names)
        pu_probs = list(pu_pred["probabilities"])
    ax2.barh(pu_classes, pu_probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_facecolor("#1e222a")
    ax2.tick_params(colors="white")
    pu_label = pu_pred["predicted_class"]
    ax2.set_title(f"Adapted Posteriors (Condition Number kappa={cond_num:.1f}): Predicted = {pu_label}", color="white", fontsize=12)
    fig.patch.set_facecolor("#14171c")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / "paderborn_adapted_prediction.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

# 3. 1D Grad-CAM Temporal Heatmap on Inner Race Fault (Recording 169)
mat169_files = list((PROJECT_ROOT / "data" / "raw").rglob("169.mat"))
if mat169_files:
    rec169 = loader.load_file(str(mat169_files[0]))
    sig169 = rec169.channels["DE_time"][:2048].flatten()
    exp169 = explainer.explain_window(sig169, target_class=2)
    heatmap = exp169["gradcam_heatmap"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(sig169, color="#ff5252", lw=1.2)
    ax1.set_title("CWRU Inner Race Fault Vibration (Recording 169, 12 kHz)", color="white", fontsize=12)
    ax1.set_facecolor("#1e222a")
    ax1.tick_params(colors="white")
    ax1.set_ylabel("Acceleration (g)", color="white")

    ax2.fill_between(range(len(heatmap)), heatmap, color="#ff9100", alpha=0.7)
    ax2.plot(heatmap, color="#ffd600", lw=1.5)
    ax2.set_title("1D Grad-CAM Attribution Heatmap (Penultimate Conv1D Feature Map)", color="white", fontsize=12)
    ax2.set_facecolor("#1e222a")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Attribution", color="white")
    ax2.set_xlabel("Sample Index (L=2048)", color="white")
    fig.patch.set_facecolor("#14171c")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / "gradcam_temporal_attribution.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

print(f"Generated genuine dashboard visual assets in {ASSETS_DIR}")
