"""Phase 14: Frozen Zero-Shot Evaluation on Real Paderborn Benchmark Bearings.

Evaluates the production SpectralFusionCNN candidate (frozen weights from models/final/)
against the 4 real Paderborn pilot bearings:
- K001: Healthy Reference (Normal, Class 0)
- KB23: Rolling Element / Ball Fault (Ball Fault, Class 1)
- KI04: Inner Race Fault (Inner Race Fault, Class 2)
- KA04: Outer Race Fault (Outer Race Fault, Class 3)

Strict Evaluation Contract:
- No model retraining, fine-tuning, threshold tuning, or calibration.
- Frozen CWRU waveform normalizer (mean, std) and spectral normalizer (min, max).
- Exact 4-class output mapping (0: Normal, 1: Ball, 2: Inner, 3: Outer).
- Reports window-level metrics, recording-level aggregation, and bearing-level groupings.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.inference.predictor import StreamingPredictor
from src.ingestion.external_loader import ExternalDataLoader
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.preprocessing.external_preprocessor import ExternalPreprocessor
from src.training.normalizer import VibrationNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase14.ZeroShot")

EXTRACTED_DIR = REPO_ROOT / "data" / "external" / "extracted"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase14"
MODELS_DIR = REPO_ROOT / "models" / "final"

CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

BEARING_METADATA = {
    "K001": {"ground_truth_class": "Normal", "class_idx": 0, "type": "Healthy Reference"},
    "KB23": {"ground_truth_class": "Ball Fault", "class_idx": 1, "type": "Rolling Element Fault"},
    "KI04": {"ground_truth_class": "Inner Race Fault", "class_idx": 2, "type": "Inner Race Fault"},
    "KA04": {"ground_truth_class": "Outer Race Fault", "class_idx": 3, "type": "Outer Race Fault"},
}


def load_frozen_artifacts() -> StreamingPredictor:
    predictor = StreamingPredictor.from_final_model(model_dir=MODELS_DIR, device="cpu")
    return predictor


def run_evaluation(
    predictor: StreamingPredictor,
    preprocessor: ExternalPreprocessor,
    primary_condition_only: bool = False,
) -> Dict[str, Any]:
    loader = ExternalDataLoader(external_dir=REPO_ROOT / "data" / "external")

    all_y_true = []
    all_y_pred = []
    all_confidences = []
    all_latencies_ms = []

    per_bearing_results = {}
    total_recordings_evaluated = 0
    total_invalid_windows = 0

    for b_id, b_meta in BEARING_METADATA.items():
        b_dir = EXTRACTED_DIR / b_id
        if not b_dir.exists():
            raise FileNotFoundError(f"Extracted directory for bearing {b_id} not found: {b_dir}")

        all_mat_files = sorted(b_dir.rglob("*.mat"))
        if primary_condition_only:
            # Select 1500 RPM, 0.7 Nm, 1000 N: N15_M07_F10
            mat_files = [f for f in all_mat_files if "N15_M07_F10" in f.name]
        else:
            mat_files = all_mat_files

        logger.info(f"Evaluating {len(mat_files)} recordings for Bearing {b_id} ({b_meta['ground_truth_class']})...")

        b_y_true = []
        b_y_pred = []
        b_confidences = []
        b_recording_majority_preds = []

        for mat_file in mat_files:
            total_recordings_evaluated += 1
            rec = loader.load_paderborn_mat(
                mat_path=mat_file,
                recording_id=mat_file.stem,
                fault_class=b_meta["ground_truth_class"],
            )

            # Preprocessing via standardized process_recording
            processed = preprocessor.process_recording(
                recording=rec,
                wave_normalizer=predictor.normalizer,
                spec_normalizer=predictor.spectral_normalizer,
            )
            t_wave = processed["tensor_wave"]  # (N, 1, 2048)
            t_spec = processed["tensor_spec"]  # (N, 1, 1025)
            n_win = processed["total_windows"]

            # Check for invalid values
            if torch.isnan(t_wave).any() or torch.isnan(t_spec).any() or torch.isinf(t_wave).any():
                total_invalid_windows += n_win
                continue

            rec_window_preds = []
            with torch.no_grad():
                for w_idx in range(len(t_wave)):
                    w_w = t_wave[w_idx : w_idx + 1]
                    w_s = t_spec[w_idx : w_idx + 1]

                    t0 = time.perf_counter()
                    logits = predictor.model(w_w, w_s)
                    lat = (time.perf_counter() - t0) * 1000.0
                    all_latencies_ms.append(lat)

                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])

                    rec_window_preds.append(pred)
                    b_y_pred.append(pred)
                    b_y_true.append(b_meta["class_idx"])
                    b_confidences.append(conf)

            # Recording-level majority vote
            rec_maj_pred = int(np.bincount(rec_window_preds).argmax())
            b_recording_majority_preds.append(rec_maj_pred)

        all_y_true.extend(b_y_true)
        all_y_pred.extend(b_y_pred)
        all_confidences.extend(b_confidences)

        # Bearing-level summary
        b_acc = float(accuracy_score(b_y_true, b_y_pred)) if b_y_true else 0.0
        b_pred_counts = {CLASS_NAMES[i]: int(np.sum(np.array(b_y_pred) == i)) for i in range(len(CLASS_NAMES))}
        b_rec_counts = {CLASS_NAMES[i]: int(np.sum(np.array(b_recording_majority_preds) == i)) for i in range(len(CLASS_NAMES))}

        per_bearing_results[b_id] = {
            "bearing_id": b_id,
            "ground_truth_class": b_meta["ground_truth_class"],
            "bearing_type": b_meta["type"],
            "recordings_evaluated": len(mat_files),
            "valid_windows": len(b_y_true),
            "window_level_accuracy": b_acc,
            "predicted_window_distribution": b_pred_counts,
            "recording_majority_vote_distribution": b_rec_counts,
            "mean_confidence": float(np.mean(b_confidences)) if b_confidences else 0.0,
        }

    # Aggregate metrics
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    overall_acc = float(accuracy_score(y_true_arr, y_pred_arr))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(CLASS_NAMES)))).tolist()

    per_class_metrics = {}
    for idx, c_name in enumerate(CLASS_NAMES):
        per_class_metrics[c_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
        }

    latency_summary = {
        "median_ms": float(np.median(all_latencies_ms)),
        "p95_ms": float(np.percentile(all_latencies_ms, 95)),
        "p99_ms": float(np.percentile(all_latencies_ms, 99)),
        "mean_ms": float(np.mean(all_latencies_ms)),
    }

    eval_summary = {
        "evaluation_name": "Phase 14 Real Paderborn Pilot Zero-Shot Benchmark",
        "primary_condition_only": primary_condition_only,
        "pilot_scope": "4-bearing exploratory pilot (K001, KB23, KI04, KA04)",
        "total_physical_bearings": len(BEARING_METADATA),
        "total_source_measurements": total_recordings_evaluated,
        "total_windows_evaluated": len(y_true_arr),
        "total_invalid_windows": total_invalid_windows,
        "accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": conf_mat,
        },
        "per_bearing_summary": per_bearing_results,
        "mean_softmax_confidence": float(np.mean(all_confidences)),
        "cpu_latency_ms": latency_summary,
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return eval_summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor = ExternalPreprocessor()
    predictor = load_frozen_artifacts()

    logger.info("Executing Frozen Zero-Shot Evaluation on Real Paderborn Pilot Data...")

    # 1. Primary Operating Condition (1500 RPM, 0.7 Nm, 1000 N -> N15_M07_F10)
    logger.info("--- Benchmark A: Primary Operating Condition (N15_M07_F10: 20 measurements / bearing) ---")
    summary_primary = run_evaluation(predictor, preprocessor, primary_condition_only=True)

    # 2. Comprehensive (All 4 Operating Conditions -> 80 measurements / bearing)
    logger.info("--- Benchmark B: All Operating Conditions (80 measurements / bearing) ---")
    summary_all = run_evaluation(predictor, preprocessor, primary_condition_only=False)

    full_results = {
        "primary_operating_condition_n15_m07_f10": summary_primary,
        "all_operating_conditions_pooled": summary_all,
    }

    out_file = OUTPUT_DIR / "zero_shot_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Zero-shot evaluation complete. Summary saved to: {out_file}")
    print("\n" + "=" * 60)
    print("PHASE 14 ZERO-SHOT EVALUATION SUMMARY (Primary Condition: N15_M07_F10)")
    print("=" * 60)
    print(f"Total Bearings: {summary_primary['total_physical_bearings']}")
    print(f"Total Measurements: {summary_primary['total_source_measurements']}")
    print(f"Total Windows: {summary_primary['total_windows_evaluated']}")
    print(f"Overall Accuracy: {summary_primary['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy: {summary_primary['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro-F1 Score: {summary_primary['macro_f1'] * 100:.2f}%")
    print(f"Mean Confidence: {summary_primary['mean_softmax_confidence'] * 100:.2f}%")
    print(f"Median CPU Latency: {summary_primary['cpu_latency_ms']['median_ms']:.2f} ms")
    print("Confusion Matrix:")
    for row_name, row in zip(CLASS_NAMES, summary_primary['confusion_matrix']['matrix']):
        print(f"  {row_name:18s}: {row}")


if __name__ == "__main__":
    main()
