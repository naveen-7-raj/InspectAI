"""Zero-shot external validation execution engine.

Loads the frozen InspectAI SpectralFusionCNN model and runs zero-shot inference
on conditioned external vibration signals without fine-tuning, retraining, or
re-fitting normalizers.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability.explainer import VibrationExplainer
from src.inference.predictor import StreamingPredictor
from src.ingestion.external_loader import ExternalDataLoader, ExternalRecording
from src.preprocessing.external_preprocessor import ExternalPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.ExternalZeroShot")


def evaluate_external_recording(
    recording: ExternalRecording,
    predictor: StreamingPredictor,
    preprocessor: ExternalPreprocessor,
    explainer: Optional[VibrationExplainer] = None,
) -> Dict[str, Any]:
    """Execute zero-shot evaluation on an individual external recording."""
    logger.info(f"Evaluating external recording: {recording.recording_id} (True Class: {recording.fault_class})")

    # 1. Condition continuous signal and segment into windows
    processed = preprocessor.process_recording(
        recording=recording,
        wave_normalizer=predictor.normalizer,
        spec_normalizer=predictor.spectral_normalizer,
    )

    tensors_wave = processed["tensor_wave"]  # (N, 1, 2048)
    tensors_spec = processed["tensor_spec"]  # (N, 1, 1025)
    n_windows = processed["total_windows"]

    predictions = []
    probabilities = []
    latencies = []

    predictor.model.eval()
    with torch.no_grad():
        for i in range(n_windows):
            w_wave = tensors_wave[i : i + 1].to(predictor.device)
            w_spec = tensors_spec[i : i + 1].to(predictor.device)

            t0 = time.perf_counter()
            logits = predictor.model(w_wave, w_spec)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            predictions.append(pred_idx)
            probabilities.append(probs)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # Class distribution among predictions
    pred_counts = {cls_name: int(np.sum(predictions == idx)) for idx, cls_name in enumerate(predictor.class_names)}
    median_latency = float(np.median(latencies))
    mean_confidence = float(np.mean(np.max(probabilities, axis=1)))

    return {
        "recording_id": recording.recording_id,
        "dataset_name": recording.dataset_name,
        "true_fault_class": recording.fault_class,
        "bearing_model": recording.bearing_model,
        "rotational_speed_rpm": recording.rotational_speed_rpm,
        "total_windows_evaluated": n_windows,
        "predicted_class_distribution": pred_counts,
        "mean_confidence": mean_confidence,
        "median_cpu_latency_ms": median_latency,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot External Benchmark Evaluator")
    parser.add_argument("--model-dir", type=str, default="models/final")
    parser.add_argument("--output", type=str, default="outputs/experiments/phase13/zero_shot_summary.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    logger.info(f"Loading frozen model from: {model_dir}")
    predictor = StreamingPredictor.from_final_model(model_dir=model_dir, device="cpu")
    preprocessor = ExternalPreprocessor()

    logger.info("External zero-shot evaluation engine initialized successfully.")
    logger.info("Ready for zero-shot benchmarking across external bearing datasets.")


if __name__ == "__main__":
    main()
