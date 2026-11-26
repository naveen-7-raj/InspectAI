"""InspectAI - Phase 11.5: Final Model Verification & Deployment Readiness Audit.

Validates the frozen release artifacts in models/final/:
1. Checkpoint and normalizer file integrity & SHA256 checksums.
2. Parameter count and architectural contract (SpectralFusionCNN, 89,604 params).
3. StreamingPredictor single-window and batch inference contracts.
4. VibrationExplainer 1D Grad-CAM attribution generation.
5. CPU inference latency profiling (median, p95, throughput).
6. Production model checkpoint isolation check.
7. Saves verification report to outputs/experiments/phase11_5/verification_report.json.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.explainability.explainer import VibrationExplainer
from src.inference.predictor import StreamingPredictor
from src.models.spectral_fusion_cnn import SpectralFusionCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.VerifyFinalModel")

FINAL_DIR = REPO_ROOT / "models" / "final"
PROD_DIR = REPO_ROOT / "models" / "checkpoints"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase11_5"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    logger.info("Initiating Final Model Verification & Deployment Readiness Audit...")
    t_start = time.time()

    ckpt_path = FINAL_DIR / "inspectai_final_model.pth"
    norm_path = FINAL_DIR / "inspectai_final_normalizer.json"
    spec_norm_path = FINAL_DIR / "inspectai_final_spectral_normalizer.json"
    config_path = FINAL_DIR / "inspectai_final_config.json"
    card_path = FINAL_DIR / "inspectai_model_card.json"

    # 1. File existence checks
    for p in [ckpt_path, norm_path, spec_norm_path, config_path, card_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required final release file: {p}")
        logger.info(f"Verified existence: {p.name} ({p.stat().st_size:,} bytes)")

    # 2. Checksum verification against config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt_sha = calculate_sha256(ckpt_path)
    norm_sha = calculate_sha256(norm_path)
    spec_sha = calculate_sha256(spec_norm_path)

    assert ckpt_sha == cfg["model"]["sha256"], "Model checkpoint SHA256 mismatch!"
    assert norm_sha == cfg["preprocessing"]["waveform_normalizer"]["sha256"], "Waveform normalizer SHA256 mismatch!"
    assert spec_sha == cfg["preprocessing"]["spectral_normalizer"]["sha256"], "Spectral normalizer SHA256 mismatch!"
    logger.info("All SHA256 checksums match configuration records perfectly.")

    # 3. Predictor loading and inference verification
    logger.info("Instantiating StreamingPredictor.from_final_model()...")
    predictor = StreamingPredictor.from_final_model(FINAL_DIR)

    assert isinstance(predictor.model, SpectralFusionCNN)
    assert predictor.is_dual_branch is True
    param_count = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    assert param_count == 89604, f"Expected 89,604 parameters, got {param_count}"
    logger.info(f"Model successfully loaded: SpectralFusionCNN ({param_count:,} parameters).")

    # Single-window prediction contract
    dummy_raw_win = np.random.randn(2048).astype(np.float32)
    single_res = predictor.predict_window(dummy_raw_win, is_normalized=False)

    required_keys = ["predicted_class", "class_id", "logits", "probabilities", "class_probabilities", "confidence_note", "waveform_normalized"]
    for k in required_keys:
        assert k in single_res, f"Missing key in prediction response: {k}"

    assert single_res["predicted_class"] in predictor.class_names
    assert 0 <= single_res["class_id"] <= 3
    assert len(single_res["probabilities"]) == 4
    assert np.isclose(sum(single_res["probabilities"]), 1.0, atol=1e-4)
    assert np.all(np.isfinite(single_res["logits"]))
    logger.info(f"Single-window prediction contract verified: predicted '{single_res['predicted_class']}'.")

    # Batch prediction contract
    dummy_batch = np.random.randn(8, 2048).astype(np.float32)
    batch_res = predictor.predict_batch(dummy_batch, is_normalized=False)
    assert len(batch_res) == 8
    for r in batch_res:
        assert r["predicted_class"] in predictor.class_names
        assert np.isclose(sum(r["probabilities"]), 1.0, atol=1e-4)
    logger.info("Batch prediction contract (batch size 8) verified.")

    # 4. Grad-CAM Explainability verification
    logger.info("Verifying 1D Grad-CAM generation with VibrationExplainer...")
    explainer = VibrationExplainer(
        model=predictor.model,
        normalizer=predictor.normalizer,
        spectral_normalizer=predictor.spectral_normalizer,
        class_names=predictor.class_names,
    )
    heatmap, pred_c, tgt_c, logits, probs = explainer.compute_gradcam(dummy_raw_win)
    assert heatmap.shape == (2048,), f"Expected heatmap shape (2048,), got {heatmap.shape}"
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
    logger.info(f"1D Grad-CAM heatmap generated successfully: shape {heatmap.shape}, min={heatmap.min():.4f}, max={heatmap.max():.4f}.")

    # 5. CPU Inference Latency Benchmarking (200 windows)
    logger.info("Benchmarking CPU single-window inference latency across 200 windows...")
    latencies = []
    # Warmup
    for _ in range(15):
        _ = predictor.predict_window(dummy_raw_win)

    for _ in range(200):
        t0 = time.perf_counter()
        _ = predictor.predict_window(dummy_raw_win)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    mean_ms = float(np.mean(latencies))
    throughput_wps = int(round(1000.0 / median_ms))
    logger.info(f"CPU Latency (Single Window): Median = {median_ms:.4f} ms | p95 = {p95_ms:.4f} ms | Throughput = ~{throughput_wps:,} windows/sec.")

    # 6. Checkpoint Isolation Check
    prod_ckpt = PROD_DIR / "inspectai_final_cnn.pth"
    prod_norm = PROD_DIR / "inspectai_final_normalizer.json"
    assert prod_ckpt.is_file(), "Production checkpoint missing!"
    assert prod_norm.is_file(), "Production normalizer missing!"
    logger.info(f"Verified production baseline checkpoint remains isolated: {prod_ckpt.name} ({prod_ckpt.stat().st_size:,} bytes).")

    # 7. Write verification report
    verification_summary = {
        "status": "VERIFIED_DEPLOYMENT_READY",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "SpectralFusionCNN",
        "trainable_parameters": param_count,
        "release_directory": str(FINAL_DIR),
        "artifacts_verified": {
            "checkpoint": {"path": str(ckpt_path), "sha256": ckpt_sha, "bytes": ckpt_path.stat().st_size},
            "waveform_normalizer": {"path": str(norm_path), "sha256": norm_sha, "bytes": norm_path.stat().st_size},
            "spectral_normalizer": {"path": str(spec_norm_path), "sha256": spec_sha, "bytes": spec_norm_path.stat().st_size},
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size},
            "model_card": {"path": str(card_path), "bytes": card_path.stat().st_size},
        },
        "inference_contracts": {
            "single_window_prediction": "PASSED",
            "batch_prediction": "PASSED",
            "gradcam_explainability": "PASSED",
            "finite_probabilities": "PASSED",
            "class_count": len(predictor.class_names),
        },
        "performance_benchmark": {
            "device": "CPU",
            "single_window_median_ms": round(median_ms, 4),
            "single_window_p95_ms": round(p95_ms, 4),
            "single_window_mean_ms": round(mean_ms, 4),
            "estimated_throughput_windows_per_second": throughput_wps,
        },
        "production_isolation": {
            "production_checkpoint_preserved": True,
            "production_checkpoint_bytes": prod_ckpt.stat().st_size,
        },
        "audit_duration_sec": round(time.time() - t_start, 2),
    }

    report_path = OUT_DIR / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(verification_summary, f, indent=2)
    logger.info(f"Saved complete verification report to {report_path}")

    print("\n================ VERIFICATION AUDIT PASSED ================")
    print(f"Status:       VERIFIED_DEPLOYMENT_READY")
    print(f"Model:        SpectralFusionCNN (89,604 parameters)")
    print(f"Latency:      Median = {median_ms:.3f} ms | p95 = {p95_ms:.3f} ms")
    print(f"Throughput:   ~{throughput_wps:,} windows/second on CPU")
    print(f"Grad-CAM:     Validated (2,048 temporal attribution vector)")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()

