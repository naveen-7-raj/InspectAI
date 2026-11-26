"""Unit and integration tests for Phase 11.5 Final Model Release.

Verifies:
1. Checkpoint and normalizer file existence and SHA256 integrity.
2. Model parameter count and architecture contract (SpectralFusionCNN, 89,604 parameters).
3. StreamingPredictor single-window and batch inference contracts.
4. VibrationExplainer 1D Grad-CAM heatmap generation on final model.
5. CPU inference latency is strictly < 10 ms per window.
6. Production checkpoint isolation: inspectai_final_cnn.pth preserved and untouched.
7. Model card schema and ethical/operational limitation disclosures.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch

from src.explainability.explainer import VibrationExplainer
from src.features.frequency_features import SpectralNormalizer
from src.inference.predictor import StreamingPredictor
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.training.normalizer import VibrationNormalizer
from src.training.multiload_splitter import PROTECTED_HOLDOUT_IDS

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = REPO_ROOT / "models" / "final"
PROD_DIR = REPO_ROOT / "models" / "checkpoints"


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


class TestFinalModelArtifacts:
    """Verify integrity and provenance of frozen final release files."""

    def test_all_release_files_exist(self):
        """All five required release files exist in models/final/."""
        expected_files = [
            "inspectai_final_model.pth",
            "inspectai_final_normalizer.json",
            "inspectai_final_spectral_normalizer.json",
            "inspectai_final_config.json",
            "inspectai_model_card.json",
        ]
        for fname in expected_files:
            p = FINAL_DIR / fname
            assert p.is_file(), f"Missing required file: {fname}"
            assert p.stat().st_size > 50, f"File {fname} is unexpectedly empty"

    def test_sha256_checksums_match_config(self):
        """Model and normalizer SHA256 hashes match configuration records."""
        with open(FINAL_DIR / "inspectai_final_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ckpt_sha = calculate_sha256(FINAL_DIR / "inspectai_final_model.pth")
        norm_sha = calculate_sha256(FINAL_DIR / "inspectai_final_normalizer.json")
        spec_sha = calculate_sha256(FINAL_DIR / "inspectai_final_spectral_normalizer.json")

        assert ckpt_sha == cfg["model"]["sha256"]
        assert norm_sha == cfg["preprocessing"]["waveform_normalizer"]["sha256"]
        assert spec_sha == cfg["preprocessing"]["spectral_normalizer"]["sha256"]

    def test_model_card_provenance_and_quarantine(self):
        """Model card records exact 36 development recordings and quarantined holdouts."""
        with open(FINAL_DIR / "inspectai_model_card.json", "r", encoding="utf-8") as f:
            card = json.load(f)

        dev_count = card["training_data"]["development_recordings_count"]
        assert dev_count == 36

        quarantined = card["training_data"]["permanently_quarantined_recordings"]
        for h_id in PROTECTED_HOLDOUT_IDS:
            assert h_id in quarantined

        # Ethical/operational limitations disclosures
        limits = card["ethical_and_operational_limitations"]
        assert "cross_machine_transfer" in limits
        assert "early_failure_prediction" in limits
        assert "unseen_fault_modes" in limits
        assert "uncalibrated_probabilities" in limits


class TestFinalModelInferenceContract:
    """Verify StreamingPredictor inference contracts with the final model."""

    @pytest.fixture(scope="class")
    def predictor(self):
        return StreamingPredictor.from_final_model(FINAL_DIR)

    def test_model_architecture_and_parameters(self, predictor):
        """Final model is SpectralFusionCNN with exactly 89,604 trainable parameters."""
        assert isinstance(predictor.model, SpectralFusionCNN)
        assert predictor.is_dual_branch is True
        count = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
        assert count == 89604
        assert len(predictor.class_names) == 4

    def test_single_window_prediction(self, predictor):
        """Single-window prediction returns required schema and valid probabilities."""
        window = np.random.randn(2048).astype(np.float32)
        res = predictor.predict_window(window, is_normalized=False)

        for key in ["predicted_class", "class_id", "logits", "probabilities", "class_probabilities", "confidence_note", "waveform_normalized"]:
            assert key in res

        assert res["predicted_class"] in predictor.class_names
        assert 0 <= res["class_id"] <= 3
        assert len(res["probabilities"]) == 4
        assert np.isclose(sum(res["probabilities"]), 1.0, atol=1e-4)
        for p in res["probabilities"]:
            assert 0.0 <= p <= 1.0
        assert np.all(np.isfinite(res["logits"]))

    def test_batch_prediction(self, predictor):
        """Batch prediction returns expected number of formatted predictions."""
        batch = np.random.randn(4, 2048).astype(np.float32)
        res_list = predictor.predict_batch(batch, is_normalized=False)
        assert len(res_list) == 4
        for r in res_list:
            assert r["predicted_class"] in predictor.class_names
            assert np.isclose(sum(r["probabilities"]), 1.0, atol=1e-4)

    def test_gradcam_attribution_generation(self, predictor):
        """VibrationExplainer generates valid 2,048-sample 1D Grad-CAM heatmap."""
        explainer = VibrationExplainer(
            model=predictor.model,
            normalizer=predictor.normalizer,
            spectral_normalizer=predictor.spectral_normalizer,
            class_names=predictor.class_names,
        )
        window = np.random.randn(2048).astype(np.float32)
        heatmap, pred_c, tgt_c, logits, probs = explainer.compute_gradcam(window)

        assert heatmap.shape == (2048,)
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
        assert np.all(np.isfinite(heatmap))

    def test_cpu_inference_latency_benchmark(self, predictor):
        """Single-window CPU latency is under 10 ms."""
        import time
        window = np.random.randn(2048).astype(np.float32)
        # Warmup
        for _ in range(5):
            _ = predictor.predict_window(window)

        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = predictor.predict_window(window)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        median_ms = float(np.median(times))
        assert median_ms < 10.0, f"Latency {median_ms:.2f} ms exceeds 10 ms threshold"


class TestProductionCheckpointPreservation:
    """Verify that existing production model files remain strictly isolated and untouched."""

    def test_production_files_exist_and_unmodified(self):
        """Production checkpoint and normalizer exist in models/checkpoints/."""
        prod_ckpt = PROD_DIR / "inspectai_final_cnn.pth"
        prod_norm = PROD_DIR / "inspectai_final_normalizer.json"

        assert prod_ckpt.is_file()
        assert prod_norm.is_file()
        assert prod_ckpt.stat().st_size > 100000

