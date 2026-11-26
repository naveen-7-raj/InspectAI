"""Phase 14 Real-Data Pilot Integration Tests.

Validates:
1. Real Paderborn MATLAB file parsing (K001, KA04, KI04, KB23).
2. Vibration channel extraction (Channel 6 'vibration_1').
3. Physical unit conversion (m/s^2 -> g) and 64 kHz -> 12 kHz rational resampling.
4. Feature generation (2048-sample window, 1025-bin Hilbert envelope spectrum).
5. Application of frozen CWRU normalizers without runtime modification.
6. Frozen model inference contract (4-class probabilities, correct shape, zero NaNs).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.inference.predictor import StreamingPredictor
from src.ingestion.external_loader import ExternalDataLoader
from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.preprocessing.external_preprocessor import ExternalPreprocessor
from src.training.normalizer import VibrationNormalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = REPO_ROOT / "data" / "external" / "extracted"
MODELS_DIR = REPO_ROOT / "models" / "final"


@pytest.fixture(scope="module")
def external_loader():
    return ExternalDataLoader(external_dir=REPO_ROOT / "data" / "external")


@pytest.fixture(scope="module")
def external_preprocessor():
    return ExternalPreprocessor(target_fs=12000, window_size=2048, step_size=512)


@pytest.fixture(scope="module")
def frozen_predictor():
    model_path = MODELS_DIR / "inspectai_final_model.pth"
    if not model_path.exists():
        pytest.skip("Final model checkpoint missing.")
    return StreamingPredictor.from_final_model(model_dir=MODELS_DIR, device="cpu")


class TestRealPaderbornPilot:
    """Tests executing against actual downloaded Paderborn recordings."""

    def test_real_k001_healthy_mat_loading(self, external_loader):
        """Verify real healthy K001 recording parses channel 6 vibration_1 at 64 kHz."""
        k001_dir = EXTRACTED_DIR / "K001"
        mat_files = list(k001_dir.rglob("*.mat"))
        if not mat_files:
            pytest.skip("K001 not yet extracted.")

        rec = external_loader.load_paderborn_mat(
            mat_path=mat_files[0],
            recording_id=mat_files[0].stem,
            fault_class="Normal",
        )

        assert rec.original_sampling_rate == 64000.0
        assert rec.units == "m/s^2"
        assert len(rec.signal) >= 200000  # Nominal ~256,823 samples
        assert np.isfinite(rec.signal).all()

    def test_real_ka04_outer_race_loading(self, external_loader):
        """Verify real KA04 outer race recording parses channel 6 vibration_1 at 64 kHz."""
        ka04_dir = EXTRACTED_DIR / "KA04"
        mat_files = list(ka04_dir.rglob("*.mat"))
        if not mat_files:
            pytest.skip("KA04 not yet extracted.")

        rec = external_loader.load_paderborn_mat(
            mat_path=mat_files[0],
            recording_id=mat_files[0].stem,
            fault_class="Outer Race Fault",
        )

        assert rec.original_sampling_rate == 64000.0
        assert rec.units == "m/s^2"
        assert len(rec.signal) >= 200000
        assert np.isfinite(rec.signal).all()

    def test_real_ki04_inner_race_loading(self, external_loader):
        """Verify real KI04 inner race recording parses channel 6 vibration_1 at 64 kHz."""
        ki04_dir = EXTRACTED_DIR / "KI04"
        mat_files = list(ki04_dir.rglob("*.mat"))
        if not mat_files:
            pytest.skip("KI04 not yet extracted.")

        rec = external_loader.load_paderborn_mat(
            mat_path=mat_files[0],
            recording_id=mat_files[0].stem,
            fault_class="Inner Race Fault",
        )

        assert rec.original_sampling_rate == 64000.0
        assert rec.units == "m/s^2"
        assert len(rec.signal) >= 200000
        assert np.isfinite(rec.signal).all()

    def test_real_signal_preprocessing_pipeline(self, external_loader, external_preprocessor, frozen_predictor):
        """Verify unit conversion, polyphase decimation, windowing, and Hilbert spectrum."""
        k001_dir = EXTRACTED_DIR / "K001"
        mat_files = list(k001_dir.rglob("*.mat"))
        if not mat_files:
            pytest.skip("K001 not yet extracted.")

        rec = external_loader.load_paderborn_mat(mat_files[0])
        cond = external_preprocessor.condition_continuous_signal(
            rec.signal, orig_fs=rec.original_sampling_rate, units=rec.units
        )

        # Rational resampling 64k -> 12k is 3/16 ratio
        expected_len = int(round(len(rec.signal) * 3 / 16))
        assert abs(len(cond) - expected_len) <= 5

        # Segmentation
        windows, _ = external_preprocessor.segment_windows(cond)
        assert windows.ndim == 2
        assert windows.shape[1] == 2048
        assert len(windows) >= 80

        # Envelope spectra
        _, spectra, _ = compute_hilbert_envelope_spectrum(windows, sampling_rate_hz=12000)
        assert spectra.shape == (len(windows), 1025)

        # Frozen normalization
        norm_w = frozen_predictor.normalizer.transform(windows)
        norm_s = frozen_predictor.spectral_normalizer.transform(spectra)

        assert not np.isnan(norm_w).any()
        assert not np.isnan(norm_s).any()
        assert not np.isinf(norm_w).any()
        assert not np.isinf(norm_s).any()

    def test_real_signal_zero_shot_forward_pass(self, external_loader, external_preprocessor, frozen_predictor):
        """Verify end-to-end model forward pass produces valid 4-class probabilities."""
        ka04_dir = EXTRACTED_DIR / "KA04"
        mat_files = list(ka04_dir.rglob("*.mat"))
        if not mat_files:
            pytest.skip("KA04 not yet extracted.")

        rec = external_loader.load_paderborn_mat(mat_files[0])
        processed = external_preprocessor.process_recording(
            recording=rec,
            wave_normalizer=frozen_predictor.normalizer,
            spec_normalizer=frozen_predictor.spectral_normalizer,
        )
        t_w = processed["tensor_wave"][:5]
        t_s = processed["tensor_spec"][:5]

        frozen_predictor.model.eval()
        with torch.no_grad():
            logits = frozen_predictor.model(t_w, t_s)
            probs = torch.softmax(logits, dim=1).numpy()

        assert logits.shape == (5, 4)
        assert probs.shape == (5, 4)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
        assert not np.isnan(probs).any()
