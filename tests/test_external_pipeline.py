"""Unit and integration tests for Phase 13 isolated external dataset pipeline."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import scipy.io as sio
import torch

from src.ingestion.external_loader import (
    ExternalDataLoader,
    ExternalRecording,
    compute_file_sha256,
)
from src.preprocessing.external_preprocessor import ExternalPreprocessor
from src.inference.predictor import StreamingPredictor
from scripts.audit_external_dataset import run_audit
from scripts.run_external_zero_shot import evaluate_external_recording


@pytest.fixture(scope="module")
def external_preprocessor():
    return ExternalPreprocessor(target_fs=12000, window_size=2048, step_size=512)


@pytest.fixture(scope="module")
def final_predictor():
    final_dir = Path("models/final")
    if not (final_dir / "inspectai_final_model.pth").exists():
        pytest.skip("Frozen final model not available.")
    return StreamingPredictor.from_final_model(model_dir=final_dir, device="cpu")


class TestExternalDataLoader:
    """Test loading and extraction of external dataset file formats."""

    def test_load_paderborn_mock_mat(self, tmp_path):
        """Verify parsing of Paderborn University structure with 64 kHz sampling."""
        fs = 64000
        duration_sec = 0.5
        n_samples = int(fs * duration_sec)
        mock_signal = np.sin(2 * np.pi * 76.3 * np.linspace(0, duration_sec, n_samples)).astype(np.float32)

        mat_path = tmp_path / "N15_M07_F10_KA01_1.mat"
        sio.savemat(str(mat_path), {"vibration": mock_signal})

        loader = ExternalDataLoader(external_dir=tmp_path)
        rec = loader.load_paderborn_mat(
            mat_path=mat_path,
            recording_id="EXT_PU_KA01_1",
            fault_class="Outer Race Fault",
        )

        assert rec.recording_id == "EXT_PU_KA01_1"
        assert rec.dataset_name == "Paderborn_University"
        assert rec.original_sampling_rate == 64000.0
        assert rec.units == "m/s^2"
        assert len(rec.signal) == n_samples
        assert len(rec.file_sha256) == 64

    def test_load_mfpt_mock_mat(self, tmp_path):
        """Verify parsing of MFPT structure with 48,828 Hz sampling."""
        fs = 48828
        duration_sec = 0.5
        n_samples = int(fs * duration_sec)
        mock_signal = (np.random.randn(n_samples) * 0.5).astype(np.float32)

        mat_path = tmp_path / "OuterRaceFault_1.mat"
        # Create nested bearing struct like MFPT
        bearing_dict = {
            "bearing": {
                "gs": mock_signal,
                "sr": fs,
                "load": 270.0,
                "rate": 16.67,
            }
        }
        sio.savemat(str(mat_path), bearing_dict)

        loader = ExternalDataLoader(external_dir=tmp_path)
        rec = loader.load_mfpt_mat(
            mat_path=mat_path,
            recording_id="EXT_MFPT_OR_1",
            fault_class="Outer Race Fault",
        )

        assert rec.recording_id == "EXT_MFPT_OR_1"
        assert rec.dataset_name == "MFPT"
        assert rec.original_sampling_rate == 48828.0
        assert rec.units == "g"
        assert len(rec.signal) == n_samples


class TestExternalPreprocessor:
    """Test conditioning, resampling, windowing, and Hilbert spectrum extraction."""

    def test_resample_paderborn_64k_to_12k(self, external_preprocessor):
        """Verify rational resampling factor 3/16 preserves length accurately."""
        fs_in = 64000
        fs_out = 12000
        duration_sec = 1.0
        t = np.linspace(0, duration_sec, int(fs_in * duration_sec), endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t).astype(np.float32)

        resampled = external_preprocessor.resample_to_target(sig, orig_fs=fs_in)
        expected_len = int(len(sig) * (fs_out / fs_in))
        assert abs(len(resampled) - expected_len) <= 2
        assert resampled.dtype == np.float32

    def test_condition_and_unit_conversion(self, external_preprocessor):
        """Verify conversion from m/s^2 to g and zero DC bias."""
        fs_in = 64000
        n_samples = 64000
        # 10 m/s^2 constant DC offset + AC sine wave
        sig = (10.0 + 5.0 * np.sin(2 * np.pi * 150 * np.linspace(0, 1.0, n_samples))).astype(np.float32)

        conditioned = external_preprocessor.condition_continuous_signal(
            sig, orig_fs=fs_in, units="m/s^2"
        )

        # DC offset should be centered near 0.0 (allowing for filter transient edge effects)
        assert abs(np.mean(conditioned)) < 0.01
        assert abs(np.mean(conditioned[100:-100])) < 1e-3
        # Signal length should match 12,000 samples (1 second at 12 kHz)
        assert abs(len(conditioned) - 12000) <= 2

    def test_window_segmentation_overlap(self, external_preprocessor):
        """Verify 75% overlap sliding window segmentation."""
        conditioned = np.random.randn(12000).astype(np.float32)
        windows, meta = external_preprocessor.segment_windows(conditioned)

        # For 12,000 samples with L=2048 and S=512:
        # (12000 - 2048) / 512 + 1 = 19.4 -> 20 windows
        assert windows.shape == (20, 2048)
        assert len(meta) == 20
        assert meta[1]["start_sample"] - meta[0]["start_sample"] == 512

    def test_full_recording_pipeline(self, external_preprocessor, final_predictor):
        """Verify end-to-end processing pipeline produces model-ready PyTorch tensors."""
        fs_in = 64000
        sig = np.random.randn(fs_in * 2).astype(np.float32)
        rec = ExternalRecording(
            recording_id="EXT_TEST_001",
            dataset_name="Paderborn_University",
            file_path=Path("dummy.mat"),
            file_sha256="0" * 64,
            signal=sig,
            original_sampling_rate=fs_in,
            units="m/s^2",
            fault_class="Normal",
            bearing_model="SKF 6203",
            rotational_speed_rpm=1500.0,
            load_description="1000 N",
        )

        processed = external_preprocessor.process_recording(
            recording=rec,
            wave_normalizer=final_predictor.normalizer,
            spec_normalizer=final_predictor.spectral_normalizer,
        )

        assert processed["tensor_wave"].shape[1:] == (1, 2048)
        assert processed["tensor_spec"].shape[1:] == (1, 1025)
        assert processed["tensor_wave"].dtype == torch.float32
        assert processed["tensor_spec"].dtype == torch.float32


class TestExternalValidationRunner:
    """Test execution of zero-shot external evaluation engine."""

    def test_zero_shot_evaluation_execution(self, final_predictor, external_preprocessor):
        """Verify evaluate_external_recording runs cleanly without error."""
        sig = (np.random.randn(64000) * 0.5).astype(np.float32)
        rec = ExternalRecording(
            recording_id="EXT_PU_EVAL_TEST",
            dataset_name="Paderborn_University",
            file_path=Path("dummy_eval.mat"),
            file_sha256="f" * 64,
            signal=sig,
            original_sampling_rate=64000.0,
            units="m/s^2",
            fault_class="Inner Race Fault",
            bearing_model="SKF 6203",
            rotational_speed_rpm=1500.0,
            load_description="1000 N",
        )

        result = evaluate_external_recording(
            recording=rec,
            predictor=final_predictor,
            preprocessor=external_preprocessor,
        )

        assert result["recording_id"] == "EXT_PU_EVAL_TEST"
        assert result["total_windows_evaluated"] > 0
        assert "predicted_class_distribution" in result
        assert 0.0 <= result["mean_confidence"] <= 1.0
        assert result["median_cpu_latency_ms"] < 50.0

    def test_audit_manifest_validation(self, tmp_path):
        """Verify audit script executes on external dataset manifest."""
        from scripts.acquire_external_dataset import generate_manifest

        manifest_path = generate_manifest(tmp_path)
        report_path = tmp_path / "test_audit.json"
        audit_res = run_audit(manifest_path, report_path)

        assert audit_res["audit_status"] == "VALIDATED"
        assert audit_res["bearing_count"] == 4
        assert len(audit_res["supported_classes"]) == 4
        assert audit_res["cwru_quarantine_overlap"] is False
