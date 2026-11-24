"""Unit and integration tests for InspectAI FastAPI Prediction Backend (Phase 6.1)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.dependencies import AppState, load_model_artifacts
from backend.main import app
from src.utils.config_parser import get_resolved_paths, load_config


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers lifespan startup and shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dummy_12k_signal():
    """Generate a clean synthetic 12 kHz vibration window of 2048 samples."""
    t = np.linspace(0, 2048 / 12000.0, 2048, endpoint=False)
    # Fundamental + harmonic + low noise
    sig = 0.5 * np.sin(2 * np.pi * 30 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
    return sig.tolist()


@pytest.fixture
def dummy_48k_signal():
    """Generate a clean synthetic 48 kHz vibration window of 8192 samples."""
    t = np.linspace(0, 8192 / 48000.0, 8192, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 30 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
    return sig.tolist()


class TestHealthEndpoint:
    """Tests verifying GET /api/v1/health operational and readiness status."""

    def test_health_success(self, client):
        """Verify 200 OK and complete status dictionary when model is loaded."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "VibrationCNN1D"
        assert data["device"] == "cpu"
        assert data["gradcam_available"] is True
        assert data["normalizer_fitted"] is True
        assert len(data["supported_classes"]) == 4
        assert "Normal" in data["supported_classes"]
        assert "Ball Fault" in data["supported_classes"]

    def test_legacy_health_redirect(self, client):
        """Verify root-level /health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_degraded_when_model_unloaded(self):
        """Verify 503 Service Unavailable when model artifacts failed to load."""
        from backend.main import app
        # Create client with empty uninitialized state
        with TestClient(app) as test_client:
            original_state = test_client.app.state.app_state
            try:
                # Mock uninitialized state
                test_client.app.state.app_state = AppState(is_loaded=False, model=None)
                resp = test_client.get("/api/v1/health")
                assert resp.status_code == 503
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["model_loaded"] is False
            finally:
                test_client.app.state.app_state = original_state


class TestPredictionEndpoint:
    """Tests verifying POST /api/v1/predict inference and validation logic."""

    def test_predict_12k_signal_no_explain(self, client, dummy_12k_signal):
        """Verify prediction on a nominal 12 kHz, 2048-sample signal without Grad-CAM."""
        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": 12000,
            "explain": False,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["predicted_class"] in [
            "Normal",
            "Ball Fault",
            "Inner Race Fault",
            "Outer Race Fault",
        ]
        assert 0 <= data["class_id"] <= 3
        assert len(data["logits"]) == 4
        assert len(data["probabilities"]) == 4
        assert np.isclose(sum(data["probabilities"]), 1.0, atol=1e-4)
        assert "uncalibrated" in data["confidence_note"].lower()
        assert data["input_sampling_rate_hz"] == 12000
        assert data["processed_sampling_rate_hz"] == 12000
        assert data["processed_signal_length"] == 2048
        assert data["model_name"] == "VibrationCNN1D"
        assert data["processing_time_ms"] > 0
        assert data["gradcam_heatmap"] is None
        assert data["important_regions"] is None

    def test_predict_48k_signal_with_resampling(self, client, dummy_48k_signal):
        """Verify prediction on a 48 kHz, 8192-sample signal downsampled to 12 kHz."""
        payload = {
            "signal": dummy_48k_signal,
            "sampling_rate_hz": 48000,
            "explain": False,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["input_sampling_rate_hz"] == 48000
        assert data["processed_sampling_rate_hz"] == 12000
        assert data["processed_signal_length"] == 2048
        assert len(data["probabilities"]) == 4

    def test_predict_with_gradcam_explanation(self, client, dummy_12k_signal):
        """Verify Grad-CAM attribution generation when explain=True."""
        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": 12000,
            "explain": True,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["gradcam_heatmap"] is not None
        assert len(data["gradcam_heatmap"]) == 2048
        assert data["target_class"] == data["predicted_class"]
        assert isinstance(data["important_regions"], list)
        heatmap = np.array(data["gradcam_heatmap"])
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    def test_predict_with_custom_target_class(self, client, dummy_12k_signal):
        """Verify Grad-CAM explanation targeted at a specific fault class."""
        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": 12000,
            "explain": True,
            "target_class": "Inner Race Fault",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["target_class"] == "Inner Race Fault"
        assert len(data["gradcam_heatmap"]) == 2048

    def test_reject_empty_signal(self, client):
        """Verify rejection of empty signal with 422 Unprocessable Entity."""
        payload = {
            "signal": [],
            "sampling_rate_hz": 12000,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code in [400, 422]
        assert "empty" in response.text.lower() or "validation" in response.text.lower()

    def test_reject_nan_or_inf_signal(self, client):
        """Verify rejection of signals containing NaN or Inf."""
        # 1. Pydantic schema rejection test
        from backend.schemas import PredictionRequest
        with pytest.raises(ValueError, match="NaN, infinite, or corrupted"):
            PredictionRequest(
                signal=[0.1] * 2047 + [float("nan")],
                sampling_rate_hz=12000,
            )

        with pytest.raises(ValueError, match="NaN, infinite, or corrupted"):
            PredictionRequest(
                signal=[0.1] * 2047 + [float("inf")],
                sampling_rate_hz=12000,
            )

        # 2. HTTP endpoint rejection of literal NaN / Infinity JSON payload
        raw_json = '{"signal": [0.1, NaN], "sampling_rate_hz": 12000}'
        response = client.post(
            "/api/v1/predict",
            content=raw_json,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [400, 422]

    def test_reject_wrong_signal_length(self, client):
        """Verify rejection when signal length cannot produce 2048 samples after resampling."""
        payload = {
            "signal": [0.1] * 500,  # 500 != 2048 for 12 kHz
            "sampling_rate_hz": 12000,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "Invalid signal length" in data["detail"]
        assert "2,048" in data["detail"] or "2048" in data["detail"]

    def test_reject_invalid_sampling_rate(self, client, dummy_12k_signal):
        """Verify rejection of non-positive sampling rate."""
        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": -100,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code in [400, 422]

    def test_reject_invalid_target_class_string(self, client, dummy_12k_signal):
        """Verify 400 Bad Request when an invalid target class name is supplied."""
        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": 12000,
            "explain": True,
            "target_class": "NonExistentDefectClass",
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400
        assert "not found" in response.text.lower()

    def test_predict_with_real_cwru_test_sample(self, client):
        """Verify prediction using an authentic window from cwru_segmented_windows.npz."""
        config = load_config()
        paths = get_resolved_paths(config)
        npz_path = paths["processed_data_dir"] / "cwru_segmented_windows.npz"

        if not npz_path.is_file():
            pytest.skip("cwru_segmented_windows.npz not available.")

        npz = np.load(npz_path)
        real_window = npz["windows"][0].tolist()

        payload = {
            "signal": real_window,
            "sampling_rate_hz": 12000,
            "explain": True,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["predicted_class"] == "Normal"
        assert data["class_id"] == 0
        assert len(data["gradcam_heatmap"]) == 2048
        assert data["processed_signal_length"] == 2048
        assert data["processing_time_ms"] > 0

    def test_startup_missing_checkpoint_raises_error(self, tmp_path, monkeypatch):
        """Verify startup error when model checkpoint is missing."""
        mock_config = load_config()
        mock_config["paths"]["model_checkpoint_dir"] = str(tmp_path)
        monkeypatch.setattr("backend.dependencies.load_config", lambda: mock_config)

        state = AppState()
        with pytest.raises(FileNotFoundError, match="Final model checkpoint not found"):
            load_model_artifacts(state)
        assert state.is_loaded is False

    def test_internal_error_sanitization(self, client, dummy_12k_signal, monkeypatch):
        """Verify unhandled internal errors return generic 500 without leaking stack traces or paths."""
        # Force an unexpected internal exception inside predictor
        def broken_predict(*args, **kwargs):
            raise RuntimeError("Unexpected low-level tensor failure at /secret/path/to/data.pth")

        monkeypatch.setattr(client.app.state.app_state.predictor, "predict_window", broken_predict)

        payload = {
            "signal": dummy_12k_signal,
            "sampling_rate_hz": 12000,
            "explain": False,
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert data["error_type"] == "InternalServerError"
        assert "/secret/path" not in data["detail"]
        assert "traceback" not in data["detail"].lower()
