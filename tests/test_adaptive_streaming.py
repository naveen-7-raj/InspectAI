"""Integration tests for production-ready adaptive streaming pipeline (Phase 18).

Verifies:
1. StreamingPredictor adaptive and baseline mode execution.
2. extract_features contract (576-dim fused feature vector).
3. Graceful fallback when adapter is not calibrated.
4. FastAPI REST /predict endpoint with inference_mode='adaptive'.
5. FastAPI WebSocket /stream/{recording_id} with adaptive & calibration query parameters.
6. Absolute immutability of frozen model parameters throughout inference.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.dependencies import AppState, load_model_artifacts
from backend.main import app
from src.inference.domain_adapter import AdapterStatus, DomainAdapter
from src.inference.predictor import StreamingPredictor


@pytest.fixture(scope="module")
def candidate_predictor():
    """Load candidate release model with attached DomainAdapter."""
    p = StreamingPredictor.from_final_model(model_dir="models/final", device="cpu")
    cwru_path = Path("data/processed/cwru_segmented_windows.npz")
    if cwru_path.is_file():
        adapter = DomainAdapter.from_processed_cwru(
            cwru_windows_path=cwru_path,
            predictor=p,
            min_calibration_samples=10,
        )
        p.domain_adapter = adapter
    return p


class TestStreamingPredictorAdaptiveInference:
    """Verify adaptive inference execution in StreamingPredictor."""

    def test_extract_features_shape(self, candidate_predictor):
        dummy_win = np.random.randn(2048).astype(np.float32)
        feat = candidate_predictor.extract_features(dummy_win)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (576,)
        assert np.all(np.isfinite(feat))

    def test_predict_window_baseline_mode(self, candidate_predictor):
        dummy_win = np.random.randn(2048).astype(np.float32)
        res = candidate_predictor.predict_window(dummy_win, mode="baseline")
        assert res["inference_mode"] == "baseline"
        assert res["is_adapted"] is False
        assert "predicted_class" in res
        assert "class_probabilities" in res

    def test_predict_window_adaptive_uncalibrated_fallback(self, candidate_predictor):
        # Reset adapter so it is uncalibrated
        if candidate_predictor.domain_adapter is not None:
            candidate_predictor.domain_adapter.reset()

        dummy_win = np.random.randn(2048).astype(np.float32)
        res = candidate_predictor.predict_window(dummy_win, mode="adaptive")

        assert res["inference_mode"] == "adaptive_fallback"
        assert res["is_adapted"] is False
        assert "baseline_predicted_class" in res
        assert res["adapter_telemetry"]["status"] == "CALIBRATING"

    def test_predict_window_adaptive_calibrated_success(self, candidate_predictor):
        adapter = candidate_predictor.domain_adapter
        assert adapter is not None

        # Calibrate with synthetic target features
        dummy_calib = np.random.randn(20, 576).astype(np.float32) * 1.5 + 0.5
        adapter.collect_calibration_batch(dummy_calib)
        success = adapter.fit()
        assert success is True
        assert adapter.is_fitted is True

        dummy_win = np.random.randn(2048).astype(np.float32)
        res = candidate_predictor.predict_window(dummy_win, mode="adaptive")

        assert res["inference_mode"] == "adaptive"
        assert res["is_adapted"] is True
        assert "baseline_predicted_class" in res
        assert "baseline_probabilities" in res
        assert res["adapter_telemetry"]["status"] == "READY"

    def test_frozen_weights_immutability(self, candidate_predictor):
        # Hash initial weights
        initial_weights = [p.clone().detach().cpu().numpy() for p in candidate_predictor.model.parameters()]

        dummy_win = np.random.randn(2048).astype(np.float32)
        for _ in range(5):
            candidate_predictor.predict_window(dummy_win, mode="adaptive")
            candidate_predictor.predict_window(dummy_win, mode="baseline")

        # Verify weights have not changed
        for init_w, curr_p in zip(initial_weights, candidate_predictor.model.parameters()):
            curr_w = curr_p.detach().cpu().numpy()
            np.testing.assert_array_equal(init_w, curr_w)


class TestFastAPIAdaptiveEndpoints:
    """Verify adaptive streaming and prediction across FastAPI routes."""

    @pytest.fixture
    def test_client(self, monkeypatch):
        monkeypatch.setenv("INSPECTAI_USE_FINAL_MODEL", "1")
        state = AppState()
        load_model_artifacts(state)
        app.state.app_state = state
        with TestClient(app) as client:
            yield client
        # Clean up environment and restore default state
        monkeypatch.delenv("INSPECTAI_USE_FINAL_MODEL", raising=False)
        default_state = AppState()
        try:
            load_model_artifacts(default_state)
        except Exception:
            pass
        app.state.app_state = default_state

    def test_rest_predict_with_adaptive_mode(self, test_client):
        signal = [0.01, -0.02, 0.03, -0.01] * 512
        payload = {
            "signal": signal,
            "sampling_rate_hz": 12000,
            "explain": False,
            "inference_mode": "adaptive",
        }
        resp = test_client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_class" in data
        assert "inference_mode" in data
        assert data["inference_mode"] in ("adaptive", "adaptive_fallback")
        assert "is_adapted" in data
        assert "confidence_note" in data

    def test_websocket_stream_with_adaptive_parameters(self, test_client):
        # Connect to WebSocket streaming route with adaptive parameters
        url = "/api/v1/stream/97?replay_speed=50.0&inference_mode=adaptive&calibrate_first=true"
        with test_client.websocket_connect(url) as ws:
            first_msg = ws.receive_json()
            assert first_msg["event"] == "prediction"
            assert "predicted_class" in first_msg
            assert "inference_mode" in first_msg
            assert "is_adapted" in first_msg
            assert "adapter_telemetry" in first_msg
            assert first_msg["model_name"] in ("SpectralFusionCNN", "VibrationCNN1D")
