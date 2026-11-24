"""Unit and integration tests for Streamlit Monitoring Dashboard (Phase 8.1)."""

from unittest.mock import MagicMock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests

from frontend.app import (
    CWRU_OPTIONS,
    CLASS_COLOR_MAP,
    build_websocket_url,
    fetch_backend_health,
    parse_stream_message,
    render_gradcam_figure,
    render_scores_figure,
    render_waveform_figure,
    update_prediction_history,
)


class TestDashboardModule:
    """Test suite verifying Streamlit helper functions, network utilities, and charting pipelines."""

    def test_dashboard_imports_and_constants(self):
        """1. Verify dashboard module imports cleanly and contains the 13 authentic CWRU entries."""
        assert len(CWRU_OPTIONS) == 13
        assert "97" in CWRU_OPTIONS
        assert "118" in CWRU_OPTIONS
        assert "169" in CWRU_OPTIONS
        assert "197" in CWRU_OPTIONS
        assert set(CLASS_COLOR_MAP.keys()) == {
            "Normal",
            "Ball Fault",
            "Inner Race Fault",
            "Outer Race Fault",
        }

    def test_fetch_backend_health_success(self, monkeypatch):
        """2. Verify health checking when backend is online and model is loaded."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "model_name": "VibrationCNN1D",
            "device": "cpu",
        }
        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        result = fetch_backend_health("http://localhost:8000")
        assert result["connected"] is True
        assert result["error"] is None
        assert result["data"]["status"] == "healthy"
        assert result["data"]["model_loaded"] is True

    def test_fetch_backend_health_degraded(self, monkeypatch):
        """3. Verify health checking when backend is online but returns 503 degraded status."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {
            "status": "degraded",
            "model_loaded": False,
        }
        monkeypatch.setattr(requests, "get", lambda url, timeout: mock_response)

        result = fetch_backend_health("http://localhost:8000")
        assert result["connected"] is True
        assert "503" in result["error"]
        assert result["data"]["model_loaded"] is False

    def test_fetch_backend_health_connection_failure(self, monkeypatch):
        """4. Verify safe error handling when backend is unreachable without crashing."""
        def mock_get_fail(url, timeout):
            raise requests.exceptions.ConnectionError("Failed to establish a new connection")

        monkeypatch.setattr(requests, "get", mock_get_fail)

        result = fetch_backend_health("http://invalid-backend-host:8000")
        assert result["connected"] is False
        assert result["data"] is None
        assert "Failed to establish a new connection" in result["error"]

    def test_build_websocket_url_http_and_https(self):
        """5. Verify proper conversion of HTTP/HTTPS URLs to WS/WSS endpoints with query parameters."""
        url_http = build_websocket_url(
            base_url="http://localhost:8000",
            recording_id="97",
            replay_speed=1.0,
            include_gradcam=False,
            include_waveform=True,
        )
        assert url_http == "ws://localhost:8000/api/v1/stream/97?replay_speed=1.0&include_gradcam=false&include_waveform=true"

        url_https = build_websocket_url(
            base_url="https://remote-inspectai.org/api/",
            recording_id="118.mat",
            replay_speed=2.0,
            include_gradcam=True,
            include_waveform=False,
        )
        assert url_https == "wss://remote-inspectai.org/api/api/v1/stream/118?replay_speed=2.0&include_gradcam=true&include_waveform=false"

    def test_parse_valid_stream_message(self):
        """6. Verify parsing of standard streaming prediction messages."""
        msg_dict = {
            "event": "prediction",
            "source_type": "simulated_cwru_replay",
            "recording_id": "97",
            "sequence_number": 12,
            "predicted_class": "Normal",
            "uncalibrated_scores": {"Normal": 0.999},
        }
        parsed = parse_stream_message(msg_dict)
        assert parsed["event"] == "prediction"
        assert parsed["sequence_number"] == 12

        # As JSON string
        import json
        raw_json = json.dumps(msg_dict)
        parsed_str = parse_stream_message(raw_json)
        assert parsed_str["event"] == "prediction"
        assert parsed_str["recording_id"] == "97"

    def test_parse_invalid_or_malformed_stream_message(self):
        """7. Verify safe handling of malformed or non-JSON WebSocket payloads."""
        malformed = "{'invalid': json"
        res = parse_stream_message(malformed)
        assert res["event"] == "error"
        assert res["error_type"] == "JSONDecodeError"

        bad_type = [1, 2, 3]  # List instead of dict or string
        res_type = parse_stream_message(bad_type)
        assert res_type["event"] == "error"
        assert res_type["error_type"] == "InvalidMessageType"

    def test_update_prediction_history_cap(self):
        """8. Verify prediction history capping at max_length (FIFO)."""
        history = []
        for i in range(70):
            entry = {"seq": i, "pred": "Normal"}
            history = update_prediction_history(history, entry, max_length=50)

        assert len(history) == 50
        assert history[0]["seq"] == 69  # Most recent first
        assert history[-1]["seq"] == 20

    def test_render_waveform_figure(self):
        """9. Verify vibration waveform figure generation and cleanup."""
        samples = np.sin(np.linspace(0, 20 * np.pi, 2048)).astype(np.float32)
        fig = render_waveform_figure(
            samples=samples,
            recording_id="97",
            sequence_number=5,
            sampling_rate_hz=12000,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_render_scores_figure(self):
        """10. Verify uncalibrated softmax scores horizontal bar chart generation."""
        scores = {
            "Normal": 0.05,
            "Ball Fault": 0.90,
            "Inner Race Fault": 0.03,
            "Outer Race Fault": 0.02,
        }
        fig = render_scores_figure(scores)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_render_gradcam_figure(self):
        """11. Verify 1D Grad-CAM 2-panel figure generation and formatting."""
        samples = np.sin(np.linspace(0, 20 * np.pi, 2048)).astype(np.float32)
        heatmap = np.clip(np.random.rand(2048), 0.0, 1.0).astype(np.float32)
        fig = render_gradcam_figure(
            samples=samples,
            heatmap=heatmap,
            predicted_class="Ball Fault",
            sampling_rate_hz=12000,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_no_synthetic_data_guarantee(self):
        """12. Verify that all 13 recording options strictly correspond to authentic CWRU files."""
        expected_ids = {
            "97", "98", "99", "100",
            "105", "118", "130", "169",
            "185", "197", "209", "222", "234",
        }
        assert set(CWRU_OPTIONS.keys()) == expected_ids

