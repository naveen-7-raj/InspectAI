"""Unit and integration tests for Simulated Real-Time Vibration Streaming (Phase 7.1)."""

import asyncio
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.dependencies import AppState
from backend.main import app
from src.streaming.simulator import (
    ALLOWED_RECORDING_IDS,
    CWRUStreamSimulator,
    VibrationStreamSimulator,
    normalize_recording_id,
)
from src.utils.config_parser import get_resolved_paths, load_config


@pytest.fixture(scope="module")
def client():
    """Create TestClient with initialized lifespan application state."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sim_97():
    """Create CWRUStreamSimulator instance for recording 97."""
    return CWRUStreamSimulator(recording_id="97")


class TestCWRUStreamSimulator:
    """Test suite verifying CWRU vibration stream simulator mechanics and data integrity."""

    def test_simulator_loads_valid_recording(self, sim_97):
        """1. Verify simulator correctly loads an authentic CWRU recording."""
        assert sim_97.recording_id == "97"
        assert sim_97.file_name == "97.mat"
        assert sim_97.fault_class == "Normal"
        assert sim_97.class_id == 0
        assert sim_97.total_windows > 0
        assert sim_97.target_sampling_rate == 12000
        assert len(sim_97.signal_data) > 2048

    def test_windows_have_exactly_2048_samples(self, sim_97):
        """2. Verify every generated window contains exactly 2048 samples."""
        for i in range(min(10, sim_97.total_windows)):
            window = sim_97.get_window(i)
            assert isinstance(window["samples"], np.ndarray)
            assert len(window["samples"]) == 2048
            assert window["samples"].dtype == np.float32

    def test_consecutive_windows_advance_by_512_samples(self, sim_97):
        """3. Verify consecutive windows advance by 512 samples with 75% overlap."""
        w0 = sim_97.get_window(0)
        w1 = sim_97.get_window(1)
        w2 = sim_97.get_window(2)

        assert w1["window_start_sample"] - w0["window_start_sample"] == 512
        assert w2["window_start_sample"] - w1["window_start_sample"] == 512

        # 75% overlap: last 1536 samples of w0 must match first 1536 samples of w1
        np.testing.assert_array_equal(w0["samples"][512:], w1["samples"][:1536])
        np.testing.assert_array_equal(w1["samples"][512:], w2["samples"][:1536])

    def test_window_timestamps_are_correct(self, sim_97):
        """4. Verify window start and end timestamps match sampling rate calculation."""
        for i in [0, 1, 5, 10]:
            w = sim_97.get_window(i)
            expected_start = round((i * 512) / 12000.0, 5)
            expected_end = round((i * 512 + 2048) / 12000.0, 5)

            assert w["window_start_seconds"] == expected_start
            assert w["window_end_seconds"] == expected_end
            assert np.isclose(w["window_end_seconds"] - w["window_start_seconds"], 2048 / 12000.0, atol=1e-4)

    def test_recording_boundaries_are_respected(self, sim_97):
        """5. Verify window count formula strictly matches signal length without cross-recording leakage."""
        expected_total = ((len(sim_97.signal_data) - 2048) // 512) + 1
        assert sim_97.total_windows == expected_total

        # Out-of-bounds index must raise IndexError
        with pytest.raises(IndexError):
            sim_97.get_window(sim_97.total_windows)

        with pytest.raises(IndexError):
            sim_97.get_window(-1)

    def test_replay_speed_validation(self, sim_97):
        """6. Verify rejection of invalid replay speed multipliers."""
        async def run_invalid_speed():
            async for _ in sim_97.stream_windows(replay_speed=0.0):
                pass

        with pytest.raises(ValueError, match="strictly positive"):
            asyncio.run(run_invalid_speed())

        async def run_negative_speed():
            async for _ in sim_97.stream_windows(replay_speed=-1.5):
                pass

        with pytest.raises(ValueError, match="strictly positive"):
            asyncio.run(run_negative_speed())

    def test_unknown_recording_ids_are_rejected(self):
        """7. Verify rejection of recording IDs not present in authentic CWRU catalog."""
        with pytest.raises(ValueError, match="Unknown CWRU recording ID"):
            CWRUStreamSimulator(recording_id="999")

        with pytest.raises(ValueError, match="Unknown CWRU recording ID"):
            CWRUStreamSimulator(recording_id="random_motor_1")

    def test_normalize_recording_id_helper(self):
        """Verify recording ID normalization removes whitespace and extensions."""
        assert normalize_recording_id("97.mat") == "97"
        assert normalize_recording_id(" 118.mat ") == "118"
        assert normalize_recording_id(185) == "185"

    def test_vibration_stream_simulator_fallback(self):
        """Verify in-memory VibrationStreamSimulator fallback generator."""
        arr = np.arange(4096, dtype=np.float32)
        v_sim = VibrationStreamSimulator(arr, chunk_size=512, window_size=2048)
        assert v_sim.total_windows == ((4096 - 2048) // 512) + 1
        chunks = list(v_sim.stream_chunks())
        assert len(chunks) == v_sim.total_windows
        assert len(chunks[0]) == 2048


class TestWebSocketStreamingRoute:
    """Test suite verifying WebSocket endpoint /api/v1/stream/{recording_id}."""

    def test_websocket_rejects_unknown_recording_id(self, client):
        """Verify WebSocket safely rejects unknown recording ID with 1008 policy violation."""
        with client.websocket_connect("/api/v1/stream/999") as websocket:
            err_msg = websocket.receive_json()
            assert err_msg["event"] == "error"
            assert err_msg["error_type"] == "InvalidRecordingId"
            assert "not recognized" in err_msg["detail"]

    def test_websocket_rejects_invalid_replay_speed(self, client):
        """Verify WebSocket safely rejects invalid replay speed multiplier."""
        with client.websocket_connect("/api/v1/stream/97?replay_speed=-2.0") as websocket:
            err_msg = websocket.receive_json()
            assert err_msg["event"] == "error"
            assert err_msg["error_type"] == "InvalidReplaySpeed"

    def test_websocket_sends_valid_prediction_messages(self, client):
        """8. Verify WebSocket streams structured predictions for authentic CWRU recording."""
        # Use fast replay speed to minimize test execution latency
        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0") as websocket:
            msg = websocket.receive_json()

            assert msg["event"] == "prediction"
            assert msg["source_type"] == "simulated_cwru_replay"
            assert msg["recording_id"] == "97"
            assert msg["sequence_number"] == 0
            assert msg["window_start_seconds"] == 0.0
            assert msg["window_end_seconds"] == pytest.approx(2048 / 12000.0, rel=1e-3)
            assert msg["sampling_rate_hz"] == 12000
            assert msg["predicted_class"] == "Normal"
            assert msg["class_id"] == 0
            assert len(msg["logits"]) == 4
            assert len(msg["uncalibrated_scores"]) == 4
            assert "Normal" in msg["uncalibrated_scores"]
            assert msg["model_name"] == "VibrationCNN1D"
            assert msg["inference_time_ms"] >= 0.0

    def test_websocket_stream_with_gradcam_option(self, client):
        """Verify optional Grad-CAM inclusion when explicitly enabled."""
        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0&include_gradcam=true") as websocket:
            msg = websocket.receive_json()
            assert msg["event"] == "prediction"
            assert "gradcam_heatmap" in msg
            assert len(msg["gradcam_heatmap"]) == 2048
            assert "important_regions" in msg

    def test_websocket_sends_completion_event(self, client, monkeypatch):
        """9. Verify WebSocket sends stream_completed event when all windows finish."""
        # Mock simulator total_windows to 3 to verify completion cleanly without streaming 100+ windows
        original_init = CWRUStreamSimulator.__init__

        def mock_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.total_windows = 3

        monkeypatch.setattr(CWRUStreamSimulator, "__init__", mock_init)

        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0") as websocket:
            messages = []
            while True:
                try:
                    data = websocket.receive_json()
                    messages.append(data)
                    if data.get("event") == "stream_completed":
                        break
                except Exception:
                    break

            assert len(messages) == 4  # 3 predictions + 1 completion
            for i in range(3):
                assert messages[i]["event"] == "prediction"
                assert messages[i]["sequence_number"] == i

            comp = messages[3]
            assert comp["event"] == "stream_completed"
            assert comp["recording_id"] == "97"
            assert comp["total_windows_streamed"] == 3

    def test_client_disconnection_handled_safely(self, client):
        """10. Verify client disconnection mid-stream is handled cleanly without orphan tasks."""
        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0") as websocket:
            msg1 = websocket.receive_json()
            assert msg1["sequence_number"] == 0
            # Close connection abruptly mid-stream
            websocket.close()

    def test_model_checkpoint_not_reloaded_per_window(self, client, monkeypatch):
        """11. Verify model checkpoint is never re-loaded during streaming (zero torch.load calls)."""
        load_count = 0
        original_torch_load = torch.load

        def tracked_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return original_torch_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", tracked_load)

        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0") as websocket:
            for _ in range(3):
                msg = websocket.receive_json()
                assert msg["event"] == "prediction"

        # Model should have been loaded 0 times during stream inference
        assert load_count == 0, "torch.load was called during streaming!"

    def test_existing_model_weights_remain_unchanged(self, client):
        """12. Verify that running streaming inference does not mutate model parameters."""
        model = client.app.state.app_state.model
        weights_before = [p.clone().detach() for p in model.parameters()]

        with client.websocket_connect("/api/v1/stream/97?replay_speed=50.0") as websocket:
            for _ in range(3):
                websocket.receive_json()

        for p_before, p_after in zip(weights_before, model.parameters()):
            assert torch.equal(p_before, p_after), "Model weights mutated during streaming inference!"

