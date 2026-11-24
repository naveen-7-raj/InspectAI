"""Live local verification script for InspectAI WebSocket streaming and REST endpoints."""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

def main():
    print("=" * 70)
    print("InspectAI Phase 7.1 Real-Time Stream Verification")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Health Endpoint Verification
        health_resp = client.get("/api/v1/health")
        assert health_resp.status_code == 200, f"Health check failed: {health_resp.status_code}"
        health_data = health_resp.json()
        print(f"[1/4] Health Check: Status={health_data['status']}, Model={health_data['model_name']}, Device={health_data['device']}")

        # 2. Synchronous REST Prediction Verification
        dummy_signal = [0.05] * 2048
        pred_resp = client.post("/api/v1/predict", json={"signal": dummy_signal, "sampling_rate_hz": 12000})
        assert pred_resp.status_code == 200, f"Prediction check failed: {pred_resp.status_code}"
        pred_data = pred_resp.json()
        print(f"[2/4] REST Prediction Check: Predicted={pred_data['predicted_class']}, Latency={pred_data['processing_time_ms']} ms")

        # 3. WebSocket Real-Time Replay Verification (Authentic Recording 118: Ball Fault, 5.0x speed)
        print("[3/4] Connecting WebSocket to /api/v1/stream/118?replay_speed=5.0 ...")
        t0 = time.perf_counter()
        with client.websocket_connect("/api/v1/stream/118?replay_speed=5.0") as ws:
            sample_messages = []
            completed_msg = None

            while True:
                try:
                    data = ws.receive_json()
                    if data.get("event") == "prediction":
                        sample_messages.append(data)
                        if len(sample_messages) <= 5:
                            seq = data["sequence_number"]
                            t_start = data["window_start_seconds"]
                            t_end = data["window_end_seconds"]
                            p_class = data["predicted_class"]
                            conf = data["uncalibrated_scores"][p_class]
                            inf_ms = data["inference_time_ms"]
                            print(f"      -> Window {seq:03d} [{t_start:.3f}s - {t_end:.3f}s]: Predicted='{p_class}' | Uncalibrated Softmax={conf:.2%} | Inf Time={inf_ms} ms")
                    elif data.get("event") == "stream_completed":
                        completed_msg = data
                        break
                except Exception as exc:
                    print(f"      Connection closed: {exc}")
                    break

            elapsed = round((time.perf_counter() - t0), 2)
            print(f"[4/4] Stream Completed: Total Windows={completed_msg['total_windows_streamed']}, Elapsed Time={elapsed} s")
            print("=" * 70)
            print("ALL VERIFICATIONS PASSED SUCCESSFULLY.")
            print("=" * 70)

if __name__ == "__main__":
    main()
