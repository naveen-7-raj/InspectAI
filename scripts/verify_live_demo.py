"""InspectAI Live Application & Demo Verification Script.

Probes all live FastAPI REST endpoints, OpenAPI schemas, and WebSocket streaming
simulation using authentic CWRU recordings.
"""

from pathlib import Path
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from src.ingestion.loader import VibrationDataLoader

DEMO_DIR = PROJECT_ROOT / "docs" / "images" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def verify_live_application():
    print("=" * 75)
    print("InspectAI Live Application & Demo Protocol Verification")
    print("=" * 75)

    with TestClient(app) as client:
        # 1. Health Endpoint Probe
        print("\n[Step 1] Probing Health Endpoint (GET /api/v1/health)...")
        res_health = client.get("/api/v1/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
        health_data = res_health.json()
        print(f"  -> HTTP Status: {res_health.status_code} OK")
        print(f"  -> System Status: {health_data['status']}")
        print(f"  -> Model: {health_data['model_name']} | Device: {health_data['device']} | API Version: {health_data['api_version']}")
        print(f"  -> Supported Classes: {health_data['supported_classes']}")
        print(f"  -> Grad-CAM Active: {health_data['gradcam_available']} | Normalizer Fitted: {health_data['normalizer_fitted']}")
        
        # Save authentic health response payload for documentation
        health_file = DEMO_DIR / "backend_health_response.json"
        with open(health_file, "w", encoding="utf-8") as f:
            json.dump(health_data, f, indent=2)
        print(f"  -> Exported authentic payload to: {health_file.name}")

        # 2. OpenAPI / Swagger Documentation Probe
        print("\n[Step 2] Probing OpenAPI / Swagger Endpoint (GET /openapi.json)...")
        res_docs = client.get("/openapi.json")
        assert res_docs.status_code == 200, f"OpenAPI probe failed: {res_docs.status_code}"
        openapi_spec = res_docs.json()
        print(f"  -> HTTP Status: {res_docs.status_code} OK")
        print(f"  -> Title: {openapi_spec['info']['title']} (v{openapi_spec['info']['version']})")
        print(f"  -> Registered Endpoints: {list(openapi_spec['paths'].keys())}")
        
        spec_file = DEMO_DIR / "openapi_specification.json"
        with open(spec_file, "w", encoding="utf-8") as f:
            json.dump(openapi_spec, f, indent=2)
        print(f"  -> Exported authentic spec to: {spec_file.name}")

        # 3. Synchronous Prediction & Grad-CAM with Authentic CWRU Window
        print("\n[Step 3] Probing Prediction Endpoint (POST /api/v1/predict)...")
        loader = VibrationDataLoader(raw_data_dir=PROJECT_ROOT / "data" / "raw" / "cwru")
        record_118 = loader.load_file("118.mat")
        signal_chunk = record_118.channels["DE_time"][:2048].tolist()
        
        payload = {
            "signal": signal_chunk,
            "sampling_rate_hz": 12000,
            "explain": True,
        }
        t0 = time.perf_counter()
        res_pred = client.post("/api/v1/predict", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        assert res_pred.status_code == 200, f"Prediction failed: {res_pred.status_code}"
        pred_data = res_pred.json()
        print(f"  -> HTTP Status: {res_pred.status_code} OK")
        print(f"  -> Signal Input: 2,048 samples from '118.mat' (Ball Fault, 12 kHz)")
        print(f"  -> Predicted Condition: '{pred_data['predicted_class']}' (Class ID: {pred_data['class_id']})")
        print(f"  -> Class Probabilities (Uncalibrated Softmax): {pred_data['class_probabilities']}")
        print(f"  -> Pipeline Processing Time: {pred_data['processing_time_ms']:.2f} ms (Total roundtrip: {latency_ms:.2f} ms)")
        print(f"  -> Grad-CAM Saliency Included: {pred_data['gradcam_heatmap'] is not None} (Length: {len(pred_data['gradcam_heatmap']) if pred_data['gradcam_heatmap'] else 0})")
        print(f"  -> Target Class Explained: '{pred_data['target_class']}'")
        print(f"  -> High Saliency Intervals: {len(pred_data.get('important_regions', []) or [])} regions identified")
        
        pred_file = DEMO_DIR / "prediction_response_118.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(pred_data, f, indent=2)
        print(f"  -> Exported authentic payload to: {pred_file.name}")

        # 4. WebSocket Simulated Streaming Probe (Using documented supported speed: 5.0x)
        print("\n[Step 4] Probing WebSocket Streaming Gateway (WS /api/v1/stream/118?replay_speed=5.0)...")
        with client.websocket_connect("/api/v1/stream/118?replay_speed=5.0&include_waveform=true&include_gradcam=true") as ws:
            stream_packets = []
            while True:
                msg = ws.receive_json()
                if msg.get("event") == "prediction":
                    stream_packets.append(msg)
                    if len(stream_packets) == 1:
                        print(f"  -> Connected! First packet received:")
                        print(f"     Window {msg['sequence_number']}: Class='{msg['predicted_class']}' | Latency={msg['inference_time_ms']} ms | Softmax={msg['uncalibrated_scores']}")
                        sample_packet_file = DEMO_DIR / "stream_packet_sample.json"
                        with open(sample_packet_file, "w", encoding="utf-8") as f:
                            json.dump(msg, f, indent=2)
                        print(f"  -> Exported authentic streaming packet to: {sample_packet_file.name}")
                elif msg.get("event") == "stream_completed":
                    print(f"  -> Stream Completed Successfully! Total Windows: {msg['total_windows_streamed']}")
                    break

        print("\n" + "=" * 75)
        print("DEMO VERIFICATION RESULT: ALL APPLICATION PROTOCOLS VERIFIED LIVE")
        print("=" * 75)


if __name__ == "__main__":
    verify_live_application()
