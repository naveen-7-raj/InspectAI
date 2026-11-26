"""WebSocket endpoint for simulated real-time vibration streaming and window prediction."""

import logging
import time
from typing import Optional
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from backend.dependencies import AppState
from src.monitoring.metrics import (
    record_adapter_telemetry,
    record_api_error,
    record_prediction,
    record_websocket_connection,
)
from src.streaming.simulator import (
    ALLOWED_RECORDING_IDS,
    CWRUStreamSimulator,
    normalize_recording_id,
)

logger = logging.getLogger("InspectAI.StreamingRoute")
router = APIRouter(tags=["Streaming"])


@router.websocket("/stream/{recording_id}")
async def stream_vibration(
    websocket: WebSocket,
    recording_id: str,
    replay_speed: float = Query(
        default=1.0,
        description="Replay speed multiplier (e.g. 0.5, 1.0, 2.0, 5.0).",
    ),
    include_gradcam: bool = Query(
        default=False,
        description="Whether to compute 1D Grad-CAM attribution for every streamed window (disabled by default).",
    ),
    include_waveform: bool = Query(
        default=True,
        description="Whether to include authentic 2,048 preprocessed window samples in prediction packet.",
    ),
    inference_mode: str = Query(
        default="baseline",
        description="Inference mode: 'baseline' (unadapted frozen model) or 'adaptive' (CORAL domain adaptation).",
    ),
    calibrate_first: bool = Query(
        default=False,
        description="Whether to buffer initial unlabelled streamed windows to calibrate domain adapter.",
    ),
) -> None:
    """Stream simulated real-time CWRU vibration windows and CNN predictions via WebSocket.

    IMPORTANT:
    This endpoint streams pre-recorded laboratory vibration data from the CWRU bearing dataset.
    It simulates an edge-sensor deployment with realistic sampling intervals (42.67 ms at 1x speed).
    It does not connect to a physical live motor or sensor.
    """
    await websocket.accept()
    record_websocket_connection(+1)

    # 1. Validate Recording Identifier
    clean_id = normalize_recording_id(recording_id)
    if clean_id not in ALLOWED_RECORDING_IDS:
        logger.warning(f"Rejected stream connection for unknown recording ID: '{recording_id}'.")
        await websocket.send_json({
            "event": "error",
            "error_type": "InvalidRecordingId",
            "detail": (
                f"Recording ID '{recording_id}' is not recognized. "
                f"Allowed CWRU recording IDs are: {', '.join(ALLOWED_RECORDING_IDS)}."
            ),
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Validate Replay Speed Multiplier
    if replay_speed <= 0 or replay_speed > 50.0:
        logger.warning(f"Rejected stream connection for invalid replay speed: {replay_speed}.")
        await websocket.send_json({
            "event": "error",
            "error_type": "InvalidReplaySpeed",
            "detail": f"Invalid replay speed ({replay_speed}). Speed must be a positive number (e.g. 0.5, 1.0, 2.0, 5.0).",
        })
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Verify Model Readiness
    state: AppState = getattr(websocket.app.state, "app_state", None)
    if state is None or not state.is_loaded or state.predictor is None:
        logger.error("Streaming connection rejected: InspectAI model artifacts not loaded.")
        await websocket.send_json({
            "event": "error",
            "error_type": "ServiceUnavailable",
            "detail": "InspectAI prediction engine is not ready. Model artifacts are not loaded.",
        })
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # 4. Initialize CWRU Stream Simulator
    try:
        simulator = CWRUStreamSimulator(
            recording_id=clean_id,
            config=state.config,
        )
    except Exception as exc:
        logger.exception(f"Failed to initialize simulator for recording {clean_id}: {exc}")
        await websocket.send_json({
            "event": "error",
            "error_type": "InitializationError",
            "detail": "Failed to load and initialize raw vibration recording.",
        })
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    logger.info(
        f"WebSocket streaming initialized for recording {clean_id} "
        f"({simulator.total_windows} windows, speed={replay_speed}x, gradcam={include_gradcam})."
    )

    # 5. Stream Sequential Windows
    adapter = getattr(state, "domain_adapter", None)
    if adapter is not None and calibrate_first:
        adapter.reset()

    try:
        async for window_data in simulator.stream_windows(replay_speed=replay_speed):
            t0 = time.perf_counter()
            samples = window_data["samples"]

            # If calibrating first, collect unlabelled feature embedding
            if calibrate_first and adapter is not None and not adapter.is_fitted:
                try:
                    feat = state.predictor.extract_features(samples, is_normalized=False)
                    calib_info = adapter.collect_calibration_sample(feat)
                    if calib_info["is_ready_to_fit"]:
                        adapter.fit()
                except Exception as exc:
                    logger.warning(f"Error during streaming calibration sample collection: {exc}")

                pred = state.predictor.predict_window(
                    window=samples,
                    is_normalized=False,
                    mode="baseline",
                )
                current_mode = "calibrating" if not adapter.is_fitted else "adaptive"
            else:
                pred = state.predictor.predict_window(
                    window=samples,
                    is_normalized=False,
                    mode=inference_mode,
                )
                current_mode = pred.get("inference_mode", inference_mode)

            inf_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            elapsed_sec = time.perf_counter() - t0
            record_prediction(
                predicted_class=pred["predicted_class"],
                inference_mode=current_mode,
                is_adapted=pred.get("is_adapted", False),
                latency_seconds=elapsed_sec,
            )
            if pred.get("adapter_telemetry"):
                record_adapter_telemetry(
                    condition_number=pred["adapter_telemetry"].get("condition_number"),
                    drift_frobenius=pred["adapter_telemetry"].get("drift_frobenius"),
                )

            msg = {
                "event": "prediction",
                "source_type": "simulated_cwru_replay",
                "recording_id": clean_id,
                "sequence_number": window_data["sequence_number"],
                "window_start_seconds": window_data["window_start_seconds"],
                "window_end_seconds": window_data["window_end_seconds"],
                "sampling_rate_hz": window_data["sampling_rate_hz"],
                "predicted_class": pred["predicted_class"],
                "class_id": pred["class_id"],
                "logits": pred["logits"],
                "uncalibrated_scores": pred["class_probabilities"],
                "inference_mode": current_mode,
                "is_adapted": pred.get("is_adapted", False),
                "baseline_predicted_class": pred.get("baseline_predicted_class", pred["predicted_class"]),
                "baseline_scores": pred.get("baseline_probabilities", pred["class_probabilities"]),
                "adapter_telemetry": pred.get("adapter_telemetry"),
                "model_name": "SpectralFusionCNN" if getattr(state.predictor, "is_dual_branch", False) else "VibrationCNN1D",
                "inference_time_ms": inf_time_ms,
            }

            # Optional authentic window samples (enabled by default for dashboard visualization)
            if include_waveform:
                msg["waveform"] = window_data["samples"].tolist()

            # Optional Grad-CAM explanation (disabled by default for low CPU latency)
            if include_gradcam and state.explainer is not None:
                exp = state.explainer.explain_window(
                    vibration_window=window_data["samples"],
                    is_normalized=False,
                    sampling_rate_hz=12000,
                )
                msg["gradcam_heatmap"] = exp["gradcam_heatmap"].tolist()
                msg["important_regions"] = exp["important_regions"]

            await websocket.send_json(msg)

        # 6. Stream Completion Event
        logger.info(f"Completed streaming all {simulator.total_windows} windows for recording {clean_id}.")
        await websocket.send_json({
            "event": "stream_completed",
            "recording_id": clean_id,
            "total_windows_streamed": simulator.total_windows,
        })
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected cleanly from stream for recording {clean_id}.")
    except Exception as exc:
        record_api_error("/api/v1/stream", type(exc).__name__)
        logger.exception(f"Unexpected streaming error for recording {clean_id}: {exc}")
        try:
            await websocket.send_json({
                "event": "error",
                "error_type": "StreamError",
                "detail": "An internal error occurred during vibration streaming.",
            })
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        record_websocket_connection(-1)

