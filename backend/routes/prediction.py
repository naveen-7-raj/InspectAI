"""Vibration fault prediction and Explainable AI (1D Grad-CAM) endpoints."""

import logging
import time
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import AppState, get_app_state
from backend.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger("InspectAI.PredictionRoute")
router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Vibration Signal & Generate Explanations",
    description=(
        "Preprocesses raw continuous accelerometer vibration readings (resampling to 12 kHz, "
        "removing DC offset, and bandpass filtering), runs inference through the frozen 1D CNN, "
        "and optionally computes 1D Grad-CAM temporal attributions."
    ),
    responses={
        200: {"description": "Successful classification and attribution output."},
        400: {"description": "Invalid input signal, corrupt values, or incompatible length."},
        503: {"description": "Prediction service unavailable because models are not loaded."},
    },
)
def predict_vibration(
    payload: PredictionRequest,
    state: AppState = Depends(get_app_state),
) -> PredictionResponse:
    """Preprocess vibration signal and execute 1D CNN fault diagnosis."""
    if not state.is_loaded or state.predictor is None or state.signal_cleaner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction engine is not ready. Model artifacts are not loaded.",
        )

    start_time = time.perf_counter()

    # 1. Validate signal length compatibility with target window (2,048 samples at 12 kHz)
    fs_in = payload.sampling_rate_hz
    fs_target = state.signal_cleaner.target_sampling_rate
    expected_input_length = int(round(2048 * (fs_in / float(fs_target))))

    if len(payload.signal) != expected_input_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid signal length ({len(payload.signal)} samples) for sampling rate {fs_in} Hz. "
                f"Expected exactly {expected_input_length} samples to yield a single 2,048-sample "
                f"window after normalization to {fs_target} Hz."
            ),
        )

    # 2. Digital Signal Preprocessing Pipeline
    try:
        processed = state.signal_cleaner.clean_signal(
            signal=payload.signal,
            orig_fs=fs_in,
            target_fs=fs_target,
        )
    except Exception as exc:
        logger.error(f"Preprocessing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signal preprocessing failed: {str(exc)}",
        )

    processed_samples = len(processed.signal)
    if processed_samples != 2048:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessed signal produced {processed_samples} samples; expected exactly 2048.",
        )

    # 3. Model Inference and Optional 1D Grad-CAM Explanation
    gradcam_heatmap = None
    target_class_label = None
    important_regions = None

    try:
        if payload.explain:
            if state.explainer is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="1D Grad-CAM explainer is not available on this server.",
                )

            exp_result = state.explainer.explain_window(
                vibration_window=processed.signal,
                target_class=payload.target_class,
                is_normalized=False,
                sampling_rate_hz=fs_target,
            )

            pred_class = exp_result["predicted_class_name"]
            pred_id = exp_result["predicted_class_id"]
            raw_logits = exp_result["logits"]
            probs = exp_result["probabilities"]
            gradcam_heatmap = exp_result["gradcam_heatmap"].tolist()
            target_class_label = exp_result["target_class_name"]
            important_regions = exp_result["important_regions"]
            class_prob_map = {
                name: round(probs[i], 6) for i, name in enumerate(state.class_names)
            }

        else:
            pred_result = state.predictor.predict_window(
                window=processed.signal,
                is_normalized=False,
            )

            pred_class = pred_result["predicted_class"]
            pred_id = pred_result["class_id"]
            raw_logits = pred_result["logits"]
            probs = pred_result["probabilities"]
            class_prob_map = pred_result["class_probabilities"]

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Inference execution failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference execution failed.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return PredictionResponse(
        predicted_class=pred_class,
        class_id=pred_id,
        logits=raw_logits,
        probabilities=probs,
        class_probabilities=class_prob_map,
        confidence_note="Softmax scores are uncalibrated model outputs and do not represent true empirical probabilities.",
        input_sampling_rate_hz=fs_in,
        processed_sampling_rate_hz=fs_target,
        processed_signal_length=processed_samples,
        model_name="VibrationCNN1D",
        processing_time_ms=elapsed_ms,
        gradcam_heatmap=gradcam_heatmap,
        target_class=target_class_label,
        important_regions=important_regions,
    )

