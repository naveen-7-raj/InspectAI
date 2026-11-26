"""Pydantic request and response schemas for InspectAI Prediction Backend."""

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Vibration signal inference and optional explanation request."""

    signal: List[float] = Field(
        ...,
        description="Raw vibration acceleration samples in physical units (g).",
        min_length=1,
    )
    sampling_rate_hz: int = Field(
        ...,
        gt=0,
        description="Nominal sampling rate in Hz (e.g. 12000 or 48000).",
        examples=[12000, 48000],
    )
    explain: bool = Field(
        default=False,
        description="Whether to generate 1D Grad-CAM attribution heatmap.",
    )
    target_class: Optional[Union[int, str]] = Field(
        default=None,
        description="Optional class ID (0-3) or class name to explain. Defaults to predicted class.",
        examples=[None, "Ball Fault", 1],
    )
    inference_mode: str = Field(
        default="baseline",
        description="Inference mode: 'baseline' (unadapted frozen model) or 'adaptive' (CORAL domain adaptation).",
    )

    @field_validator("signal")
    @classmethod
    def validate_signal_values(cls, v: List[float]) -> List[float]:
        """Verify signal contains only valid, finite numeric samples."""
        if not v:
            raise ValueError("Vibration signal cannot be empty.")

        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Vibration signal contains NaN, infinite, or corrupted values.")

        return v

    @field_validator("sampling_rate_hz")
    @classmethod
    def validate_sampling_rate(cls, v: int) -> int:
        """Verify sampling rate is a recognized non-zero rate."""
        if v <= 0:
            raise ValueError("Sampling rate must be a strictly positive integer.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "signal": [0.012, -0.045, 0.089, -0.015] * 512,  # 2048 samples
                "sampling_rate_hz": 12000,
                "explain": True,
                "target_class": None,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Detailed fault prediction and XAI response container."""

    predicted_class: str = Field(..., description="Predicted bearing fault category name.")
    class_id: int = Field(..., ge=0, le=3, description="Predicted class index (0-3).")
    logits: List[float] = Field(..., description="Raw unnormalized model output logits.")
    probabilities: List[float] = Field(
        ...,
        description="Uncalibrated model softmax scores across classes (Normal, Ball, Inner, Outer).",
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Mapping from class name to uncalibrated softmax score.",
    )
    confidence_note: str = Field(
        ...,
        description="Important notice stating softmax scores are uncalibrated and not empirical probabilities.",
    )
    input_sampling_rate_hz: int = Field(..., description="Raw input sampling frequency in Hz.")
    processed_sampling_rate_hz: int = Field(..., description="Sampling frequency after normalization (12000 Hz).")
    processed_signal_length: int = Field(..., description="Number of samples after preprocessing (2048).")
    model_name: str = Field(..., description="Underlying CNN architecture name.")
    processing_time_ms: float = Field(..., description="Total pipeline execution time in milliseconds.")
    gradcam_heatmap: Optional[List[float]] = Field(
        default=None,
        description="1D Grad-CAM attribution scores [0.0, 1.0] across 2048 samples if explain=True.",
    )
    target_class: Optional[str] = Field(
        default=None,
        description="Class label targeted by Grad-CAM attribution.",
    )
    important_regions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Detected contiguous temporal intervals where attribution exceeds threshold.",
    )
    inference_mode: str = Field(
        default="baseline",
        description="Executed mode: 'baseline' or 'adaptive'.",
    )
    is_adapted: bool = Field(
        default=False,
        description="Whether domain adaptation was applied.",
    )
    adapter_telemetry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic telemetry from domain adapter.",
    )
    baseline_predicted_class: Optional[str] = Field(
        default=None,
        description="Unadapted baseline class prediction.",
    )
    baseline_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Unadapted baseline softmax probabilities.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "Ball Fault",
                "class_id": 1,
                "logits": [-2.15, 4.32, -0.85, -1.32],
                "probabilities": [0.0015, 0.9852, 0.0057, 0.0076],
                "class_probabilities": {
                    "Normal": 0.0015,
                    "Ball Fault": 0.9852,
                    "Inner Race Fault": 0.0057,
                    "Outer Race Fault": 0.0076,
                },
                "confidence_note": "Softmax scores are uncalibrated model outputs and do not represent true empirical probabilities.",
                "input_sampling_rate_hz": 12000,
                "processed_sampling_rate_hz": 12000,
                "processed_signal_length": 2048,
                "model_name": "VibrationCNN1D",
                "processing_time_ms": 12.45,
                "gradcam_heatmap": None,
                "target_class": None,
                "important_regions": None,
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check and model readiness status response."""

    status: str = Field(..., description="Service health status ('healthy' or 'degraded').")
    model_loaded: bool = Field(..., description="Whether the frozen 1D CNN checkpoint is loaded and ready.")
    model_name: str = Field(..., description="Model architecture identifier.")
    supported_classes: List[str] = Field(..., description="List of recognized bearing fault categories.")
    device: str = Field(..., description="Inference execution device ('cpu').")
    gradcam_available: bool = Field(..., description="Whether 1D Grad-CAM explainability is active.")
    normalizer_fitted: bool = Field(..., description="Whether signal normalizer is initialized.")
    api_version: str = Field(..., description="InspectAI API release version.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "VibrationCNN1D",
                "supported_classes": ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"],
                "device": "cpu",
                "gradcam_available": True,
                "normalizer_fitted": True,
                "api_version": "0.1.0",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standardized API error message container."""

    detail: str = Field(..., description="Human-readable explanation of error.")
    error_type: str = Field(..., description="Category or class of error.")

