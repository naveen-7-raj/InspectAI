"""InspectAI Observability & Prometheus Metrics Infrastructure (Phase 19).

Provides lightweight Prometheus metrics exposition for the InspectAI prediction and streaming backend.
Tracks:
1. Prediction volume and classification distributions by inference mode.
2. Latency histograms for baseline vs adaptive execution.
3. DomainAdapter calibration events, duration, and covariance condition numbers.
4. Domain drift measurements (Frobenius distance between source and target covariance).
5. Active WebSocket client connections and error occurrences.
6. Shadow recalibrations and rollback operations.
"""

from typing import Optional, Tuple
import logging
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("InspectAI.Monitoring")

# Shared Prometheus Registry for InspectAI
REGISTRY = CollectorRegistry(auto_describe=True)

# 1. Prediction Counters & Latency
PREDICTIONS_TOTAL = Counter(
    "inspectai_predictions_total",
    "Total predictions processed by InspectAI",
    ["predicted_class", "inference_mode", "is_adapted"],
    registry=REGISTRY,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inspectai_inference_latency_seconds",
    "Inference latency in seconds",
    ["inference_mode"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

# 2. Domain Adapter & Calibration Metrics
CALIBRATION_TOTAL = Counter(
    "inspectai_calibration_total",
    "Total adapter calibration events and outcomes",
    ["status"],
    registry=REGISTRY,
)

CALIBRATION_DURATION_SECONDS = Gauge(
    "inspectai_calibration_duration_seconds",
    "Duration of most recent domain adapter calibration in seconds",
    registry=REGISTRY,
)

ADAPTER_CONDITION_NUMBER = Gauge(
    "inspectai_adapter_condition_number",
    "Condition number of the target feature covariance matrix",
    registry=REGISTRY,
)

DOMAIN_DRIFT_FROBENIUS = Gauge(
    "inspectai_domain_drift_frobenius",
    "Frobenius norm distance between source and target covariance matrices",
    registry=REGISTRY,
)

# 3. Connection Health & API Errors
API_ERRORS_TOTAL = Counter(
    "inspectai_api_errors_total",
    "Total backend API and streaming errors",
    ["endpoint", "error_type"],
    registry=REGISTRY,
)

ACTIVE_WEBSOCKETS = Gauge(
    "inspectai_active_websockets",
    "Current number of active streaming WebSocket connections",
    registry=REGISTRY,
)

# 4. Drift & Recalibration Lifecycle Metrics
RECALIBRATIONS_TOTAL = Counter(
    "inspectai_recalibrations_total",
    "Shadow recalibration events executed by drift monitor",
    ["trigger", "outcome"],
    registry=REGISTRY,
)

ROLLBACKS_TOTAL = Counter(
    "inspectai_rollbacks_total",
    "Total rollback events to previous adapter or baseline due to instability",
    ["reason"],
    registry=REGISTRY,
)


def record_prediction(
    predicted_class: str,
    inference_mode: str,
    is_adapted: bool,
    latency_seconds: float,
) -> None:
    """Record single inference prediction event and latency."""
    try:
        PREDICTIONS_TOTAL.labels(
            predicted_class=str(predicted_class),
            inference_mode=str(inference_mode),
            is_adapted="true" if is_adapted else "false",
        ).inc()
        INFERENCE_LATENCY_SECONDS.labels(
            inference_mode=str(inference_mode),
        ).observe(max(0.0, float(latency_seconds)))
    except Exception as exc:
        logger.debug(f"Metrics record_prediction failed: {exc}")


def record_calibration(status: str, duration_seconds: Optional[float] = None) -> None:
    """Record calibration event and optional computation duration."""
    try:
        CALIBRATION_TOTAL.labels(status=str(status)).inc()
        if duration_seconds is not None:
            CALIBRATION_DURATION_SECONDS.set(max(0.0, float(duration_seconds)))
    except Exception as exc:
        logger.debug(f"Metrics record_calibration failed: {exc}")


def record_adapter_telemetry(
    condition_number: Optional[float],
    drift_frobenius: Optional[float] = None,
) -> None:
    """Update adapter numerical health gauges."""
    try:
        if condition_number is not None and condition_number > 0:
            ADAPTER_CONDITION_NUMBER.set(float(condition_number))
        if drift_frobenius is not None and drift_frobenius >= 0:
            DOMAIN_DRIFT_FROBENIUS.set(float(drift_frobenius))
    except Exception as exc:
        logger.debug(f"Metrics record_adapter_telemetry failed: {exc}")


def record_api_error(endpoint: str, error_type: str) -> None:
    """Record API or WebSocket exception."""
    try:
        API_ERRORS_TOTAL.labels(
            endpoint=str(endpoint),
            error_type=str(error_type),
        ).inc()
    except Exception as exc:
        logger.debug(f"Metrics record_api_error failed: {exc}")


def record_websocket_connection(delta: int) -> None:
    """Increment or decrement active WebSocket counter."""
    try:
        if delta > 0:
            ACTIVE_WEBSOCKETS.inc(delta)
        elif delta < 0:
            ACTIVE_WEBSOCKETS.dec(abs(delta))
    except Exception as exc:
        logger.debug(f"Metrics record_websocket_connection failed: {exc}")


def record_recalibration(trigger: str, outcome: str) -> None:
    """Record drift-triggered shadow recalibration result."""
    try:
        RECALIBRATIONS_TOTAL.labels(
            trigger=str(trigger),
            outcome=str(outcome),
        ).inc()
    except Exception as exc:
        logger.debug(f"Metrics record_recalibration failed: {exc}")


def record_rollback(reason: str) -> None:
    """Record adapter rollback event."""
    try:
        ROLLBACKS_TOTAL.labels(reason=str(reason)).inc()
    except Exception as exc:
        logger.debug(f"Metrics record_rollback failed: {exc}")


def get_metrics_exposition() -> Tuple[bytes, str]:
    """Generate Prometheus exposition text and appropriate HTTP Content-Type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
