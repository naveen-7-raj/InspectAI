"""Prometheus Metrics Route for InspectAI (Phase 19)."""

from fastapi import APIRouter, Response
from src.monitoring.metrics import get_metrics_exposition

router = APIRouter(tags=["Observability"])


@router.get("/metrics", summary="Prometheus Metrics Exposition")
@router.get("/api/v1/metrics", summary="Prometheus Metrics Exposition (API v1)")
def prometheus_metrics() -> Response:
    """Expose application metrics in standard Prometheus text exposition format."""
    payload, content_type = get_metrics_exposition()
    return Response(content=payload, media_type=content_type)
