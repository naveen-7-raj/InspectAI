"""API route handlers for InspectAI backend."""

from backend.routes.health import router as health_router
from backend.routes.prediction import router as prediction_router
from backend.routes.streaming import router as streaming_router
from backend.routes.metrics import router as metrics_router

__all__ = ["health_router", "prediction_router", "streaming_router", "metrics_router"]
