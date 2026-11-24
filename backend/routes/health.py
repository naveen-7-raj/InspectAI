"""Health check and model readiness endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from backend.dependencies import AppState, get_app_state
from backend.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Model Readiness",
    description="Reports the operational status of the API, model weights availability, and device configuration.",
    responses={
        200: {"description": "API and model are fully loaded and operational."},
        503: {"description": "API is online but model loading failed or is incomplete."},
    },
)
def get_health(state: AppState = Depends(get_app_state)) -> JSONResponse:
    """Check API operational health and model readiness status."""
    is_ready = bool(state.is_loaded and state.model is not None and state.normalizer is not None)

    response_data = HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name="VibrationCNN1D" if is_ready else "None",
        supported_classes=state.class_names,
        device=state.device,
        gradcam_available=bool(is_ready and state.explainer is not None),
        normalizer_fitted=bool(is_ready and state.normalizer is not None and state.normalizer.is_fitted),
        api_version="0.1.0",
    )

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=response_data.model_dump())

