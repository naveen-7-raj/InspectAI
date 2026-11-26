"""InspectAI FastAPI Application: Explainable Industrial Machine Health Monitoring Backend."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.dependencies import AppState, load_model_artifacts
from backend.routes.health import router as health_router
from backend.routes.prediction import router as prediction_router
from backend.routes.streaming import router as streaming_router
from backend.routes.metrics import router as metrics_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.FastAPI")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: initialize and cache frozen CNN and normalizer on startup."""
    state = AppState()
    try:
        load_model_artifacts(state)
        logger.info("InspectAI model artifacts successfully initialized in lifespan.")
    except Exception as exc:
        logger.error(f"Startup warning: model loading failed: {exc}")
        state.is_loaded = False
        state.load_error = str(exc)

    app.state.app_state = state
    yield
    logger.info("InspectAI FastAPI application shutting down.")


app = FastAPI(
    title="InspectAI - Explainable Industrial Health Monitoring API",
    description=(
        "REST API for real-time vibration signal classification and Explainable AI (1D Grad-CAM). "
        "Processes continuous accelerometer time-series (12 kHz & 48 kHz), runs inference using "
        "the frozen 1D CNN, and generates temporal attribution heatmaps."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Standardized error handlers avoiding internal stack trace leakage
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic schema validation failures into clean JSON errors."""
    error_details = []
    for err in exc.errors():
        field_loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        error_details.append(f"{field_loc}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "; ".join(error_details),
            "error_type": "ValidationError",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Pass through intended HTTP exceptions cleanly."""
    error_type = "InternalServerError" if exc.status_code >= 500 else "HTTPException"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_type": error_type,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled server errors and sanitize error messages."""
    logger.exception(f"Unhandled exception during request processing: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred during request processing.",
            "error_type": "InternalServerError",
        },
    )


# Register API v1 routes
app.include_router(health_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(streaming_router, prefix="/api/v1")
app.include_router(metrics_router)


# Root-level health endpoint for backwards compatibility and load balancers
@app.get("/health", include_in_schema=False)
def legacy_health(request: Request) -> JSONResponse:
    """Redirect to /api/v1/health."""
    state = getattr(request.app.state, "app_state", None)
    if state is None or not state.is_loaded:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "degraded"})
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "healthy", "system": "InspectAI"})
