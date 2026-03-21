"""
main.py — FastAPI application entry point for the Work Step Time Prediction API.

Startup sequence
----------------
1. FastAPI lifespan context calls ``predictor.load()`` once.
2. If the model artifact is missing, the service starts in *degraded* mode:
   - ``GET /health`` returns HTTP 503.
   - ``POST /predict`` returns HTTP 503 with the standard error envelope.
   - ``GET /model-info`` returns whatever metadata is available.
3. Custom exception handlers ensure no stack traces leak to API clients.

Running locally
---------------
    pip install -e ".[serve]"
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.predictor import predictor
from app.routers import health, model_info, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load pipeline once at startup ───────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained pipeline on startup; log gracefully if unavailable.

    The service starts in *degraded* mode when the artifact is missing rather
    than refusing to start — this lets ``GET /health`` signal the problem
    through the standard monitoring channel instead of a crash.
    """
    logger.info("Starting Work Step Time Prediction API…")
    try:
        predictor.load()
        logger.info("Pipeline ready. Targets: %d", predictor.n_targets)
    except FileNotFoundError as exc:
        logger.warning(
            "Pipeline artifact not found (%s). "
            "Service starting in degraded mode — run 'make train' to create the artifact.",
            exc,
        )
    except ValueError as exc:
        logger.error(
            "Pipeline artifact is malformed (%s). "
            "Service starting in degraded mode.",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error loading pipeline (%s). "
            "Service starting in degraded mode.",
            exc,
        )
    yield
    logger.info("Shutting down Work Step Time Prediction API.")


# ── Application ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="Work Step Time Prediction API",
    description=(
        "Production-style inference service for predicting automotive body-shop "
        "repair work steps and their durations from raw JSON repair orders.\n\n"
        "**Two-stage ML pipeline**\n\n"
        "- **Stage 1 (LightGBM classifiers):** predict which of 14 work steps occur.\n"
        "- **Stage 2 (per-target regressors):** predict duration in hours for active steps.\n\n"
        "All time values in this API are in **hours**."
    ),
    version="1.0.0",
    contact={"name": "Work Step Time Prediction", "url": "https://github.com/danielHelmke377/work-step-time-prediction"},
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(model_info.router)


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map Pydantic validation errors to the standard error envelope (HTTP 422)."""
    details = [
        {
            "field": ".".join(str(loc) for loc in err["loc"]),
            "issue": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request body failed validation.",
                "details": details,
            }
        },
    )


@app.exception_handler(Exception)
async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled exceptions.

    Logs the full traceback server-side but returns only a safe generic
    message to the client — no stack traces, no internal paths.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please check the server logs.",
                "details": None,
            }
        },
    )
