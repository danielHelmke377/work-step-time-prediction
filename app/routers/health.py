"""
health.py — GET /health

Liveness + readiness endpoint.

Returns HTTP 200 when the model is loaded and ready to serve predictions.
Returns HTTP 503 when the service is alive but the model failed to load,
so that load balancers and orchestrators route traffic away from the instance.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.predictor import predictor
from app.schemas import HealthResponse

router = APIRouter(tags=["Observability"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"description": "Service is alive but model is not loaded."}},
    summary="Liveness and readiness check",
    description=(
        "Returns **200 OK** when the prediction model is loaded and ready.\n\n"
        "Returns **503 Service Unavailable** when the process is up but the "
        "model artifact has not been loaded successfully. Use this endpoint "
        "for load balancer health checks and orchestrator readiness probes."
    ),
)
async def health_check() -> JSONResponse:
    """Report service liveness and model readiness."""
    ready = predictor.is_ready
    body = HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        n_targets=predictor.n_targets,
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content=body.model_dump(),
    )
