"""
model_info.py — GET /model-info

Returns stable metadata about the loaded pipeline artifact, including
version, training date, target list, and feature engineering summary.

Resilience: if ``models/model_info.json`` does not exist (e.g. the pipeline
was trained before the sidecar feature was added), the endpoint returns
whatever metadata it can derive from the in-memory pipeline and logs a warning.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.predictor import predictor
from app.schemas import ModelInfoResponse

router = APIRouter(tags=["Model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and configuration",
    description=(
        "Returns metadata about the currently loaded pipeline artifact: "
        "version, training date, target list, feature groups, and classifier types.\n\n"
        "The ``model_version`` and ``trained_on`` fields are sourced from "
        "``models/model_info.json`` (written by ``scripts/train.py``). "
        "If the sidecar file is absent they are reported as ``'unknown'``."
    ),
)
async def model_info() -> JSONResponse:
    """Return metadata for the loaded pipeline artifact."""
    info = predictor.get_model_info()
    response = ModelInfoResponse(**info)
    return JSONResponse(content=response.model_dump())
