"""
predict.py — POST /predict

Runs the two-stage repair-order pipeline for a single input order and
returns structured predictions for all 14 work steps with optional
keyword-based explanations.

Error handling
--------------
- ``ModelNotReadyError`` → HTTP 503 with the standard error envelope.
- Unexpected exceptions are re-raised and caught by the global handler in
  ``app/main.py``, which returns HTTP 500 without leaking stack traces.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.predictor import ModelNotReadyError, predictor
from app.schemas import ErrorResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Request body failed validation."},
        503: {"model": ErrorResponse, "description": "Model not loaded — check GET /health."},
        500: {"model": ErrorResponse, "description": "Unexpected internal error."},
    },
    summary="Predict work steps and durations for a single repair order",
    description=(
        "Accepts a single repair order JSON and runs the two-stage ML pipeline:\n\n"
        "1. **Stage 1 (LightGBM):** classifies which of the 14 work steps are predicted to occur.\n"
        "2. **Stage 2 (per-target regressors):** predicts duration in **hours** for each active step.\n\n"
        "Set ``explain: true`` in the request to receive keyword-based explanations "
        "for each predicted-active work step."
    ),
)
async def predict(body: PredictRequest) -> JSONResponse:
    """Run the two-stage pipeline and return structured work-step predictions."""
    try:
        result = predictor.predict(body.model_dump())
    except ModelNotReadyError as exc:
        logger.warning("Prediction attempted before model was loaded: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "MODEL_NOT_READY",
                    "message": str(exc),
                    "details": None,
                }
            },
        )

    response = PredictResponse(**result)
    return JSONResponse(content=response.model_dump())
