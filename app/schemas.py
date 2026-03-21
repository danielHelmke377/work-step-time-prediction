"""
schemas.py — Pydantic v2 request and response models for the prediction API.

All time values are in **hours**, matching the pipeline's own key names
(``predicted_hours``, ``total_input_time_hrs``).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# ── Request models ────────────────────────────────────────────────────────────


class CalculatedPosition(BaseModel):
    """A single line item (position) from a repair order."""

    model_config = {"extra": "ignore"}

    text: str = Field(default="", description="Position description text (German automotive domain).")
    total_time: float = Field(default=0.0, description="Duration for this position, in hours.")
    total_price: float = Field(default=0.0, description="Price for this position, in EUR.")
    cost_center: str = Field(
        default="unknown_cc",
        description="Generic cost-center label (e.g. 'bodywork', 'painting', 'hail').",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls_and_negatives(cls, values: Any) -> Any:
        """
        Normalise missing or null fields before type-validation runs:
        - null / missing text → ""
        - null / missing cost_center → "unknown_cc"
        - null / missing numeric fields → 0.0
        - negative numeric fields → clamped to 0.0 (per contract; not a 422)
        """
        if not isinstance(values, dict):
            return values

        if values.get("text") is None:
            values["text"] = ""

        cc = values.get("cost_center")
        if cc is None or (isinstance(cc, str) and not cc.strip()):
            values["cost_center"] = "unknown_cc"

        for field in ("total_time", "total_price"):
            val = values.get(field)
            if val is None:
                values[field] = 0.0
            else:
                try:
                    fval = float(val)
                    if fval < 0:
                        values[field] = 0.0
                except (ValueError, TypeError):
                    pass  # Pydantic will raise a typed validation error

        return values


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    model_config = {"extra": "ignore"}

    make: str = Field(
        default="unknown",
        description="Vehicle manufacturer name (e.g. 'VOLKSWAGEN'). Case-insensitive.",
    )
    calculated_positions: list[CalculatedPosition] = Field(
        min_length=1,
        description="Line items from the repair order. At least one item is required.",
    )
    explain: bool = Field(
        default=False,
        description="When true, the response includes a keyword-based explanation for each active step.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_make(cls, values: Any) -> Any:
        """Treat null or blank make as 'unknown'."""
        if isinstance(values, dict):
            make = values.get("make")
            if not make or (isinstance(make, str) and not make.strip()):
                values["make"] = "unknown"
        return values


# ── Prediction response models ────────────────────────────────────────────────


class WorkStepPrediction(BaseModel):
    """Prediction result for a single work-step target."""

    label: str = Field(description="Human-readable work-step name.")
    active: bool = Field(description="True if the work step is predicted to occur.")
    probability: float = Field(description="Raw Stage-1 classifier probability [0.0–1.0].")
    threshold: float = Field(description="Per-target F1-optimised decision threshold.")
    predicted_hours: float = Field(description="Predicted duration in hours. 0.0 when inactive.")


class InputSummary(BaseModel):
    """Aggregated statistics derived from the incoming repair order positions."""

    total_time_hrs: float = Field(description="Sum of total_time across all positions, in hours.")
    total_price_eur: float = Field(description="Sum of total_price across all positions, in EUR.")


class ExplanationEntry(BaseModel):
    """Keyword-based explanation for a single predicted-active work step."""

    triggered_keywords: list[str] = Field(
        description="Domain keyword names that matched position text (prefix 'kw_' stripped)."
    )
    matching_positions: list[str] = Field(
        description="Up to 3 position text snippets that triggered the keyword match."
    )


class PredictResponse(BaseModel):
    """Successful prediction response for a single repair order."""

    make: str = Field(description="Normalised vehicle make used during prediction.")
    n_positions: int = Field(description="Number of positions after empty-row filtering.")
    input_summary: InputSummary
    predictions: dict[str, WorkStepPrediction] = Field(
        description=(
            "Per-target predictions keyed by internal target ID (e.g. 'hailrepair'). "
            "All 14 targets are always present."
        )
    )
    active_steps: list[str] = Field(
        description="Ordered list of target IDs where active=true."
    )
    total_predicted_hours: float = Field(
        description="Sum of predicted_hours across all targets, in hours."
    )
    elapsed_ms: float = Field(description="Wall-clock inference time in milliseconds.")
    explanation: dict[str, ExplanationEntry] | None = Field(
        default=None,
        description="Populated only when explain=true in the request; null otherwise.",
    )


# ── Error envelope ────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """A single field-level validation error."""

    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all non-2xx responses."""

    error: ErrorBody


# ── Health response ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = Field(description="'ok' when the model is ready, 'degraded' otherwise.")
    model_loaded: bool = Field(description="True if the pipeline artifact was loaded successfully.")
    n_targets: int = Field(description="Number of output targets in the loaded pipeline; 0 if not loaded.")


# ── Model-info response ───────────────────────────────────────────────────────


class TargetInfo(BaseModel):
    """Identifier and human-readable label for a single work-step target."""

    id: str
    label: str


class ModelInfoResponse(BaseModel):
    """Response for ``GET /model-info``."""

    model_version: str
    trained_on: str
    pipeline_file: str
    time_unit: str = Field(default="hours", description="Unit for all time fields in this API.")
    n_targets: int
    targets: list[TargetInfo]
    feature_groups: list[str]
    stage1_classifier: str
    stage2_regressors: list[str]
