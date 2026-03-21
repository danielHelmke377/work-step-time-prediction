"""
predictor.py — Central service wrapper for pipeline loading and inference.

This module owns the singleton ``predictor`` instance that is loaded once
at startup (via ``app/main.py`` lifespan) and reused for every request.

Design decisions
----------------
- Time unit: all time values are in **hours**, matching the pipeline's own
  key names (``predicted_hours``, ``total_input_time_hrs``).
- No refactoring of ``repair_order.pipeline``: the adapter converts between
  the API's snake_case field names and the pipeline's camelCase dict keys.
- Explanation logic mirrors ``scripts/predict.py::_explain_prediction()``
  but lives here so routers never import from ``scripts/``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from repair_order.config import KEYWORD_FLAGS, TARGET_KEYWORD_MAP, TARGET_LABELS
from repair_order.pipeline import load_pipeline

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PIPELINE_PATH = _REPO_ROOT / "models" / "two_stage_pipeline.pkl"
_MODEL_INFO_PATH = _REPO_ROOT / "models" / "model_info.json"

# Static metadata that describes the current pipeline architecture.
_FEATURE_GROUPS = [
    "tfidf_word_ngrams",
    "tfidf_char_ngrams",
    "numeric_aggregates",
    "keyword_flags",
    "make_encoding",
]


class ModelNotReadyError(Exception):
    """Raised when a prediction is requested before the pipeline is loaded."""


class PipelinePredictor:
    """
    Wraps the ``repair_order`` pipeline for use inside the FastAPI serving layer.

    Responsibilities
    ----------------
    - Load the pipeline pickle once at startup via ``load()``.
    - Expose readiness state for ``GET /health``.
    - Expose model metadata for ``GET /model-info``.
    - Adapt API request dicts → pipeline record format.
    - Call ``predict_order()`` and map the result to the API response schema.
    - Generate keyword-based explanations when requested.
    """

    def __init__(self) -> None:
        self._pipeline: dict[str, Any] | None = None
        self._pipeline_path: Path = _DEFAULT_PIPELINE_PATH
        self._model_info: dict[str, Any] = {}
        # Pre-compile keyword patterns once; reused for explanation.
        self._compiled_keywords: dict[str, re.Pattern] = {
            k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self, path: Path | None = None) -> None:
        """
        Load the pipeline pickle from disk and validate its structure.

        Called once from the FastAPI lifespan context in ``app/main.py``.

        Parameters
        ----------
        path:
            Optional path override. Defaults to ``models/two_stage_pipeline.pkl``
            relative to the repository root.

        Raises
        ------
        FileNotFoundError:
            If the pickle does not exist. The caller (lifespan) catches this
            and starts the service in degraded mode rather than crashing.
        ValueError:
            If the pickle is missing required pipeline keys.
        """
        target = path or _DEFAULT_PIPELINE_PATH
        self._pipeline_path = target
        self._pipeline = load_pipeline(target)
        self._model_info = self._load_model_info()
        logger.info("Pipeline loaded from %s. Targets: %d", target, self.n_targets)

    def _load_model_info(self) -> dict[str, Any]:
        """
        Load the ``models/model_info.json`` sidecar if present.

        Falls back to minimal metadata derived from the in-memory pipeline
        when the sidecar does not exist (e.g. pipeline trained before the
        sidecar feature was added to ``scripts/train.py``).
        """
        if _MODEL_INFO_PATH.exists():
            with open(_MODEL_INFO_PATH, encoding="utf-8") as f:
                return json.load(f)
        logger.warning(
            "models/model_info.json not found — returning minimal metadata. "
            "Re-train with the updated scripts/train.py to generate the sidecar."
        )
        return {"model_version": "unknown", "trained_on": "unknown"}

    # ── Readiness ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the pipeline artifact has been loaded successfully."""
        return self._pipeline is not None

    @property
    def n_targets(self) -> int:
        """Number of output targets in the loaded pipeline; 0 when not loaded."""
        return 0 if self._pipeline is None else len(self._pipeline["output_targets"])

    # ── Model metadata ────────────────────────────────────────────────────────

    def get_model_info(self) -> dict[str, Any]:
        """
        Return metadata for ``GET /model-info``.

        Combines the ``model_info.json`` sidecar (version, training date)
        with live pipeline state (target list) and static architecture constants.
        """
        targets = []
        if self._pipeline:
            targets = [
                {"id": t, "label": TARGET_LABELS.get(t, t)}
                for t in self._pipeline["output_targets"]
            ]

        # Express pipeline path relative to repo root for portability.
        try:
            rel_path = str(self._pipeline_path.relative_to(_REPO_ROOT))
        except ValueError:
            rel_path = str(self._pipeline_path)

        return {
            "model_version": self._model_info.get("model_version", "unknown"),
            "trained_on": self._model_info.get("trained_on", "unknown"),
            "pipeline_file": rel_path,
            "time_unit": "hours",
            "n_targets": self.n_targets,
            "targets": targets,
            "feature_groups": _FEATURE_GROUPS,
            "stage1_classifier": "LGBMClassifier",
            "stage2_regressors": ["LGBMRegressor", "Ridge", "mean_fallback"],
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Run the two-stage pipeline for a single repair order.

        Parameters
        ----------
        request_data:
            Validated dict produced by ``PredictRequest.model_dump()``.

        Returns
        -------
        dict conforming to ``PredictResponse``.

        Raises
        ------
        ModelNotReadyError:
            If the pipeline has not been loaded yet.
        """
        if not self.is_ready:
            raise ModelNotReadyError(
                "The prediction model is not loaded. Check GET /health for status."
            )

        from repair_order.pipeline import predict_order  # local import avoids startup cost

        record = self._adapt_request(request_data)
        result = predict_order(record, self._pipeline)
        result["_order"] = self._pipeline["output_targets"]

        explanation: dict | None = None
        if request_data.get("explain", False):
            explanation = self._build_explanation(result, record)

        return self._format_response(result, explanation)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _adapt_request(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an API request dict into the internal record format expected
        by ``repair_order.pipeline.predict_order()``.

        Field mapping (API → pipeline):
            make                → input.make
            calculated_positions → input.calculatedPositions
              .text             →   position.text
              .total_time       →   position.totalTime   (hours, passed through as-is)
              .total_price      →   position.totalPrice  (EUR, passed through as-is)
              .cost_center      →   position.genericCostCenter
        """
        positions = [
            {
                "text": pos.get("text", ""),
                "totalTime": float(pos.get("total_time", 0.0)),
                "totalPrice": float(pos.get("total_price", 0.0)),
                "genericCostCenter": pos.get("cost_center") or "unknown_cc",
            }
            for pos in req.get("calculated_positions", [])
        ]
        return {
            "input": {
                "make": req.get("make") or "unknown",
                "calculatedPositions": positions,
            }
        }

    def _build_explanation(
        self,
        result: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, dict]:
        """
        Build keyword-based explanation for each active target.

        Mirrors ``scripts/predict.py::RepairOrderPredictor._explain_prediction()``
        so the serving layer does not import from ``scripts/``.
        """
        positions = record["input"]["calculatedPositions"]
        explanations: dict[str, dict] = {}

        for t in result["active_targets"]:
            kw_names = TARGET_KEYWORD_MAP.get(t, [])
            triggered = [
                kw.replace("kw_", "")
                for kw in kw_names
                if result["_feats"].get(kw, 0)
            ]
            matching_texts: list[str] = []
            for pos in positions:
                txt = (pos.get("text") or "").strip()
                if not txt:
                    continue
                for kw in kw_names:
                    pat = self._compiled_keywords.get(kw)
                    if pat and pat.search(txt):
                        matching_texts.append(txt[:80])
                        break
            # Deduplicate while preserving order; keep at most 3 snippets.
            matching_texts = list(dict.fromkeys(matching_texts))[:3]

            explanations[t] = {
                "triggered_keywords": triggered,
                "matching_positions": matching_texts,
            }

        return explanations

    def _format_response(
        self,
        result: dict[str, Any],
        explanation: dict | None,
    ) -> dict[str, Any]:
        """
        Map ``predict_order()`` output to the ``PredictResponse`` schema dict.

        All time values use the pipeline's native hours representation —
        no unit conversion is performed.
        """
        predictions: dict[str, dict] = {}
        for t in result["_order"]:
            p = result["predictions"][t]
            predictions[t] = {
                "label": TARGET_LABELS.get(t, t),
                "active": p["active"],
                "probability": p["prob"],
                "threshold": p["threshold"],
                "predicted_hours": p["predicted_hours"],
            }

        # Wrap explanation entries to satisfy ExplanationEntry schema.
        formatted_explanation: dict[str, dict] | None = None
        if explanation is not None:
            formatted_explanation = {
                t: {
                    "triggered_keywords": exp["triggered_keywords"],
                    "matching_positions": exp["matching_positions"],
                }
                for t, exp in explanation.items()
            }

        return {
            "make": result["make"],
            "n_positions": result["n_positions"],
            "input_summary": {
                "total_time_hrs": result["total_input_time_hrs"],
                "total_price_eur": result["total_input_price_eur"],
            },
            "predictions": predictions,
            "active_steps": result["active_targets"],
            "total_predicted_hours": result["total_predicted_hours"],
            "elapsed_ms": result["elapsed_ms"],
            "explanation": formatted_explanation,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
#
# Imported by both app/main.py (lifespan load) and router modules (inference).
# This avoids the need to pass the predictor through app.state lookups on
# every request.

predictor = PipelinePredictor()
