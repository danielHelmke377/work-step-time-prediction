"""
test_api.py — Integration tests for the Work Step Time Prediction API service.

Test structure
--------------
- TestDegradedMode:     Endpoints behave correctly when no model is loaded.
- TestValidationEdgeCases: Pydantic coercion and extra-field rules.
- TestModelLoaded:      Full prediction contract when artifact is available
                        (skipped automatically if pipeline pickle is absent).

The module-level TestClient does NOT start the lifespan context, so the
``predictor`` singleton starts unloaded. TestModelLoaded.load_pipeline()
calls predictor.load() explicitly, after which the remaining tests in that
class run against the live artifact.

Run with:
    pytest tests/test_api.py -vv
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import predictor

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = REPO_ROOT / "models" / "two_stage_pipeline.pkl"

# Instantiated without a context manager so the FastAPI lifespan does NOT run.
# Model loading is controlled entirely by the TestModelLoaded fixture.
client = TestClient(app, raise_server_exceptions=False)


# ── Degraded-mode tests (no model loaded) ─────────────────────────────────────


class TestDegradedMode:
    """Verify correct behaviour when the pipeline artifact is not loaded."""

    def test_health_returns_503(self):
        r = client.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["n_targets"] == 0

    def test_model_info_returns_200_without_sidecar(self):
        """GET /model-info must work even before the model loads."""
        r = client.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        assert "model_version" in body
        assert "n_targets" in body
        assert "targets" in body
        assert body["time_unit"] == "hours"

    def test_predict_returns_503_with_error_envelope(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "TEST", "total_time": 1.0}]},
        )
        assert r.status_code == 503
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "MODEL_NOT_READY"
        assert body["error"]["message"] != ""
        assert body["error"]["details"] is None

    def test_swagger_ui_available(self):
        assert client.get("/docs").status_code == 200

    def test_openapi_schema_has_all_paths(self):
        schema = client.get("/openapi.json").json()
        for path in ("/health", "/predict", "/model-info"):
            assert path in schema["paths"], f"Missing path in OpenAPI schema: {path}"


# ── Validation and coercion edge cases (no model needed for result code) ──────


class TestValidationEdgeCases:
    """Verify request validation rules defined in the API contract."""

    def test_empty_positions_returns_422_envelope(self):
        r = client.post("/predict", json={"calculated_positions": []})
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert isinstance(body["error"]["details"], list)
        assert len(body["error"]["details"]) > 0

    def test_missing_positions_field_returns_422(self):
        r = client.post("/predict", json={"make": "VW"})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_invalid_numeric_type_returns_422(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "X", "total_price": "not-a-number"}]},
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_extra_fields_are_silently_ignored(self):
        """Unknown top-level keys must not cause a 422 — they are stripped by Pydantic."""
        r = client.post(
            "/predict",
            json={
                "calculated_positions": [{"text": "TEST", "total_time": 1.0}],
                "order_id": "IGNORED_EXTRA",
                "internal_ref": 99,
            },
        )
        # Validation passes; 503 means the payload reached the model-not-ready check.
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MODEL_NOT_READY"

    def test_extra_position_fields_are_silently_ignored(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "X", "total_time": 1.0, "unknown_key": "ignored"}]},
        )
        assert r.status_code == 503  # validation passed

    def test_null_make_is_coerced_to_unknown(self):
        r = client.post(
            "/predict",
            json={"make": None, "calculated_positions": [{"text": "TEST", "total_time": 1.0}]},
        )
        assert r.status_code == 503  # coercion worked; reached model check

    def test_negative_price_is_clamped_not_rejected(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "X", "total_price": -50.0, "total_time": 1.0}]},
        )
        assert r.status_code == 503  # passes validation

    def test_null_text_and_numerics_are_coerced(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": None, "total_time": None, "total_price": None}]},
        )
        assert r.status_code == 503  # passes validation


# ── Model-loaded tests (require trained pipeline artifact) ────────────────────


class TestModelLoaded:
    """
    Integration tests against the live pipeline artifact.

    Skipped automatically if ``models/two_stage_pipeline.pkl`` is absent.
    Run 'make train' or 'python scripts/train.py' to create the artifact.
    """

    @pytest.fixture(scope="class", autouse=True)
    def load_pipeline(self):
        if not PIPELINE_PATH.exists():
            pytest.skip(
                f"Pipeline not found at {PIPELINE_PATH}. "
                "Run 'make train' or 'python scripts/train.py' first."
            )
        predictor.load()

    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["n_targets"] > 0

    def test_model_info_returns_all_required_fields(self):
        r = client.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        assert body["n_targets"] > 0
        assert body["time_unit"] == "hours"
        assert isinstance(body["targets"], list)
        assert len(body["targets"]) == body["n_targets"]
        for t in body["targets"]:
            assert "id" in t and "label" in t
        assert "feature_groups" in body
        assert "stage1_classifier" in body
        assert "stage2_regressors" in body

    def test_predict_success_response_shape(self):
        r = client.post(
            "/predict",
            json={
                "make": "VOLKSWAGEN",
                "calculated_positions": [
                    {
                        "text": "HAGELSCHADENREPARATUR METHODE",
                        "total_time": 10.0,
                        "total_price": 2500.0,
                        "cost_center": "hail",
                    }
                ],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "make" in body
        assert "n_positions" in body
        assert "predictions" in body
        assert "active_steps" in body
        assert "total_predicted_hours" in body
        assert "elapsed_ms" in body
        assert "input_summary" in body
        assert "total_time_hrs" in body["input_summary"]
        assert "total_price_eur" in body["input_summary"]
        assert body["explanation"] is None  # explain not requested

    def test_predict_all_expected_targets_present(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "KAROSSERIE", "total_time": 5.0}]},
        )
        assert r.status_code == 200
        predictions = r.json()["predictions"]
        info = client.get("/model-info").json()
        expected_ids = {t["id"] for t in info["targets"]}
        assert set(predictions.keys()) == expected_ids

    def test_predict_each_target_has_required_fields(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "TEST", "total_time": 1.0}]},
        )
        assert r.status_code == 200
        for tid, pred in r.json()["predictions"].items():
            assert "label" in pred,          f"missing label for {tid}"
            assert "active" in pred,         f"missing active for {tid}"
            assert "probability" in pred,    f"missing probability for {tid}"
            assert "threshold" in pred,      f"missing threshold for {tid}"
            assert "predicted_hours" in pred, f"missing predicted_hours for {tid}"

    def test_predict_inactive_targets_have_zero_hours(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "TEST", "total_time": 1.0}]},
        )
        assert r.status_code == 200
        for tid, pred in r.json()["predictions"].items():
            if not pred["active"]:
                assert pred["predicted_hours"] == 0.0, (
                    f"Inactive target {tid} should have predicted_hours=0.0"
                )

    def test_predict_total_hours_matches_per_target_sum(self):
        r = client.post(
            "/predict",
            json={
                "calculated_positions": [
                    {"text": "LACKIEREN KAROSSERIE REINIGUNG", "total_time": 8.0, "total_price": 1200.0}
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        computed = sum(p["predicted_hours"] for p in body["predictions"].values())
        assert abs(computed - body["total_predicted_hours"]) < 0.01, (
            f"total_predicted_hours {body['total_predicted_hours']} != sum {computed:.4f}"
        )

    def test_predict_with_explain_returns_explanation_for_active_steps(self):
        r = client.post(
            "/predict",
            json={
                "make": "VW",
                "calculated_positions": [
                    {
                        "text": "HAGELSCHADENREPARATUR DELLEN PDR",
                        "total_time": 10.0,
                        "total_price": 2000.0,
                        "cost_center": "hail",
                    }
                ],
                "explain": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "explanation" in body
        if body["active_steps"]:
            assert body["explanation"] is not None
            for step in body["active_steps"]:
                assert step in body["explanation"], f"Active step {step} missing from explanation"
                exp = body["explanation"][step]
                assert "triggered_keywords" in exp
                assert "matching_positions" in exp
                assert isinstance(exp["triggered_keywords"], list)
                assert isinstance(exp["matching_positions"], list)

    def test_predict_elapsed_ms_is_positive(self):
        r = client.post(
            "/predict",
            json={"calculated_positions": [{"text": "LACKIEREN", "total_time": 2.0}]},
        )
        assert r.status_code == 200
        assert r.json()["elapsed_ms"] > 0

    def test_model_info_version_present_after_training(self):
        """model_info.json sidecar should exist after training and expose a version."""
        model_info_path = REPO_ROOT / "models" / "model_info.json"
        if not model_info_path.exists():
            pytest.skip("model_info.json not present — re-train with updated scripts/train.py")
        r = client.get("/model-info")
        body = r.json()
        assert body["model_version"] != "unknown", (
            "model_version should be set after training; check models/model_info.json"
        )
        assert body["trained_on"] != "unknown"
