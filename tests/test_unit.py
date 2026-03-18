"""
test_unit.py — Unit tests for the repair_order feature engineering package.
===========================================================================
These tests are fast, deterministic, and require no trained model or
saved artefacts. They verify the feature-building contract that every
model decision depends on.

Run with:
    pytest -vv
"""

from repair_order.features import build_numeric_features, build_order_text, preprocess_positions

# ---------------------------------------------------------------------------
# Minimal synthetic order fixture used across tests
# ---------------------------------------------------------------------------

_POSITIONS = [
    {
        "text": "HAGEL DELLEN PDR",
        "totalTime": 120,
        "totalPrice": 250.0,
        "genericCostCenter": "bodywork",
    },
    {
        "text": "LACKIEREN KOTFLÜGEL",
        "totalTime": 60,
        "totalPrice": 180.0,
        "genericCostCenter": "painting",
    },
]

_ORDER = {
    "input": {
        "calculatedPositions": _POSITIONS,
        "make": "VOLKSWAGEN",
    },
    "output": {},
}


# ---------------------------------------------------------------------------
# preprocess_positions
# ---------------------------------------------------------------------------

class TestPreprocessPositions:
    def test_returns_list(self):
        result = preprocess_positions(_POSITIONS)
        assert isinstance(result, list)

    def test_filters_empty_rows(self):
        positions = _POSITIONS + [{"text": "", "totalTime": 0, "totalPrice": 0, "genericCostCenter": ""}]
        result = preprocess_positions(positions)
        # The empty row should be filtered out
        assert len(result) == len(_POSITIONS)

    def test_coerces_price_and_time_to_float(self):
        positions = [{"text": "TEST", "totalTime": "90", "totalPrice": "150", "genericCostCenter": "painting"}]
        result = preprocess_positions(positions)
        assert result[0]["totalTime"] == 90.0
        assert result[0]["totalPrice"] == 150.0


# ---------------------------------------------------------------------------
# build_order_text
# ---------------------------------------------------------------------------

class TestBuildOrderText:
    def test_returns_string(self):
        cleaned = preprocess_positions(_POSITIONS)
        text = build_order_text(cleaned)
        assert isinstance(text, str)

    def test_contains_position_tokens(self):
        cleaned = preprocess_positions(_POSITIONS)
        text = build_order_text(cleaned).lower()
        assert "hagel" in text or "dellen" in text, "Expected domain tokens from first position"
        assert "lackier" in text or "kotfl" in text, "Expected domain tokens from second position"

    def test_nonempty_for_valid_order(self):
        cleaned = preprocess_positions(_POSITIONS)
        text = build_order_text(cleaned)
        assert len(text.strip()) > 0


# ---------------------------------------------------------------------------
# build_numeric_features
# ---------------------------------------------------------------------------

class TestBuildNumericFeatures:
    def test_returns_dict(self):
        cleaned = preprocess_positions(_POSITIONS)
        result = build_numeric_features(cleaned, "VOLKSWAGEN", {})
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_feature_dict_has_cost_centre_keys(self):
        """Cost-centre time totals must be present for each configured cost centre."""
        from repair_order.config import COST_CENTERS

        cleaned = preprocess_positions(_POSITIONS)
        result = build_numeric_features(cleaned, "VOLKSWAGEN", {})
        for cc in COST_CENTERS:
            assert f"time_{cc}" in result, f"Missing feature time_{cc}"

    def test_feature_dict_has_keyword_flags(self):
        """Binary keyword flags must be present for all configured patterns."""
        from repair_order.config import KEYWORD_FLAGS

        cleaned = preprocess_positions(_POSITIONS)
        result = build_numeric_features(cleaned, "VOLKSWAGEN", {})
        for kw in KEYWORD_FLAGS:
            assert kw in result, f"Missing keyword feature {kw}"

    def test_make_features_present(self):
        cleaned = preprocess_positions(_POSITIONS)
        result = build_numeric_features(cleaned, "VOLKSWAGEN", {})
        assert "make_freq" in result
        assert "make_other" in result
