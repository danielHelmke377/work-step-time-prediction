"""
pipeline.py — Pipeline loading and prediction utilities.
=========================================================
Shared logic for loading a saved two-stage pipeline pickle and
running a single-order prediction. Used by predict.py and
gbert_base/code/predict_gbert.py.

Usage:
    from repair_order.pipeline import load_pipeline, predict_order
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from repair_order.features import (
    build_numeric_features,
    build_order_text,
    preprocess_positions,
)

# Keys that every saved pipeline dict must contain
REQUIRED_PIPELINE_KEYS = [
    "clf_models",
    "reg_models",
    "thresholds",
    "best_clf_per_target",
    "output_targets",
    "numeric_features",
    "make_freq_lookup",
]


def load_pipeline(path: Path) -> dict[str, Any]:
    """
    Load a saved two-stage pipeline pickle and validate its structure.

    Parameters
    ----------
    path : Path to the .pkl file

    Returns
    -------
    pipeline dict

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if required keys are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {path}\n"
            "Run model_phase2.py (or model_gbert.py for the BERT variant) first."
        )
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    missing = [k for k in REQUIRED_PIPELINE_KEYS if k not in pipeline]
    if missing:
        raise ValueError(f"Pipeline missing required keys: {missing}")
    return pipeline


def featurize_order(
    record: dict,
    pipeline: dict,
    text_featurizer=None,
) -> tuple:
    """
    Build the feature matrix for a single order.

    Parameters
    ----------
    record          : raw order dict (must have `input.calculatedPositions`)
    pipeline        : loaded pipeline dict
    text_featurizer : callable(text) → sparse_matrix  (TF-IDF or BERT)
                      If None, the pipeline's own tfidf_word/tfidf_char are used.

    Returns
    -------
    X      : (1, n_features) sparse matrix
    feats  : numeric feature dict
    text   : combined order text string
    """
    positions = preprocess_positions(record["input"]["calculatedPositions"])
    make      = record["input"].get("make", "unknown")
    text      = build_order_text(positions)
    feats     = build_numeric_features(positions, make, pipeline["make_freq_lookup"])

    if text_featurizer is not None:
        X_text = text_featurizer(text)
    else:
        # Default: TF-IDF (word + char), stored in pipeline
        t = text if text.strip() else "[leer]"
        X_word = pipeline["tfidf_word"].transform([t])
        X_char = pipeline["tfidf_char"].transform([t])
        X_text = sparse.hstack([X_word, X_char], format="csr")

    feat_series = (
        pd.Series(feats)
        .reindex(pipeline["numeric_features"], fill_value=0)
    )
    X_num = sparse.csr_matrix(feat_series.values.reshape(1, -1))
    X     = sparse.hstack([X_text, X_num], format="csr")
    return X, feats, text


def predict_order(
    record: dict,
    pipeline: dict,
    text_featurizer=None,
) -> dict[str, Any]:
    """
    Run the two-stage prediction for a single order.

    Stage 1: binary classifier per target (occurrence)
    Stage 2: conditional regressor per target (duration in hours)

    Parameters
    ----------
    record          : raw order dict
    pipeline        : loaded pipeline dict
    text_featurizer : optional override for text feature extraction

    Returns
    -------
    dict with keys:
        make, n_positions, total_input_time_hrs, total_input_price_eur,
        predictions  (per-target: prob, active, predicted_hours, threshold),
        active_targets, total_predicted_hours, elapsed_ms,
        _feats, _text, _order
    """
    t0 = time.perf_counter()
    X, feats, text = featurize_order(record, pipeline, text_featurizer)

    targets      = pipeline["output_targets"]
    clf_models   = pipeline["clf_models"]
    reg_models   = pipeline["reg_models"]
    thresholds   = pipeline["thresholds"]
    clf_type_map = pipeline["best_clf_per_target"]

    predictions: dict[str, dict] = {}
    for t in targets:
        mtype  = clf_type_map[t]
        model  = clf_models[mtype][t]
        prob   = model.predict_proba(X)[0, 1]
        thr    = thresholds[mtype][t]
        active = bool(prob >= thr)

        if active:
            rm = reg_models["lgbm"].get(t) or reg_models["ridge"].get(t)
            if isinstance(rm, tuple) and rm[0] == "mean_fallback":
                duration = float(rm[1])
            else:
                duration = max(0.0, float(rm.predict(X)[0]))
        else:
            duration = 0.0

        predictions[t] = {
            "prob":            round(float(prob), 4),
            "active":          active,
            "predicted_hours": round(duration, 2),
            "threshold":       round(thr, 2),
        }

    active_targets        = [t for t in targets if predictions[t]["active"]]
    total_predicted_hours = sum(predictions[t]["predicted_hours"] for t in targets)
    elapsed_ms            = (time.perf_counter() - t0) * 1000

    return {
        "make":                  record["input"].get("make", "unknown"),
        "n_positions":           len(record["input"]["calculatedPositions"]),
        "total_input_time_hrs":  round(feats["total_time"], 2),
        "total_input_price_eur": round(feats["total_price"], 2),
        "predictions":           predictions,
        "active_targets":        active_targets,
        "total_predicted_hours": round(total_predicted_hours, 2),
        "elapsed_ms":            round(elapsed_ms, 1),
        "_feats":                feats,
        "_text":                 text,
        "_order":                targets,
    }
