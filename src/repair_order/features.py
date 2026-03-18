"""
features.py — Shared feature engineering for Work Step Time Prediction.
=======================================================================
Functions for preprocessing order positions, building the combined order
text string, and computing the numeric/keyword/make feature vector.

These functions are identical across train.py, predict.py, and the gbert_base equivalents — importing from here eliminates drift.
the gbert_base equivalents — importing from here eliminates drift.

Usage:
    from repair_order.features import (
        preprocess_positions,
        build_order_text,
        build_numeric_features,
    )
"""

import re
from typing import Any

import pandas as pd

from repair_order.config import COST_CENTERS, KEYWORD_FLAGS, TOP_MAKES

# Pre-compile all keyword patterns once at import time
_compiled_keywords: dict[str, re.Pattern] = {
    k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()
}


def preprocess_positions(positions: list[dict]) -> list[dict]:
    """
    Clean a list of raw order positions.

    Filters out completely empty rows (no text, zero price, zero time).
    Coerces price and time to float ≥ 0.

    Parameters
    ----------
    positions : list of raw position dicts from `calculatedPositions`

    Returns
    -------
    list of cleaned position dicts with keys:
        text, totalPrice, totalTime, genericCostCenter
    """
    cleaned = []
    for p in positions:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter") or "unknown_cc"
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({
            "text":              text,
            "totalPrice":        price,
            "totalTime":         time_,
            "genericCostCenter": cc,
        })
    return cleaned


def build_order_text(positions: list[dict]) -> str:
    """
    Concatenate position text for all positions with price > 0 OR time > 0.

    Returns a single lower-cased string suitable for TF-IDF or BERT encoding.

    Parameters
    ----------
    positions : cleaned positions (output of preprocess_positions)
    """
    parts = [
        p["text"].strip().lower()
        for p in positions
        if p["text"].strip() and (p["totalPrice"] > 0 or p["totalTime"] > 0)
    ]
    return " ".join(parts)


def build_numeric_features(
    positions: list[dict],
    make: str,
    make_freq_lookup: dict[str, int],
) -> dict[str, Any]:
    """
    Compute the numeric/keyword/make feature vector for one order.

    Features include:
    - Aggregate time and price statistics (total, mean, max, std, median)
    - Per-cost-center sums and counts
    - Time/price ratios per cost center
    - 17 binary keyword flags (German automotive domain)
    - Car make frequency + one-hot for top 8 makes

    Parameters
    ----------
    positions        : cleaned positions (output of preprocess_positions)
    make             : vehicle make string (e.g. "BMW")
    make_freq_lookup : dict mapping make → frequency rank (from training data)

    Returns
    -------
    dict of feature_name → float/int
    """
    feats: dict[str, Any] = {}
    df = pd.DataFrame(positions)
    df["totalPrice"] = df["totalPrice"].astype(float)
    df["totalTime"]  = df["totalTime"].astype(float)
    useful = df[(df["totalPrice"] > 0) | (df["totalTime"] > 0)]

    feats["n_positions"]  = len(df)
    feats["n_useful"]     = len(useful)
    feats["n_zero_time"]  = int((df["totalTime"] == 0).sum())
    feats["n_zero_price"] = int((df["totalPrice"] == 0).sum())
    feats["total_time"]   = df["totalTime"].sum()
    feats["total_price"]  = df["totalPrice"].sum()
    feats["max_time"]     = df["totalTime"].max()
    feats["mean_time"]    = df["totalTime"].mean()
    feats["median_time"]  = df["totalTime"].median()
    feats["max_price"]    = df["totalPrice"].max()
    feats["mean_price"]   = df["totalPrice"].mean()
    feats["std_time"]     = df["totalTime"].std(ddof=0)
    feats["std_price"]    = df["totalPrice"].std(ddof=0)

    # Per-cost-centre aggregates
    for cc in COST_CENTERS:
        sub = df[df["genericCostCenter"] == cc]
        feats[f"time_{cc}"]  = sub["totalTime"].sum()
        feats[f"price_{cc}"] = sub["totalPrice"].sum()
        feats[f"n_{cc}"]     = len(sub)

    # Ratios
    tt = feats["total_time"]
    tp = feats["total_price"]
    feats["ratio_painting_time"]  = feats["time_painting"]  / tt if tt > 0 else 0
    feats["ratio_bodywork_time"]  = feats["time_bodywork"]   / tt if tt > 0 else 0
    feats["ratio_hail_time"]      = feats["time_hail"]       / tt if tt > 0 else 0
    feats["ratio_material_price"] = feats["price_material"]  / tp if tp > 0 else 0

    # Boolean cost-centre indicators
    feats["has_hail_cc"]     = int(feats["n_hail"] > 0)
    feats["has_painting_cc"] = int(feats["n_painting"] > 0)
    feats["has_bodywork_cc"] = int(feats["n_bodywork"] > 0)

    # Keyword binary flags
    combined_text = " ".join(p["text"].lower() for p in positions)
    for kw, pattern in _compiled_keywords.items():
        feats[kw] = int(bool(pattern.search(combined_text)))

    # Make features
    make_clean = (make or "unknown").strip().upper()
    feats["make_freq"] = make_freq_lookup.get(make_clean, 1)
    for m in TOP_MAKES:
        feats[f"make_{m.replace('-', '_')}"] = int(make_clean == m)
    feats["make_other"] = int(make_clean not in TOP_MAKES)

    return feats
