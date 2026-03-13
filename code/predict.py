"""
predict.py — Inference Script
================================
Loads the saved two-stage pipeline and predicts repair work steps
from a raw JSON order (or an example pulled from the dataset).

Usage:
    # Predict from a JSON file containing a single order:
    python predict.py --input my_order.json

    # Predict from a specific record in the training data (by index):
    python predict.py --demo 42

    # Run a batch of N random demo orders and print a summary table:
    python predict.py --batch 10

    # Evaluate against true labels for a demo record:
    python predict.py --demo 42 --evaluate
"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path(r"C:\Users\Administrator\baum\orders_simplified_sample.json")
PIPELINE_PATH = Path(r"C:\Users\Administrator\baum\models\two_stage_pipeline.pkl")

# ── Domain keyword patterns (mirrored from model_phase2.py) ──────────────────
KEYWORD_FLAGS = {
    "kw_vermessung":    r"vermess|kinematik|spur|sturz",
    "kw_kalibrier":     r"kalibrier|adas|fas|kamera.*kalib|radar.*ausrich",
    "kw_glas":          r"glas|scheibe|frontscheibe|windschutz|heckscheibe|windlauf",
    "kw_hagel":         r"hagel|dellen|pdr|smart.*repar",
    "kw_reifen":        r"reifen|felge|rad(?:wechsel|montage|service)",
    "kw_reinigung":     r"reinigung|waesch|polier",
    "kw_lack":          r"lackier|lack(?!material)|oberflaech.*lack|neu.*lack",
    "kw_vorbereitung":  r"vorbereitung|grundier|fuellerauftrag|vorbereit",
    "kw_klebetechnik":  r"klebe|klebetechnik",
    "kw_montage":       r"a\+e|montage|einbau|ausbau|demontage|ersatz",
    "kw_hybrid":        r"hybrid|elektro|hochspannung|hv.system",
    "kw_plastik":       r"plastik|kunststoff|stossfaenger.*repar",
    "kw_karosserie":    r"karosserie|blech|beul|richt",
    "kw_scheibe_ers":   r"scheibe.*ers|frontscheibe|windschutz.*ers",
    "kw_dellen":        r"dellen|dent|beule",
    "kw_sensor":        r"sensor|pdc|adas|ultraschall",
    "kw_material":      r"kleinmaterial|ersatzteil|lackmaterial",
}

COST_CENTERS = ["bodywork", "painting", "paintmaterial", "material", "others", "hail"]
TOP_MAKES    = ["VOLKSWAGEN", "MERCEDES-BENZ", "BMW", "FORD", "SKODA",
                "AUDI", "OPEL", "TESLA"]

# Human-readable target labels for the report
TARGET_LABELS = {
    "calibration":        "Calibration (ADAS/cameras)",
    "wheelmeasurement":   "Wheel alignment measurement",
    "bodymeasurement":    "Body/chassis measurement",
    "dismounting":        "Dis-/mounting",
    "bodyrepair":         "Body repair",
    "assembly":           "Assembly",
    "plasticrepair":      "Plastic repair",
    "allTiresService":    "Tyre service",
    "cleaning":           "Cleaning",
    "paintingPreparation":"Painting — preparation",
    "paintingSpraying":   "Painting — spraying",
    "paintingFinish":     "Painting — finish",
    "hailrepair":         "Hail repair",
    "glas":               "Glass replacement",
}

# Which keyword flags are diagnostic for which targets (for explanation)
TARGET_KEYWORD_MAP = {
    "calibration":        ["kw_kalibrier", "kw_sensor"],
    "wheelmeasurement":   ["kw_vermessung"],
    "bodymeasurement":    ["kw_vermessung", "kw_karosserie"],
    "dismounting":        ["kw_montage"],
    "bodyrepair":         ["kw_karosserie"],
    "assembly":           ["kw_montage"],
    "plasticrepair":      ["kw_plastik"],
    "allTiresService":    ["kw_reifen"],
    "cleaning":           ["kw_reinigung"],
    "paintingPreparation":["kw_vorbereitung", "kw_lack"],
    "paintingSpraying":   ["kw_lack"],
    "paintingFinish":     ["kw_lack", "kw_klebetechnik"],
    "hailrepair":         ["kw_hagel", "kw_dellen"],
    "glas":               ["kw_glas", "kw_scheibe_ers"],
}

compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}


# ============================================================================
# FEATURE ENGINEERING (must match model_phase2.py exactly)
# ============================================================================

def preprocess_positions(positions):
    cleaned = []
    for p in positions:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter", None) or "unknown_cc"
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({"text": text, "totalPrice": price,
                         "totalTime": time_, "genericCostCenter": cc})
    return cleaned


def build_order_text(positions):
    parts = []
    for p in positions:
        t = p["text"].strip()
        if t and (p["totalPrice"] > 0 or p["totalTime"] > 0):
            parts.append(t.lower())
    return " ".join(parts)


def build_numeric_features(positions, make, make_freq_lookup):
    feats = {}
    df = pd.DataFrame(positions)
    df["totalPrice"] = df["totalPrice"].astype(float)
    df["totalTime"]  = df["totalTime"].astype(float)
    useful = df[(df["totalPrice"] > 0) | (df["totalTime"] > 0)]

    feats["n_positions"]     = len(df)
    feats["n_useful"]        = len(useful)
    feats["n_zero_time"]     = (df["totalTime"] == 0).sum()
    feats["n_zero_price"]    = (df["totalPrice"] == 0).sum()
    feats["total_time"]      = df["totalTime"].sum()
    feats["total_price"]     = df["totalPrice"].sum()
    feats["max_time"]        = df["totalTime"].max()
    feats["mean_time"]       = df["totalTime"].mean()
    feats["median_time"]     = df["totalTime"].median()
    feats["max_price"]       = df["totalPrice"].max()
    feats["mean_price"]      = df["totalPrice"].mean()
    feats["std_time"]        = df["totalTime"].std(ddof=0)
    feats["std_price"]       = df["totalPrice"].std(ddof=0)

    for cc in COST_CENTERS:
        sub = df[df["genericCostCenter"] == cc]
        feats[f"time_{cc}"]  = sub["totalTime"].sum()
        feats[f"price_{cc}"] = sub["totalPrice"].sum()
        feats[f"n_{cc}"]     = len(sub)

    tt = feats["total_time"]
    feats["ratio_painting_time"]  = feats["time_painting"]  / tt if tt > 0 else 0
    feats["ratio_bodywork_time"]  = feats["time_bodywork"]   / tt if tt > 0 else 0
    feats["ratio_hail_time"]      = feats["time_hail"]       / tt if tt > 0 else 0
    feats["ratio_material_price"] = feats["price_material"]  / feats["total_price"] \
                                    if feats["total_price"] > 0 else 0

    feats["has_hail_cc"]     = int(feats["n_hail"] > 0)
    feats["has_painting_cc"] = int(feats["n_painting"] > 0)
    feats["has_bodywork_cc"] = int(feats["n_bodywork"] > 0)

    combined_text = " ".join(p["text"].lower() for p in positions)
    for kw, pattern in compiled_keywords.items():
        feats[kw] = int(bool(pattern.search(combined_text)))

    make_clean = (make or "unknown").strip().upper()
    feats["make_freq"] = make_freq_lookup.get(make_clean, 1)
    for m in TOP_MAKES:
        feats[f"make_{m.replace('-','_')}"] = int(make_clean == m)
    feats["make_other"] = int(make_clean not in TOP_MAKES)

    return feats


def featurize_order(record, pipeline):
    """Convert a raw order dict into the feature vector expected by the pipeline."""
    positions = preprocess_positions(record["input"]["calculatedPositions"])
    make      = record["input"].get("make", "unknown")

    text  = build_order_text(positions)
    feats = build_numeric_features(positions, make, pipeline["make_freq_lookup"])

    # Align to training feature order
    feat_series  = pd.Series(feats).reindex(pipeline["numeric_features"], fill_value=0)
    X_word       = pipeline["tfidf_word"].transform([text])
    X_char       = pipeline["tfidf_char"].transform([text])
    X_num        = sparse.csr_matrix(feat_series.values.reshape(1, -1))
    X            = sparse.hstack([X_word, X_char, X_num], format="csr")
    return X, feats, text


# ============================================================================
# PREDICTION
# ============================================================================

def predict_order(record, pipeline, verbose=True):
    """
    Run the full two-stage prediction for one order.

    Returns
    -------
    dict with keys:
        make, n_positions, total_input_time, total_input_price,
        predictions: {target: {prob, predicted_minutes, active}}
        active_targets: list of active target names
        total_predicted_minutes: float
    """
    t0 = time.perf_counter()

    X, feats, text = featurize_order(record, pipeline)
    targets        = pipeline["output_targets"]
    clf_models     = pipeline["clf_models"]
    reg_models     = pipeline["reg_models"]
    thresholds     = pipeline["thresholds"]
    clf_type_map   = pipeline["best_clf_per_target"]

    predictions = {}
    for t in targets:
        mtype  = clf_type_map[t]
        model  = clf_models[mtype][t]
        prob   = model.predict_proba(X)[0, 1]
        thr    = thresholds[mtype][t]
        active = prob >= thr

        if active:
            rm = reg_models["lgbm"].get(t) or reg_models["ridge"].get(t)
            if isinstance(rm, tuple) and rm[0] == "mean_fallback":
                duration = rm[1]
            else:
                duration = max(0.0, float(rm.predict(X)[0]))
        else:
            duration = 0.0

        predictions[t] = {
            "prob":               round(float(prob), 4),
            "active":             bool(active),
            "predicted_minutes":  round(duration, 2),
            "threshold":          round(thr, 2),
        }

    active_targets         = [t for t in targets if predictions[t]["active"]]
    total_predicted_minutes = sum(predictions[t]["predicted_minutes"] for t in targets)
    elapsed_ms             = (time.perf_counter() - t0) * 1000

    return {
        "make":                   record["input"].get("make", "unknown"),
        "n_positions":            len(record["input"]["calculatedPositions"]),
        "total_input_time_min":   round(feats["total_time"], 2),
        "total_input_price_eur":  round(feats["total_price"], 2),
        "predictions":            predictions,
        "active_targets":         active_targets,
        "total_predicted_minutes": round(total_predicted_minutes, 2),
        "elapsed_ms":             round(elapsed_ms, 1),
        "_feats":                 feats,   # for internal use only
        "_text":                  text,
    }


# ============================================================================
# EXPLANATION
# ============================================================================

def explain_prediction(result, record):
    """
    For each active target, return the top matching position texts
    and which keyword flags triggered.
    """
    positions    = record["input"]["calculatedPositions"]
    explanations = {}

    for t in result["active_targets"]:
        kw_names   = TARGET_KEYWORD_MAP.get(t, [])
        triggered  = []
        for kw in kw_names:
            if result["_feats"].get(kw, 0):
                triggered.append(kw.replace("kw_", ""))

        # Find positions whose text matches any triggered keyword
        matching_texts = []
        for p in positions:
            txt = (p.get("text") or "").strip()
            if not txt:
                continue
            for kw in kw_names:
                pat = compiled_keywords.get(kw)
                if pat and pat.search(txt):
                    matching_texts.append(txt[:80])
                    break
        matching_texts = list(dict.fromkeys(matching_texts))[:3]  # deduplicate, top 3

        explanations[t] = {
            "triggered_keywords": triggered,
            "matching_positions": matching_texts,
        }

    return explanations


# ============================================================================
# FORMATTED REPORT
# ============================================================================

BAR = "=" * 68

def print_report(result, explanations=None, true_output=None):
    print(f"\n{BAR}")
    print(f"  REPAIR ORDER PREDICTION REPORT")
    print(BAR)
    print(f"  Make            : {result['make']}")
    print(f"  Line items      : {result['n_positions']}")
    print(f"  Total input time: {result['total_input_time_min']:.1f} min")
    print(f"  Total input cost: EUR {result['total_input_price_eur']:,.2f}")
    print(f"  Inference time  : {result['elapsed_ms']:.1f} ms")

    print(f"\n  {'TARGET':<30} {'ACTIVE':>7} {'PROB':>7} {'PRED(min)':>10}"
          + ("  TRUE(min)" if true_output else ""))
    print("  " + "-" * (60 + (10 if true_output else 0)))

    targets = result["predictions"]
    for t in result["_order"]:
        p      = targets[t]
        label  = TARGET_LABELS.get(t, t)
        active = "YES" if p["active"] else "---"
        prob   = f"{p['prob']:.2f}"
        mins   = f"{p['predicted_minutes']:.2f}" if p["active"] else "   0.00"
        line   = f"  {label:<30} {active:>7} {prob:>7} {mins:>10}"
        if true_output is not None:
            true_val = true_output.get(t, 0) or 0
            true_str = f"{float(true_val):>9.2f}"
            correct  = "" if (p["active"] == (float(true_val) > 0)) else "  <-- MISS"
            line    += f"  {true_str}{correct}"
        print(line)

    print(f"\n  Total predicted repair time: {result['total_predicted_minutes']:.2f} min"
          f"  ({result['total_predicted_minutes']/60:.2f} hrs)")

    if true_output is not None:
        true_total = sum(float(v or 0) for v in true_output.values())
        error      = abs(result["total_predicted_minutes"] - true_total)
        print(f"  True total repair time     : {true_total:.2f} min"
              f"  ({true_total/60:.2f} hrs)")
        print(f"  Total time error           : {error:.2f} min")

    if explanations:
        print(f"\n  EXPLANATION - Why each work step was predicted:")
        print("  " + "-" * 60)
        for t in result["active_targets"]:
            label = TARGET_LABELS.get(t, t)
            exp   = explanations.get(t, {})
            kwds  = ", ".join(exp.get("triggered_keywords", [])) or "text pattern"
            print(f"\n  [{label}]")
            print(f"    Keywords matched : {kwds}")
            for pos in exp.get("matching_positions", []):
                print(f"    Position text    : \"{pos}\"")

    print(f"\n{BAR}\n")


def print_batch_summary(results, true_outputs=None):
    """Print a compact table summarising multiple predictions."""
    print(f"\n{BAR}")
    print(f"  BATCH PREDICTION SUMMARY  ({len(results)} orders)")
    print(BAR)
    print(f"  {'#':<4} {'Make':<20} {'Items':>5} "
          f"{'Pred Steps':>10} {'Pred Total(min)':>15}"
          + (f"  {'True Total(min)':>15} {'Error(min)':>11}" if true_outputs else ""))
    print("  " + "-" * (58 + (28 if true_outputs else 0)))

    errors = []
    for i, res in enumerate(results):
        pred_total = res["total_predicted_minutes"]
        line = (f"  {i:<4} {res['make']:<20} {res['n_positions']:>5} "
                f"{len(res['active_targets']):>10} {pred_total:>15.2f}")
        if true_outputs:
            true_total = sum(float(v or 0) for v in true_outputs[i].values())
            err        = abs(pred_total - true_total)
            errors.append(err)
            line += f"  {true_total:>15.2f} {err:>11.2f}"
        print(line)

    if errors:
        print(f"\n  Mean absolute error (total time): {np.mean(errors):.2f} min")
        print(f"  Median absolute error           : {np.median(errors):.2f} min")
        print(f"  Max absolute error              : {np.max(errors):.2f} min")
    print(f"\n{BAR}\n")


# ============================================================================
# LOAD PIPELINE
# ============================================================================

def load_pipeline(path):
    """Load and validate the saved pipeline. Build make frequency lookup."""
    print(f"  Loading pipeline from: {path}")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)

    required = ["tfidf_word", "tfidf_char", "numeric_features",
                "clf_models", "reg_models", "thresholds",
                "best_clf_per_target", "output_targets"]
    for key in required:
        if key not in pipeline:
            raise ValueError(f"Pipeline missing key: '{key}'")

    # Build make frequency lookup from tfidf vocabulary (proxy)
    # In production this would be saved; here we rebuild from data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    valid = [r for r in raw if "input" in r and "output" in r]
    make_series = pd.Series([r["input"].get("make", "unknown") or "unknown"
                              for r in valid]).str.upper()
    pipeline["make_freq_lookup"] = make_series.value_counts().to_dict()

    print(f"  Pipeline loaded. Targets: {len(pipeline['output_targets'])}")
    return pipeline, valid


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict repair work steps from a JSON order."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  type=str,
                       help="Path to a JSON file containing a single order dict.")
    group.add_argument("--demo",   type=int,
                       help="Index of a record in the training data to use as demo.")
    group.add_argument("--batch",  type=int,
                       help="Run N random records from the dataset as a batch demo.")

    parser.add_argument("--evaluate", action="store_true",
                        help="Show true labels alongside predictions (demo/batch only).")
    parser.add_argument("--explain", action="store_true", default=True,
                        help="Show keyword-based explanation (default: True).")
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    args = parser.parse_args()

    print(f"\n{BAR}")
    print(f"  Repair Order Inference Engine")
    print(BAR)

    pipeline, dataset = load_pipeline(PIPELINE_PATH)

    # ── Single file input ────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        # Wrap bare list if needed
        if isinstance(record, list):
            record = record[0]
        result              = predict_order(record, pipeline)
        result["_order"]    = pipeline["output_targets"]
        explanations        = explain_prediction(result, record) if args.explain else None
        print_report(result, explanations=explanations)

    # ── Demo single record from dataset ─────────────────────────────────────
    elif args.demo is not None:
        idx = args.demo % len(dataset)
        record = dataset[idx]
        print(f"  Demo record index: {idx} / {len(dataset)-1}")
        result           = predict_order(record, pipeline)
        result["_order"] = pipeline["output_targets"]
        explanations     = explain_prediction(result, record) if args.explain else None
        true_out         = record["output"] if args.evaluate else None
        print_report(result, explanations=explanations, true_output=true_out)

    # ── Batch demo ───────────────────────────────────────────────────────────
    elif args.batch is not None:
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(args.batch, len(dataset)),
                             replace=False)
        results     = []
        true_outputs = [] if args.evaluate else None
        for idx in indices:
            record = dataset[idx]
            res    = predict_order(record, pipeline, verbose=False)
            res["_order"] = pipeline["output_targets"]
            results.append(res)
            if args.evaluate:
                true_outputs.append(record["output"])
        print_batch_summary(results, true_outputs=true_outputs)

        # Print detail for first record
        print(f"  Detailed report for first batch record (index {indices[0]}):")
        exp = explain_prediction(results[0], dataset[indices[0]]) if args.explain else None
        print_report(results[0], explanations=exp,
                     true_output=dataset[indices[0]]["output"] if args.evaluate else None)


if __name__ == "__main__":
    main()
