"""
predict_gbert.py — gbert-base Inference Script
================================================
Loads the saved gbert-base two-stage pipeline and predicts repair work steps
and durations from a raw JSON order.

Uses BertTokenizer + BertModel directly (deepset/gbert-base lacks the
model_type key in config.json needed by the transformers AutoModel API).

Dependencies:
    pip install transformers torch scikit-learn lightgbm pandas scipy numpy

Usage:
    # Demo from dataset record index:
    python gbert_base/code/predict_gbert.py --demo 42 --evaluate

    # Predict from a JSON file:
    python gbert_base/code/predict_gbert.py --input my_order.json

    # Run a batch of N random demo orders:
    python gbert_base/code/predict_gbert.py --batch 10 --evaluate
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

try:
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("[ERROR] transformers is not installed.")
    print("        Run: pip install transformers torch")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_PATH     = Path(r"C:\Users\Administrator\baum\data\orders_simplified_sample.json")
PIPELINE_PATH = ROOT / "models" / "gbert_two_stage_pipeline.pkl"

MODEL_ID = "deepset/gbert-base"
MAX_LEN  = 256

# ── Domain config ─────────────────────────────────────────────────────────────
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
TOP_MAKES = ["VOLKSWAGEN", "MERCEDES-BENZ", "BMW", "FORD", "SKODA", "AUDI", "OPEL", "TESLA"]
TARGET_LABELS = {
    "calibration":         "Calibration (ADAS/cameras)",
    "wheelmeasurement":    "Wheel alignment measurement",
    "bodymeasurement":     "Body/chassis measurement",
    "dismounting":         "Dis-/mounting",
    "bodyrepair":          "Body repair",
    "assembly":            "Assembly",
    "plasticrepair":       "Plastic repair",
    "allTiresService":     "Tyre service",
    "cleaning":            "Cleaning",
    "paintingPreparation": "Painting — preparation",
    "paintingSpraying":    "Painting — spraying",
    "paintingFinish":      "Painting — finish",
    "hailrepair":          "Hail repair",
    "glas":                "Glass replacement",
}
TARGET_KEYWORD_MAP = {
    "calibration":         ["kw_kalibrier", "kw_sensor"],
    "wheelmeasurement":    ["kw_vermessung"],
    "bodymeasurement":     ["kw_vermessung", "kw_karosserie"],
    "dismounting":         ["kw_montage"],
    "bodyrepair":          ["kw_karosserie"],
    "assembly":            ["kw_montage"],
    "plasticrepair":       ["kw_plastik"],
    "allTiresService":     ["kw_reifen"],
    "cleaning":            ["kw_reinigung"],
    "paintingPreparation": ["kw_vorbereitung", "kw_lack"],
    "paintingSpraying":    ["kw_lack"],
    "paintingFinish":      ["kw_lack", "kw_klebetechnik"],
    "hailrepair":          ["kw_hagel", "kw_dellen"],
    "glas":                ["kw_glas", "kw_scheibe_ers"],
}
compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}
BAR = "=" * 68


# ============================================================================
# BERT EMBEDDING (single order, at inference time)
# ============================================================================

def embed_text(text, tokenizer, bert_model, device):
    """Embed a single text string → numpy array (768,), L2-normalised."""
    if not text.strip():
        text = "[leer]"
    encoded = tokenizer(
        [text], padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = bert_model(**encoded)
    mask     = encoded["attention_mask"].unsqueeze(-1).float()
    pooled   = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb      = pooled.cpu().numpy()[0]
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    return emb_norm.astype(np.float32)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def preprocess_positions(positions):
    cleaned = []
    for p in positions:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter") or "unknown_cc"
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

    feats["n_positions"]  = len(df)
    feats["n_useful"]     = len(useful)
    feats["n_zero_time"]  = (df["totalTime"] == 0).sum()
    feats["n_zero_price"] = (df["totalPrice"] == 0).sum()
    feats["total_time"]   = df["totalTime"].sum()
    feats["total_price"]  = df["totalPrice"].sum()
    feats["max_time"]     = df["totalTime"].max()
    feats["mean_time"]    = df["totalTime"].mean()
    feats["median_time"]  = df["totalTime"].median()
    feats["max_price"]    = df["totalPrice"].max()
    feats["mean_price"]   = df["totalPrice"].mean()
    feats["std_time"]     = df["totalTime"].std(ddof=0)
    feats["std_price"]    = df["totalPrice"].std(ddof=0)

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


def featurize_order(record, pipeline, tokenizer, bert_model, device):
    positions = preprocess_positions(record["input"]["calculatedPositions"])
    make      = record["input"].get("make", "unknown")
    text      = build_order_text(positions)
    feats     = build_numeric_features(positions, make, pipeline["make_freq_lookup"])
    emb       = embed_text(text, tokenizer, bert_model, device)
    feat_series = pd.Series(feats).reindex(pipeline["numeric_features"], fill_value=0)
    X_bert  = sparse.csr_matrix(emb.reshape(1, -1))
    X_num   = sparse.csr_matrix(feat_series.values.reshape(1, -1))
    X       = sparse.hstack([X_bert, X_num], format="csr")
    return X, feats, text


# ============================================================================
# PREDICTION
# ============================================================================

def predict_order(record, pipeline, tokenizer, bert_model, device):
    t0 = time.perf_counter()
    X, feats, text = featurize_order(record, pipeline, tokenizer, bert_model, device)
    targets      = pipeline["output_targets"]
    clf_models   = pipeline["clf_models"]
    reg_models   = pipeline["reg_models"]
    thresholds   = pipeline["thresholds"]
    clf_type_map = pipeline["best_clf_per_target"]

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
            "prob":            round(float(prob), 4),
            "active":          bool(active),
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


# ============================================================================
# EXPLANATION
# ============================================================================

def explain_prediction(result, record):
    positions    = record["input"]["calculatedPositions"]
    explanations = {}
    for t in result["active_targets"]:
        kw_names  = TARGET_KEYWORD_MAP.get(t, [])
        triggered = [kw.replace("kw_", "") for kw in kw_names
                     if result["_feats"].get(kw, 0)]
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
        explanations[t] = {
            "triggered_keywords": triggered,
            "matching_positions": list(dict.fromkeys(matching_texts))[:3],
        }
    return explanations


# ============================================================================
# REPORT
# ============================================================================

def print_report(result, explanations=None, true_output=None):
    print(f"\n{BAR}")
    print("  REPAIR ORDER PREDICTION REPORT  [gbert-base]")
    print(BAR)
    print(f"  Make            : {result['make']}")
    print(f"  Line items      : {result['n_positions']}")
    print(f"  Total input time: {result['total_input_time_hrs']:.2f} hrs")
    print(f"  Total input cost: EUR {result['total_input_price_eur']:,.2f}")
    print(f"  Inference time  : {result['elapsed_ms']:.1f} ms")

    print(f"\n  {'TARGET':<30} {'ACTIVE':>7} {'PROB':>7} {'PRED(hrs)':>10}"
          + ("  TRUE(hrs)" if true_output else ""))
    print("  " + "-" * (60 + (10 if true_output else 0)))

    for t in result["_order"]:
        p     = result["predictions"][t]
        label = TARGET_LABELS.get(t, t)
        hrs   = f"{p['predicted_hours']:.2f}" if p["active"] else "   0.00"
        line  = f"  {label:<30} {'YES' if p['active'] else '---':>7} {p['prob']:>7.2f} {hrs:>10}"
        if true_output is not None:
            tv   = float(true_output.get(t, 0) or 0)
            miss = "" if (p["active"] == (tv > 0)) else "  <-- MISS"
            line += f"  {tv:>9.2f}{miss}"
        print(line)

    print(f"\n  Total predicted repair time: {result['total_predicted_hours']:.2f} hrs")
    if true_output is not None:
        true_total = sum(float(v or 0) for v in true_output.values())
        error = abs(result["total_predicted_hours"] - true_total)
        print(f"  True total repair time     : {true_total:.2f} hrs")
        print(f"  Total time error           : {error:.2f} hrs")

    if explanations:
        print("\n  EXPLANATION — Why each work step was predicted:")
        print("  " + "-" * 60)
        for t in result["active_targets"]:
            exp  = explanations.get(t, {})
            kwds = ", ".join(exp.get("triggered_keywords", [])) or "text pattern"
            print(f"\n  [{TARGET_LABELS.get(t, t)}]")
            print(f"    Keywords matched : {kwds}")
            for pos in exp.get("matching_positions", []):
                print(f'    Position text    : "{pos}"')
    print(f"\n{BAR}\n")


# ============================================================================
# LOAD PIPELINE + BERT
# ============================================================================

def load_pipeline(path):
    print(f"  Loading pipeline from: {path}")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    required = ["clf_models", "reg_models", "thresholds",
                "best_clf_per_target", "output_targets",
                "numeric_features", "make_freq_lookup"]
    for key in required:
        if key not in pipeline:
            raise ValueError(f"Pipeline missing key: '{key}'")
    print(f"  Pipeline loaded. Targets: {len(pipeline['output_targets'])}")
    return pipeline


def load_bert(model_id=MODEL_ID):
    print(f"  Loading BERT tokenizer & model: {model_id} …")
    tokenizer  = BertTokenizer.from_pretrained(model_id)
    bert_model = BertModel.from_pretrained(model_id)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    print(f"  BERT ready (device: {device})\n")
    return tokenizer, bert_model, device


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="gbert-base inference: predict repair work steps."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str)
    group.add_argument("--demo",  type=int)
    group.add_argument("--batch", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--explain",    action="store_true", default=True)
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    args = parser.parse_args()

    print(f"\n{BAR}")
    print("  Repair Order Inference Engine — gbert-base")
    print(BAR)

    if not PIPELINE_PATH.exists():
        print(f"[ERROR] Pipeline not found: {PIPELINE_PATH}")
        print("  Run first: python gbert_base/code/model_gbert.py")
        sys.exit(1)

    pipeline              = load_pipeline(PIPELINE_PATH)
    tokenizer, bert_model, device = load_bert(
        pipeline.get("bert_model_id", MODEL_ID)
    )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = [r for r in raw if "input" in r and "output" in r
               and "calculatedPositions" in r["input"]]

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            record = json.load(f)
        if isinstance(record, list):
            record = record[0]
        result       = predict_order(record, pipeline, tokenizer, bert_model, device)
        explanations = explain_prediction(result, record) if args.explain else None
        print_report(result, explanations=explanations)

    elif args.demo is not None:
        idx    = args.demo % len(dataset)
        record = dataset[idx]
        print(f"  Demo record index: {idx}")
        result       = predict_order(record, pipeline, tokenizer, bert_model, device)
        explanations = explain_prediction(result, record) if args.explain else None
        true_out     = record["output"] if args.evaluate else None
        print_report(result, explanations=explanations, true_output=true_out)

    elif args.batch is not None:
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(args.batch, len(dataset)),
                             replace=False)
        results = []
        for idx in indices:
            record = dataset[int(idx)]
            res    = predict_order(record, pipeline, tokenizer, bert_model, device)
            results.append(res)

        print(f"\n  Batch of {len(results)} predictions complete.")
        total_hrs = [r["total_predicted_hours"] for r in results]
        print(f"  Mean predicted total time: {np.mean(total_hrs):.2f} hrs")

        if args.evaluate:
            errors = []
            for res, idx in zip(results, indices):
                true_total = sum(float(v or 0)
                                 for v in dataset[int(idx)]["output"].values())
                errors.append(abs(res["total_predicted_hours"] - true_total))
            print(f"  Mean absolute error (total time): {np.mean(errors):.2f} hrs")

        print(f"\n  Detailed report for first batch record (index {indices[0]}):")
        exp = explain_prediction(results[0], dataset[int(indices[0])]) if args.explain else None
        true_out_0 = dataset[int(indices[0])]["output"] if args.evaluate else None
        print_report(results[0], explanations=exp, true_output=true_out_0)


if __name__ == "__main__":
    main()
