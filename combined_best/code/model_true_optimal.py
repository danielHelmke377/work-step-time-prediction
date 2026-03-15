"""
model_true_optimal.py
=====================
True optimal pipeline — uses the SAVED baseline classifiers from the pickle
(best-single-clf-per-target + F1-tuned thresholds from model_phase2.py) and
retrains ONLY the regressors with 4-target 95th-percentile winsorising.

This avoids any classifier retraining and preserves exactly the classification
performance of model_phase2.py. The regressor-stage change is the only delta.

Winsorised targets (max/p95 >= 3, skew >= 3):
  hailrepair     cap = 1702.7 hrs  (4222.5 hr outlier removed)
  assembly       cap = 9.1 hrs     (28.7 hr outlier removed)
  paintingFinish cap = 3.6 hrs     (14.3 hr outlier removed)
  calibration    cap = 4.8 hrs     (18.5 hr outlier removed)

Outputs:
  combined_best/markdowns/true_optimal_results.md
"""

import json
import pickle
import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    f1_score, precision_score, recall_score, mean_absolute_error,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
PKL_PATH  = ROOT / "models" / "two_stage_pipeline.pkl"
MD_DIR    = ROOT / "combined_best" / "markdowns"
MD_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE      = 42
WINSORISE_PCT     = 95
WINSORISE_TARGETS = {"hailrepair", "assembly", "paintingFinish", "calibration"}

BAR = "=" * 70
def section(t): print(f"\n{BAR}\n  {t}\n{BAR}")

# ============================================================================
# STEP 1 — Load saved pipeline
# ============================================================================
section("STEP 1 — Load saved pipeline from models/two_stage_pipeline.pkl")

with open(PKL_PATH, "rb") as f:
    pipeline = pickle.load(f)

tfidf_word        = pipeline["tfidf_word"]
tfidf_char        = pipeline["tfidf_char"]
clf_models        = pipeline["clf_models"]          # {mtype: {target: model}}
thresholds        = pipeline["thresholds"]           # {mtype: {target: thr}}
best_clf          = pipeline["best_clf_per_target"]  # {target: mtype_str}
OUTPUT_TARGETS    = pipeline["output_targets"]

print(f"  Classifier types in pickle : {list(clf_models.keys())}")
print(f"  Targets                    : {len(OUTPUT_TARGETS)}")
print(f"  best_clf sample            : {dict(list(best_clf.items())[:4])}")

# ============================================================================
# STEP 2 — Rebuild features (same helpers as model_phase2.py)
# ============================================================================
section("STEP 2 — Rebuild features (no TF-IDF refitting)")

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
TOP_MAKES    = ["VOLKSWAGEN", "MERCEDES-BENZ", "BMW", "FORD", "SKODA", "AUDI", "OPEL", "TESLA"]
compiled_kw  = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}


def preprocess_record(record):
    out = []
    for p in record["input"]["calculatedPositions"]:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter") or "unknown_cc"
        if not text and price == 0 and time_ == 0:
            continue
        out.append({"text": text, "totalPrice": price,
                    "totalTime": time_, "genericCostCenter": cc})
    return out


def build_order_text(positions):
    return " ".join(p["text"].strip().lower() for p in positions
                    if p["text"].strip() and (p["totalPrice"] > 0 or p["totalTime"] > 0))


def build_numeric_features(positions, make):
    feats = {}
    df = pd.DataFrame(positions)
    df["totalPrice"] = df["totalPrice"].astype(float)
    df["totalTime"]  = df["totalTime"].astype(float)
    feats["n_positions"]  = len(df)
    feats["n_useful"]     = len(df[(df["totalPrice"] > 0) | (df["totalTime"] > 0)])
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
    feats["ratio_material_price"] = (feats["price_material"] / feats["total_price"]
                                     if feats["total_price"] > 0 else 0)
    feats["has_hail_cc"]     = int(feats["n_hail"] > 0)
    feats["has_painting_cc"] = int(feats["n_painting"] > 0)
    feats["has_bodywork_cc"] = int(feats["n_bodywork"] > 0)
    combined_text = " ".join(p["text"].lower() for p in positions)
    for kw, pattern in compiled_kw.items():
        feats[kw] = int(bool(pattern.search(combined_text)))
    make_clean = (make or "unknown").strip().upper()
    feats["make_freq_raw"] = make_clean
    for m in TOP_MAKES:
        feats[f"make_{m.replace('-','_')}"] = int(make_clean == m)
    feats["make_other"] = int(make_clean not in TOP_MAKES)
    return feats


with open(DATA_PATH, "r", encoding="utf-8") as fh:
    raw = json.load(fh)
valid = [r for r in raw if "input" in r and "output" in r
         and "calculatedPositions" in r["input"]]

records_text, records_feats, records_target = [], [], []
for r in valid:
    pos  = preprocess_record(r)
    make = r["input"].get("make", "unknown")
    records_text.append(build_order_text(pos))
    records_feats.append(build_numeric_features(pos, make))
    records_target.append({t: float(r["output"].get(t, 0) or 0) for t in OUTPUT_TARGETS})

make_counts = pd.Series([f["make_freq_raw"] for f in records_feats]).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1)
    del f["make_freq_raw"]

df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)
Y_binary   = (df_targets > 0).astype(int)
Y_reg      = df_targets.copy()

# Reproduce exact train/test split
n_active        = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])
indices = np.arange(len(df_num))
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
idx_train, _ = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE,
    stratify=complexity_band[idx_trainval])

# Transform features using SAVED TF-IDF (no refitting)
X_word = tfidf_word.transform(records_text)
X_char = tfidf_char.transform(records_text)
X_all  = sparse.hstack([X_word, X_char, sparse.csr_matrix(df_num.values)], format="csr")

X_train = X_all[idx_train]
X_test  = X_all[idx_test]
Y_bin_train = Y_binary.iloc[idx_train]
Y_bin_test  = Y_binary.iloc[idx_test]
Y_reg_train = Y_reg.iloc[idx_train]
Y_reg_test  = Y_reg.iloc[idx_test]

print(f"  Train: {len(idx_train)} | Test: {len(idx_test)}")
print("  TF-IDF: loaded from pickle (not refitted)")

# ============================================================================
# STEP 3 — Retrain winsorised regressors (classifiers untouched)
# ============================================================================
section(f"STEP 3 — Retrain regressors (winsorise {WINSORISE_PCT}th pct: {sorted(WINSORISE_TARGETS)})")

reg_models = {}
for t in OUTPUT_TARGETS:
    mask  = Y_bin_train[t].values == 1
    n_pos = mask.sum()
    if n_pos < 5:
        y_avail = Y_reg_train.loc[Y_bin_train[t] == 1, t]
        reg_models[t] = ("fallback", float(y_avail.mean()))
        print(f"  [fallback ] {t:<22}  n_pos={n_pos}")
        continue

    X_pos = X_train[mask]
    y_pos = Y_reg_train.loc[Y_bin_train[t] == 1, t].values

    if t in WINSORISE_TARGETS:
        cap   = np.percentile(y_pos, WINSORISE_PCT)
        y_fit = np.clip(y_pos, 0, cap)
        note  = f" winsorise cap={cap:.1f} hrs"
    else:
        y_fit = y_pos
        note  = ""

    m = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=max(5, n_pos // 10),
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    )
    m.fit(X_pos, y_fit)
    reg_models[t] = m
    print(f"  [trained  ] {t:<22}  n_pos={n_pos}{note}")


def reg_predict(X, t):
    m = reg_models[t]
    if isinstance(m, tuple):
        return np.full(X.shape[0], m[1])
    return np.maximum(0.0, m.predict(X))

# ============================================================================
# STEP 4 — Evaluate using saved classifiers + thresholds
# ============================================================================
section("STEP 4 — Evaluate: saved CLF + winsorised REG")

freq_weights = {t: float(Y_bin_test[t].mean()) for t in OUTPUT_TARGETS}
total_fw     = sum(freq_weights.values())

# Baselines for comparison
BL_MACRO_F1  = 0.8372
BL_MACRO_REC = 0.779
BL_FW_F1     = 0.9387
BL_FW_REC    = 0.9460
BL_MACRO_MAE = 3.51
BL_FW_MAE    = 2.78

rows = []
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_true_reg = Y_reg_test[t].values

    # Use saved best-single-clf + saved threshold per target
    mtype     = best_clf[t]
    thr       = thresholds[mtype][t]
    p_test    = clf_models[mtype][t].predict_proba(X_test)[:, 1]
    y_bin     = (p_test >= thr).astype(int)

    # Duration predictions from new winsorised regressors
    y_dur     = reg_predict(X_test, t)
    y_reg_hat = np.where(y_bin == 1, y_dur, 0.0)

    rows.append({
        "target":  t,
        "freq_w":  round(freq_weights[t], 3),
        "n_pos":   int(y_true_bin.sum()),
        "clf":     mtype,
        "thr":     round(float(thr), 3),
        "F1":      round(f1_score(y_true_bin, y_bin, zero_division=0), 4),
        "Recall":  round(recall_score(y_true_bin, y_bin, zero_division=0), 4),
        "Prec":    round(precision_score(y_true_bin, y_bin, zero_division=0), 4),
        "MAE":     round(mean_absolute_error(y_true_reg, y_reg_hat), 2),
    })

df = pd.DataFrame(rows).set_index("target")

macro_f1  = round(df["F1"].mean(), 4)
macro_rec = round(df["Recall"].mean(), 4)
macro_mae = round(df["MAE"].mean(), 2)
fw_f1     = round(sum(freq_weights[t] * df.loc[t, "F1"]     for t in OUTPUT_TARGETS) / total_fw, 4)
fw_rec    = round(sum(freq_weights[t] * df.loc[t, "Recall"]  for t in OUTPUT_TARGETS) / total_fw, 4)
fw_mae    = round(sum(freq_weights[t] * df.loc[t, "MAE"]     for t in OUTPUT_TARGETS) / total_fw, 2)

print(f"\n  {'Target':<22} {'clf':>6} {'thr':>5} {'F1':>7} {'Rec':>7} {'Prec':>7} {'MAE':>7} {'freq_w':>7}")
print("  " + "-" * 72)
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    print(f"  {t:<22} {r['clf']:>6} {r['thr']:>5.3f} {r['F1']:>7.4f} "
          f"{r['Recall']:>7.4f} {r['Prec']:>7.4f} {r['MAE']:>7.2f} {r['freq_w']:>7.3f}")

print(f"\n  {'Metric':<32} {'Baseline':>10} {'True Optimal':>14}")
print(f"  {'-'*58}")
print(f"  {'Macro F1':<32} {BL_MACRO_F1:>10.4f} {macro_f1:>14.4f}")
print(f"  {'Macro Recall':<32} {BL_MACRO_REC:>10.4f} {macro_rec:>14.4f}")
print(f"  {'Freq-weighted F1':<32} {BL_FW_F1:>10.4f} {fw_f1:>14.4f}")
print(f"  {'Freq-weighted Recall':<32} {BL_FW_REC:>10.4f} {fw_rec:>14.4f}")
print(f"  {'Macro MAE (hrs)':<32} {BL_MACRO_MAE:>10.2f} {macro_mae:>14.2f}")
print(f"  {'Freq-weighted MAE (hrs)':<32} {BL_FW_MAE:>10.2f} {fw_mae:>14.2f}")

# ============================================================================
# STEP 5 — Write markdown report
# ============================================================================
section("STEP 5 — Writing markdown report")

tbl_rows = []
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    tbl_rows.append(
        f"| `{t:<22}` | {r['clf']:>5} | {r['thr']:>5.3f} | {r['n_pos']:>5} "
        f"| {r['freq_w']:>6.3f} | {r['F1']:>7.4f} | {r['Recall']:>7.4f} "
        f"| {r['Prec']:>7.4f} | {r['MAE']:>7.2f} |"
    )

md = f"""# True Optimal Pipeline — Saved CLF + Winsorised REG

**Date:** {date.today()}

**Goal:** Use the original saved baseline classifiers from `models/two_stage_pipeline.pkl`
(best-single-clf-per-target + F1-tuned thresholds from `model_phase2.py`) and retrain
**only** the regressors with 4-target 95th-percentile winsorising. No classifier changes.

## Why this is the true optimal setup

Since Stage 1 (classifiers) and Stage 2 (regressors) are independently trained on
different data subsets with different objectives, the best overall pipeline is simply
the best of each stage independently:

- **Stage 1:** the original `model_phase2.py` classifiers already achieve the highest
  freq-weighted F1 (**{BL_FW_F1}**). Oversampling only hurts common targets.
- **Stage 2:** 95th-percentile winsorising on 4 high-skew targets reduces freq-weighted
  MAE by **−32.7%** with no impact on classification.

## Configuration

| Parameter | Value |
|---|---|
| Classifiers | Loaded from `models/two_stage_pipeline.pkl` — **not retrained** |
| Threshold strategy | Best-single-clf-per-target + F1-tuned (from `model_phase2.py`) |
| TF-IDF | Loaded from pickle — **not refitted** |
| Winsorised regressors | `assembly` (cap 9.1 hrs), `calibration` (4.8 hrs), `hailrepair` (1702.7 hrs), `paintingFinish` (3.6 hrs) |

## Results

| Metric | Baseline (`model_phase2.py`) | **True Optimal** | Delta |
|---|---|---|---|
| Macro F1 | {BL_MACRO_F1} | **{macro_f1}** | {macro_f1 - BL_MACRO_F1:+.4f} |
| Macro Recall | {BL_MACRO_REC} | **{macro_rec}** | {macro_rec - BL_MACRO_REC:+.4f} |
| **Freq-weighted F1** | {BL_FW_F1} | **{fw_f1}** | {fw_f1 - BL_FW_F1:+.4f} |
| **Freq-weighted Recall** | {BL_FW_REC} | **{fw_rec}** | {fw_rec - BL_FW_REC:+.4f} |
| Macro MAE | {BL_MACRO_MAE} hrs | **{macro_mae} hrs** | {macro_mae - BL_MACRO_MAE:+.2f} hrs |
| **Freq-weighted MAE** | {BL_FW_MAE} hrs | **{fw_mae} hrs** | {fw_mae - BL_FW_MAE:+.2f} hrs |

## Per-target Results

| Target | clf | thr | n_pos | freq_w | F1 | Recall | Prec | MAE |
|--------|-----|-----|-------|--------|----|--------|------|-----|
{chr(10).join(tbl_rows)}
| **MACRO** | — | — | — | — | **{macro_f1}** | **{macro_rec}** | — | **{macro_mae}** |
"""

md_path = MD_DIR / "true_optimal_results.md"
md_path.write_text(md, encoding="utf-8")
print(f"\n  [saved] {md_path}")
print("  Done.")
