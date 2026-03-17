"""
Log-Transform Experiment — hailrepair Stage 2 Regressor
=========================================================
This script is a targeted experiment branching from model_phase2.py.

The only change vs the baseline:
  - At TRAINING time:  y_pos for `hailrepair` is wrapped with np.log1p()
  - At PREDICTION time: the inverse np.expm1() is applied before returning

All classifiers, thresholds, and all other regression targets are
identical to the baseline pipeline.

Why log1p for hailrepair?
  hailrepair durations range from ~1 to >100 hrs — extreme skew.
  A linear regressor (Ridge) tries to minimise squared error, so large
  outliers dominate and pull predictions upward for all examples.
  log1p compresses the scale so the model fits the distribution shape
  better; expm1 then maps predictions back to the original scale.

Outputs:
  - log_transform/markdowns/log_transform_results.md
  - Console comparison table: baseline MAE vs log-transform MAE
"""

import json
import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — LightGBM regressors will be skipped.")

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
MD_DIR   = ROOT / "log_transform" / "markdowns"
MD_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── The only config change vs baseline ───────────────────────────────────────
LOG_TARGETS = {"hailrepair"}    # targets that get log1p treatment

# ── Constants (identical to model_phase2.py) ─────────────────────────────────
OUTPUT_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas",
]

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

compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n  -- {title} --")


# ============================================================================
# STEP 1: LOAD & PREPROCESS  (identical to baseline)
# ============================================================================
section("STEP 1 - Load & Preprocess")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

valid = [r for r in raw
         if "input" in r and "output" in r
         and "calculatedPositions" in r["input"]]
print(f"\n  Records loaded: {len(valid):,}")


def preprocess_record(record):
    cleaned = []
    for p in record["input"]["calculatedPositions"]:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter", None)
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({
            "text":              text,
            "totalPrice":        price,
            "totalTime":         time_,
            "genericCostCenter": cc or "unknown_cc",
        })
    return cleaned


# ============================================================================
# STEP 2: FEATURE ENGINEERING  (identical to baseline)
# ============================================================================
section("STEP 2 - Feature Engineering")

def build_order_text(positions):
    parts = []
    for p in positions:
        t = p["text"].strip()
        if t and (p["totalPrice"] > 0 or p["totalTime"] > 0):
            parts.append(t.lower())
    return " ".join(parts)

def build_numeric_features(positions, make):
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
    feats["ratio_material_price"] = (feats["price_material"] / feats["total_price"]
                                     if feats["total_price"] > 0 else 0)
    feats["has_hail_cc"]     = int(feats["n_hail"] > 0)
    feats["has_painting_cc"] = int(feats["n_painting"] > 0)
    feats["has_bodywork_cc"] = int(feats["n_bodywork"] > 0)

    combined_text = " ".join(p["text"].lower() for p in positions)
    for kw, pattern in compiled_keywords.items():
        feats[kw] = int(bool(pattern.search(combined_text)))

    make_clean = (make or "unknown").strip().upper()
    feats["make_freq_raw"] = make_clean
    for m in TOP_MAKES:
        feats[f"make_{m.replace('-','_')}"] = int(make_clean == m)
    feats["make_other"] = int(make_clean not in TOP_MAKES)

    return feats


print("\n  Building order-level features...")
records_text   = []
records_feats  = []
records_target = []

for r in valid:
    positions = preprocess_record(r)
    make      = r["input"].get("make", "unknown")
    feats     = build_numeric_features(positions, make)
    text      = build_order_text(positions)
    records_text.append(text)
    records_feats.append(feats)
    records_target.append({t: float(r["output"].get(t, 0) or 0)
                           for t in OUTPUT_TARGETS})

make_list   = [f["make_freq_raw"] for f in records_feats]
make_counts = pd.Series(make_list).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1)
    del f["make_freq_raw"]

df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)
n_records  = len(df_num)

numeric_feature_names = list(df_num.columns)
print(f"  Numeric/keyword/make features: {len(numeric_feature_names)}")

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2), max_features=2000, min_df=2,
    sublinear_tf=True, analyzer="word",
    token_pattern=r"(?u)\b[\w\+]+\b",
)
X_word = tfidf_word.fit_transform(records_text)

tfidf_char = TfidfVectorizer(
    ngram_range=(3, 5), max_features=1000, min_df=3,
    sublinear_tf=True, analyzer="char_wb",
)
X_char = tfidf_char.fit_transform(records_text)

X_num_sparse = sparse.csr_matrix(df_num.values)
X_all        = sparse.hstack([X_word, X_char, X_num_sparse], format="csr")
print(f"  Feature matrix: {X_all.shape[0]} x {X_all.shape[1]}")

Y_binary = (df_targets > 0).astype(int)
Y_reg    = df_targets.copy()


# ============================================================================
# STEP 3: TRAIN / VAL / TEST SPLIT  (identical to baseline)
# ============================================================================
section("STEP 3 - Train / Val / Test Split")

n_active       = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])
indices = np.arange(n_records)
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
cb_trainval = complexity_band[idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=cb_trainval)

X_train = X_all[idx_train];  X_val = X_all[idx_val];  X_test = X_all[idx_test]
Y_bin_train = Y_binary.iloc[idx_train]; Y_bin_val = Y_binary.iloc[idx_val]
Y_bin_test  = Y_binary.iloc[idx_test]
Y_reg_train = Y_reg.iloc[idx_train];   Y_reg_val = Y_reg.iloc[idx_val]
Y_reg_test  = Y_reg.iloc[idx_test]
print(f"\n  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")


# ============================================================================
# STEP 4: STAGE 1 — CLASSIFIERS  (identical to baseline)
# ============================================================================
section("STEP 4 - Stage 1: Classification")

def train_classifiers(X_tr, Y_tr):
    models = {"logreg": {}, "lgbm": {}}
    for t in OUTPUT_TARGETS:
        y = Y_tr[t].values
        n_pos = y.sum(); n_neg = len(y) - n_pos
        w = {0: 1.0, 1: max(1.0, n_neg / (n_pos + 1e-6))}
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                class_weight=w, random_state=RANDOM_STATE)
        lr.fit(X_tr, y)
        models["logreg"][t] = lr
        if HAS_LGB:
            scale = max(1.0, n_neg / (n_pos + 1e-6))
            lc = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                    num_leaves=31, scale_pos_weight=scale,
                                    random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
            lc.fit(X_tr, y)
            models["lgbm"][t] = lc
    return models

print("\n  Training classifiers...")
clf_models = train_classifiers(X_train, Y_bin_train)
clf_types  = [k for k in clf_models if clf_models[k]]

def get_proba(models, X, t, mtype):
    return models[mtype][t].predict_proba(X)[:, 1]

# Threshold tuning
best_thresholds = {}
val_clf_results = {}
for mtype in clf_types:
    best_thresholds[mtype] = {}
    val_clf_results[mtype] = {}
    for t in OUTPUT_TARGETS:
        y_true = Y_bin_val[t].values
        y_prob = get_proba(clf_models, X_val, t, mtype)
        best_f1, best_thr = -1, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        best_thresholds[mtype][t] = best_thr
        y_best = (y_prob >= best_thr).astype(int)
        val_clf_results[mtype][t] = {
            "F1":        round(f1_score(y_true, y_best, zero_division=0), 4),
            "threshold": round(best_thr, 2),
        }

best_clf_type_per_target = {}
for t in OUTPUT_TARGETS:
    scores = {mtype: val_clf_results[mtype][t]["F1"] for mtype in clf_types}
    best_clf_type_per_target[t] = max(scores, key=scores.get)

print("  Threshold tuning complete.")


# ============================================================================
# STEP 5: STAGE 2 — REGRESSORS  (LOG-TRANSFORM APPLIED HERE)
# ============================================================================
section("STEP 5 - Stage 2: Regression  [LOG-TRANSFORM EXPERIMENT]")

print(f"\n  LOG_TARGETS = {LOG_TARGETS}")
print("  For these targets: y_fit = np.log1p(y_pos) at train, "
      "np.expm1(y_pred) at predict.\n")

def train_regressors_with_log(X_tr, Y_tr_reg, Y_tr_bin):
    """
    Identical to baseline except:
      - For targets in LOG_TARGETS, y_pos is log1p-transformed before fitting.
    """
    models = {"ridge": {}, "lgbm": {}}
    train_stats = {}

    for t in OUTPUT_TARGETS:
        mask_pos = Y_tr_bin[t].values == 1
        n_pos    = mask_pos.sum()
        train_stats[t] = {"n_pos_train": n_pos}

        if n_pos < 5:
            mean_val = Y_tr_reg.loc[Y_tr_bin[t] == 1, t].mean()
            models["ridge"][t] = ("mean_fallback", mean_val)
            models["lgbm"][t]  = ("mean_fallback", mean_val)
            train_stats[t]["fallback"] = True
            print(f"  [WARN] {t}: only {n_pos} positive — using mean fallback")
            continue

        X_pos = X_tr[mask_pos]
        y_pos = Y_tr_reg.loc[Y_tr_bin[t] == 1, t].values
        train_stats[t]["fallback"] = False

        # ── KEY CHANGE: log1p transform for log-targets ──────────────────────
        y_fit = np.log1p(y_pos) if t in LOG_TARGETS else y_pos
        if t in LOG_TARGETS:
            print(f"  [LOG1P] {t}: raw range [{y_pos.min():.2f}, {y_pos.max():.2f}] "
                  f"→ log range [{y_fit.min():.3f}, {y_fit.max():.3f}]")
        # ────────────────────────────────────────────────────────────────────

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_pos, y_fit)
        models["ridge"][t] = ridge

        if HAS_LGB:
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=max(5, n_pos // 10),
                random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
            )
            lgb_reg.fit(X_pos, y_fit)
            models["lgbm"][t] = lgb_reg

    return models, train_stats


def reg_predict_with_log(models, X, target, model_type):
    """
    Predict and apply expm1 inverse transform for log-targets.
    """
    m = models[model_type][target]
    if isinstance(m, tuple) and m[0] == "mean_fallback":
        return np.full(X.shape[0], m[1])
    y_raw = np.maximum(0.0, m.predict(X))
    # ── KEY CHANGE: expm1 inverse transform ─────────────────────────────────
    return np.expm1(y_raw) if target in LOG_TARGETS else y_raw


print("  Training regressors...")
reg_models, reg_train_stats = train_regressors_with_log(X_train, Y_reg_train, Y_bin_train)


# ============================================================================
# STEP 6: FULL EVALUATION ON TEST SET
# ============================================================================
section("STEP 6 - End-to-End Evaluation on Test Set")

def end_to_end_predict(X, clf_models, reg_models, thresholds, clf_type_map):
    pred_bin = {}
    pred_reg = {}
    for t in OUTPUT_TARGETS:
        mtype  = clf_type_map[t]
        y_prob = get_proba(clf_models, X, t, mtype)
        thr    = thresholds[mtype][t]
        y_bin  = (y_prob >= thr).astype(int)
        y_dur  = reg_predict_with_log(reg_models, X, t, "lgbm" if HAS_LGB else "ridge")
        pred_bin[t] = y_bin
        pred_reg[t] = np.where(y_bin == 1, y_dur, 0.0)
    return pd.DataFrame(pred_bin), pd.DataFrame(pred_reg)

print("\n  Running end-to-end predictions on test set...")
Y_pred_bin, Y_pred_reg = end_to_end_predict(
    X_test, clf_models, reg_models, best_thresholds, best_clf_type_per_target)
Y_pred_bin.index = Y_bin_test.index
Y_pred_reg.index = Y_reg_test.index

# Per-target metrics
test_results = {}
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_pred_b   = Y_pred_bin[t].values
    y_true_reg = Y_reg_test[t].values
    y_pred_r   = Y_pred_reg[t].values

    f1      = f1_score(y_true_bin, y_pred_b, zero_division=0)
    mae_e2e = mean_absolute_error(y_true_reg, y_pred_r)

    tp_mask = y_true_bin == 1
    mae_pos = (mean_absolute_error(y_true_reg[tp_mask], y_pred_r[tp_mask])
               if tp_mask.sum() >= 2 else float("nan"))

    test_results[t] = {
        "F1":      round(f1, 4),
        "MAE_e2e": round(mae_e2e, 2),
        "MAE_pos": round(mae_pos, 2) if not np.isnan(mae_pos) else float("nan"),
        "n_pos":   int(tp_mask.sum()),
        "log_transformed": t in LOG_TARGETS,
    }

tdf = pd.DataFrame(test_results).T.sort_values("F1", ascending=False)
print(f"\n{tdf.to_string()}")

all_f1  = [v["F1"]      for v in test_results.values()]
all_mae = [v["MAE_e2e"] for v in test_results.values()]
print(f"\n  Macro-avg F1        : {np.nanmean(all_f1):.4f}")
print(f"  Mean MAE e2e (hrs)  : {np.nanmean(all_mae):.4f}")
print(f"  hailrepair MAE (e2e): {test_results['hailrepair']['MAE_e2e']:.2f} hrs  "
      f"(baseline: 39.60 hrs)")


# ============================================================================
# STEP 7: COMPARISON TABLE vs BASELINE
# ============================================================================
section("STEP 7 - Comparison vs Baseline")

# Baseline MAEs from model_phase2.py test run
BASELINE_MAE = {
    "paintingPreparation": 2.10,
    "cleaning":            0.48,
    "hailrepair":         39.60,
    "paintingFinish":      0.61,
    "paintingSpraying":    0.67,
    "dismounting":         1.16,
    "assembly":            1.62,
    "plasticrepair":       0.12,
    "bodyrepair":          1.44,
    "wheelmeasurement":    0.21,
    "glas":                0.09,
    "calibration":         0.39,
    "bodymeasurement":     0.08,
    "allTiresService":     0.63,
}

BASELINE_F1 = {
    "paintingPreparation": 0.9863,
    "cleaning":            0.9836,
    "hailrepair":          0.9828,
    "paintingFinish":      0.9718,
    "paintingSpraying":    0.9714,
    "dismounting":         0.9600,
    "assembly":            0.9502,
    "plasticrepair":       0.9375,
    "bodyrepair":          0.8533,
    "wheelmeasurement":    0.8519,
    "glas":                0.8000,
    "calibration":         0.6500,
    "bodymeasurement":     0.6000,
    "allTiresService":     0.2222,
}

rows = []
for t in OUTPUT_TARGETS:
    base_mae = BASELINE_MAE.get(t, float("nan"))
    new_mae  = test_results[t]["MAE_e2e"]
    delta    = new_mae - base_mae
    pct      = (delta / base_mae * 100) if base_mae > 0 else float("nan")
    rows.append({
        "target":       t,
        "baseline_MAE": base_mae,
        "log_MAE":      new_mae,
        "delta_hrs":    round(delta, 2),
        "delta_pct":    round(pct, 1),
        "log_applied":  t in LOG_TARGETS,
    })

cdf = pd.DataFrame(rows).set_index("target")
print(f"\n{cdf.to_string()}")

hail_base = BASELINE_MAE["hailrepair"]
hail_new  = test_results["hailrepair"]["MAE_e2e"]
print(f"\n  hailrepair MAE: {hail_base:.2f} → {hail_new:.2f} hrs "
      f"({(hail_new - hail_base) / hail_base * 100:+.1f}%)")
print(f"  Mean MAE all targets: {np.nanmean(list(BASELINE_MAE.values())):.2f} → "
      f"{np.nanmean(all_mae):.2f} hrs")


# ============================================================================
# STEP 8: WRITE MARKDOWN REPORT
# ============================================================================
section("STEP 8 - Write Markdown Report")

from datetime import date

hail_delta_pct = (hail_new - hail_base) / hail_base * 100
mean_base = np.nanmean(list(BASELINE_MAE.values()))
mean_new  = np.nanmean(all_mae)
mean_f1_base = np.nanmean(list(BASELINE_F1.values()))
mean_f1_new  = np.nanmean(all_f1)

# Build comparison table string
table_rows = []
for t in sorted(OUTPUT_TARGETS):
    base_mae = BASELINE_MAE.get(t, float("nan"))
    new_mae  = test_results[t]["MAE_e2e"]
    delta    = new_mae - base_mae
    pct_str  = f"{delta / base_mae * 100:+.1f}%" if base_mae > 0 else "—"
    tag      = " ← log1p" if t in LOG_TARGETS else ""
    table_rows.append(
        f"| {t:<22} | {base_mae:>12.2f} | {new_mae:>9.2f} | {delta:>+9.2f} | {pct_str:>8}{tag} |"
    )
table_str = "\n".join(table_rows)

md_content = f"""# Log-Transform Experiment — hailrepair Regressor

**Date:** {date.today()}  
**Experiment:** Replace raw `hailrepair` target with `log1p(hailrepair)` at training time,
apply `expm1()` at prediction time. All other targets and classifiers unchanged.

---

## Motivation

`hailrepair` has extreme duration skew. From the training data:

- Values range from **<1 hr** (minor dent repair) to **>100 hrs** (full hail restoration)
- A linear regressor (Ridge) minimises squared error, so large outliers dominate the fit
- On the baseline test set, `hailrepair` MAE was **39.60 hrs** — far above all other targets
- The `log1p` transform compresses this scale so the model fits the underlying distribution
  shape, not just the extreme outliers

## Why log1p / expm1?

`np.log1p(x)` = `ln(x + 1)` — safe for x = 0, maps large values to a much smaller range.  
`np.expm1(y)` = `e^y – 1` — the exact inverse, applied at prediction time to recover hours.

```
Training:   y_fit  = np.log1p(y_hours)      # compress scale
Prediction: y_pred = np.expm1(model.predict(X))  # restore hours
```

---

## Results — Per-Target Comparison (Test Set, MAE in hours)

| Target                 | Baseline MAE | Log1p MAE | Delta (hrs) | Delta (%) |
|------------------------|-------------|-----------|------------|-----------|
{table_str}

---

## Summary

| Metric | Baseline | Log-Transform | Change |
|---|---|---|---|
| `hailrepair` MAE (hrs) | {hail_base:.2f} | {hail_new:.2f} | {hail_delta_pct:+.1f}% |
| Mean MAE all 14 targets (hrs) | {mean_base:.2f} | {mean_new:.2f} | {mean_new - mean_base:+.2f} hrs |
| Macro-avg F1 | {mean_f1_base:.4f} | {mean_f1_new:.4f} | {mean_f1_new - mean_f1_base:+.4f} |

---

## Interpretation

**On `hailrepair`:** {"The log1p transform **improved** MAE from " + f"{hail_base:.2f} to {hail_new:.2f} hrs ({hail_delta_pct:+.1f}%)." if hail_new < hail_base else "The transform did not reduce MAE as expected — see discussion below."}

**On other targets:** All non-log targets are completely unaffected — the transformation is
applied exclusively to `hailrepair`, so all other MAE values should be within rounding 
tolerance of the baseline.

**F1 scores:** Identical to baseline because classifiers are not modified.

---

## Limitations & Next Steps

1. **Data scale:** With only {len(idx_train)} training records, even the log-transformed model
   has limited positive examples for `hailrepair`. More data would improve estimates substantially.

2. **Alternative distributions:** A **Tweedie** or **Gamma** regressor with a log-link natively
   handles positive-skewed targets and may outperform the Ridge + log1p approach.

3. **LightGBM with `objective="regression_l1"`** (MAE-optimised) may also handle skew better
   than the default L2 objective.

4. **Recommended:** If `hailrepair` MAE improved, integrate `LOG_TARGETS` into the main
   `model_phase2.py` pipeline and retrain the production model.
"""

md_path = MD_DIR / "log_transform_results.md"
md_path.write_text(md_content, encoding="utf-8")
print(f"\n  [saved] {md_path}")
print("\n  Experiment complete.")
