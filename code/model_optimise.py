"""
model_optimise.py
=================
Optimisation layer on top of the trained Stage 1 classifiers.

Loads the saved two-stage pipeline from models/two_stage_pipeline.pkl
(produced by model_phase2.py) and applies two improvements without
touching the original training script:

  Opt-1  Soft-vote Ensemble
         Average LogReg + LightGBM predicted probabilities per target,
         tune a new threshold on the validation set, and compare F1
         against the individual models.

  Opt-2  Recall-Constrained Threshold Tuning
         For each target find the threshold that maximises Precision
         subject to Recall >= MIN_RECALL (default 0.90).
         High recall is preferred in the body-shop context because
         missing a work step leads to under-quoting the job.

Outputs
-------
  Console        : full comparison tables
  CSV            : model_plots/ensemble_vs_individual.csv
  CSV            : model_plots/recall_constrained_thresholds.csv
  Markdown       : markdowns/optimisation_results.md
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent  # repo root
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
MODEL_DIR = ROOT / "models"
PLOT_DIR  = ROOT / "model_plots"
MD_DIR    = ROOT / "markdowns"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MIN_RECALL   = 0.90   # Opt-2: minimum acceptable recall per target

SECTION_BAR = "=" * 70

def section(title):
    print(f"\n{SECTION_BAR}")
    print(f"  {title}")
    print(SECTION_BAR)

def subsection(title):
    print(f"\n  -- {title} --")

# ============================================================================
# STEP 1: Load pipeline + raw data (needed to rebuild X_val / X_test)
# ============================================================================
section("STEP 1 - Load saved pipeline")

pkl_path = MODEL_DIR / "two_stage_pipeline.pkl"
with open(pkl_path, "rb") as f:
    pipeline = pickle.load(f)

tfidf_word          = pipeline["tfidf_word"]
tfidf_char          = pipeline["tfidf_char"]
clf_models          = pipeline["clf_models"]
thresholds_f1       = pipeline["thresholds"]          # F1-optimal thresholds
best_clf_per_target = pipeline["best_clf_per_target"]
OUTPUT_TARGETS      = pipeline["output_targets"]
numeric_features    = pipeline["numeric_features"]

print(f"\n  Pipeline loaded from: {pkl_path}")
print(f"  Targets : {len(OUTPUT_TARGETS)}")
print(f"  Clf types available: {list(clf_models.keys())}")

HAS_LGB = "lgbm" in clf_models and bool(clf_models["lgbm"])

# ── Reconstruct features from raw data ────────────────────────────────────────
import re

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
    cleaned = []
    for p in record["input"]["calculatedPositions"]:
        text  = (p.get("text") or "").strip()
        price = max(0.0, float(p.get("totalPrice") or 0))
        time_ = float(p.get("totalTime") or 0)
        cc    = p.get("genericCostCenter", None)
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({"text": text, "totalPrice": price,
                         "totalTime": time_, "genericCostCenter": cc or "unknown_cc"})
    return cleaned

def build_order_text(positions):
    return " ".join(
        p["text"].strip().lower()
        for p in positions
        if p["text"].strip() and (p["totalPrice"] > 0 or p["totalTime"] > 0)
    )

def build_numeric_features(positions, make):
    feats = {}
    df = pd.DataFrame(positions)
    df["totalPrice"] = df["totalPrice"].astype(float)
    df["totalTime"]  = df["totalTime"].astype(float)
    useful = df[(df["totalPrice"] > 0) | (df["totalTime"] > 0)]

    feats["n_positions"] = len(df)
    feats["n_useful"]    = len(useful)
    feats["n_zero_time"] = (df["totalTime"] == 0).sum()
    feats["n_zero_price"]= (df["totalPrice"] == 0).sum()

    feats["total_time"]  = df["totalTime"].sum()
    feats["total_price"] = df["totalPrice"].sum()
    feats["max_time"]    = df["totalTime"].max()
    feats["mean_time"]   = df["totalTime"].mean()
    feats["median_time"] = df["totalTime"].median()
    feats["max_price"]   = df["totalPrice"].max()
    feats["mean_price"]  = df["totalPrice"].mean()
    feats["std_time"]    = df["totalTime"].std(ddof=0)
    feats["std_price"]   = df["totalPrice"].std(ddof=0)

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

print("\n  Rebuilding feature matrix from raw data...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

valid = [r for r in raw
         if "input" in r and "output" in r
         and "calculatedPositions" in r["input"]]

records_text, records_feats, records_target = [], [], []
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

X_word      = tfidf_word.transform(records_text)
X_char      = tfidf_char.transform(records_text)
X_num_sparse = sparse.csr_matrix(df_num.values)
X_all        = sparse.hstack([X_word, X_char, X_num_sparse], format="csr")

Y_binary = (df_targets > 0).astype(int)

# Reproduce same split (must match model_phase2.py exactly)
from sklearn.model_selection import train_test_split
n_active       = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                          labels=["low", "medium", "high", "very_high"])
indices = np.arange(n_records)
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
cb_trainval = complexity_band[idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=cb_trainval)

X_val  = X_all[idx_val]
X_test = X_all[idx_test]
Y_bin_val  = Y_binary.iloc[idx_val]
Y_bin_test = Y_binary.iloc[idx_test]

print(f"  Feature matrix: {X_all.shape}  |  val={len(idx_val)}  test={len(idx_test)}")


# ============================================================================
# OPT-1: SOFT-VOTE ENSEMBLE
# ============================================================================
section("OPT-1 - Soft-Vote Ensemble (LogReg + LightGBM)")

def get_proba(clf_models, X, target, mtype):
    return clf_models[mtype][target].predict_proba(X)[:, 1]

def get_ensemble_proba(clf_models, X, target):
    """Average LogReg and LightGBM probabilities (equal weight)."""
    p_lr  = get_proba(clf_models, X, target, "logreg")
    p_lgb = get_proba(clf_models, X, target, "lgbm")
    return (p_lr + p_lgb) / 2.0

subsection("Tuning ensemble thresholds on validation set (F1-optimal)")

ensemble_thresholds = {}
ensemble_val_results = {}

for t in OUTPUT_TARGETS:
    y_true  = Y_bin_val[t].values
    y_prob  = get_ensemble_proba(clf_models, X_val, t)

    best_f1, best_thr = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    ensemble_thresholds[t] = best_thr
    y_best = (y_prob >= best_thr).astype(int)
    ensemble_val_results[t] = {
        "thr_ensemble": round(best_thr, 2),
        "F1_ensemble":  round(f1_score(y_true, y_best, zero_division=0), 4),
        "Prec_ensemble":round(precision_score(y_true, y_best, zero_division=0), 4),
        "Rec_ensemble": round(recall_score(y_true, y_best, zero_division=0), 4),
    }

subsection("Comparison: LogReg vs LightGBM vs Ensemble (Validation F1)")

compare_rows = []
for t in OUTPUT_TARGETS:
    f1_lr  = thresholds_f1["logreg"] and round(
        f1_score(Y_bin_val[t].values,
                 (get_proba(clf_models, X_val, t, "logreg") >= thresholds_f1["logreg"][t]).astype(int),
                 zero_division=0), 4)
    f1_lgb = (round(
        f1_score(Y_bin_val[t].values,
                 (get_proba(clf_models, X_val, t, "lgbm") >= thresholds_f1["lgbm"][t]).astype(int),
                 zero_division=0), 4) if HAS_LGB else float("nan"))
    f1_ens = ensemble_val_results[t]["F1_ensemble"]
    best   = max(
        {"logreg": f1_lr, "lgbm": f1_lgb, "ensemble": f1_ens}.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else -1
    )[0]
    compare_rows.append({
        "target":   t,
        "F1_logreg": f1_lr,
        "F1_lgbm":   f1_lgb,
        "F1_ensemble": f1_ens,
        "best_model":  best,
    })

df_compare = pd.DataFrame(compare_rows).set_index("target")
print("\n" + df_compare.to_string())

# How many targets does ensemble win?
wins = (df_compare["best_model"] == "ensemble").sum()
print(f"\n  Ensemble is best for {wins}/{len(OUTPUT_TARGETS)} targets on val set.")

# Macro F1 comparison
for col in ["F1_logreg", "F1_lgbm", "F1_ensemble"]:
    print(f"  Macro {col}: {df_compare[col].mean():.4f}")

# ---- Evaluate ensemble on TEST set ----
subsection("Ensemble performance on Test Set")

test_ensemble_rows = []
for t in OUTPUT_TARGETS:
    y_true = Y_bin_test[t].values
    y_prob = get_ensemble_proba(clf_models, X_test, t)
    thr    = ensemble_thresholds[t]
    y_pred = (y_prob >= thr).astype(int)
    test_ensemble_rows.append({
        "target":    t,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Threshold": round(thr, 2),
    })

df_ens_test = pd.DataFrame(test_ensemble_rows).set_index("target")
print("\n" + df_ens_test.sort_values("F1", ascending=False).to_string())
print(f"\n  Macro Precision : {df_ens_test['Precision'].mean():.4f}")
print(f"  Macro Recall    : {df_ens_test['Recall'].mean():.4f}")
print(f"  Macro Accuracy  : {df_ens_test['Accuracy'].mean():.4f}")
print(f"  Macro F1        : {df_ens_test['F1'].mean():.4f}")

# Save
df_compare.to_csv(PLOT_DIR / "ensemble_vs_individual.csv")
df_ens_test["split"] = "test"
df_ens_test.to_csv(PLOT_DIR / "ensemble_test_metrics.csv")
print(f"\n  [saved] {PLOT_DIR / 'ensemble_vs_individual.csv'}")
print(f"  [saved] {PLOT_DIR / 'ensemble_test_metrics.csv'}")


# ============================================================================
# OPT-2: RECALL-CONSTRAINED THRESHOLD TUNING
# ============================================================================
section(f"OPT-2 - Recall-Constrained Threshold Tuning (Recall >= {MIN_RECALL})")

subsection("Finding Precision-maximising threshold at Recall >= 0.90 per target")

recall_rows = []
for t in OUTPUT_TARGETS:
    y_true = Y_bin_val[t].values
    y_prob = get_ensemble_proba(clf_models, X_val, t)

    # F1-optimal (reference)
    thr_f1    = ensemble_thresholds[t]
    y_f1      = (y_prob >= thr_f1).astype(int)

    # Recall-constrained: find best Precision with Recall >= MIN_RECALL
    best_prec, best_thr_rc = -1, None
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        prec   = precision_score(y_true, y_pred, zero_division=0)
        if rec >= MIN_RECALL and prec > best_prec:
            best_prec   = prec
            best_thr_rc = thr

    if best_thr_rc is None:
        # Recall >= 0.90 not achievable: use lowest threshold
        best_thr_rc = 0.01
        y_rc = (y_prob >= best_thr_rc).astype(int)
        note = "MIN_RECALL_UNACHIEVABLE"
    else:
        y_rc = (y_prob >= best_thr_rc).astype(int)
        note = "OK"

    recall_rows.append({
        "target":         t,
        # F1-optimal (ensemble)
        "thr_F1":         round(thr_f1, 2),
        "Prec_F1":        round(precision_score(y_true, y_f1, zero_division=0), 4),
        "Rec_F1":         round(recall_score(y_true, y_f1, zero_division=0), 4),
        "F1_F1":          round(f1_score(y_true, y_f1, zero_division=0), 4),
        # Recall-constrained
        "thr_RC":         round(best_thr_rc, 2),
        "Prec_RC":        round(precision_score(y_true, y_rc, zero_division=0), 4),
        "Rec_RC":         round(recall_score(y_true, y_rc, zero_division=0), 4),
        "F1_RC":          round(f1_score(y_true, y_rc, zero_division=0), 4),
        "status":         note,
    })

df_rc = pd.DataFrame(recall_rows).set_index("target")
print("\n" + df_rc.to_string())

unachievable = df_rc[df_rc["status"] == "MIN_RECALL_UNACHIEVABLE"].index.tolist()
if unachievable:
    print(f"\n  [WARN] Recall >= {MIN_RECALL} not achievable for: {unachievable}")
    print("         These targets have too few positive validation examples.")

print(f"\n  Macro Precision  F1-optimal : {df_rc['Prec_F1'].mean():.4f}")
print(f"  Macro Precision  Recall>=.90: {df_rc['Prec_RC'].mean():.4f}")
print(f"  Macro Recall     F1-optimal : {df_rc['Rec_F1'].mean():.4f}")
print(f"  Macro Recall     Recall>=.90: {df_rc['Rec_RC'].mean():.4f}")
print(f"  Macro F1         F1-optimal : {df_rc['F1_F1'].mean():.4f}")
print(f"  Macro F1         Recall>=.90: {df_rc['F1_RC'].mean():.4f}")

df_rc.to_csv(PLOT_DIR / "recall_constrained_thresholds.csv")
print(f"\n  [saved] {PLOT_DIR / 'recall_constrained_thresholds.csv'}")


# ============================================================================
# MARKDOWN REPORT
# ============================================================================
section("Writing Markdown Report")

# Compute deltas for headline summary
macro_f1_lr   = df_compare["F1_logreg"].mean()
macro_f1_lgb  = df_compare["F1_lgbm"].mean()
macro_f1_ens  = df_compare["F1_ensemble"].mean()

macro_prec_f1 = df_rc["Prec_F1"].mean()
macro_prec_rc = df_rc["Prec_RC"].mean()
macro_rec_f1  = df_rc["Rec_F1"].mean()
macro_rec_rc  = df_rc["Rec_RC"].mean()
macro_f1_f1   = df_rc["F1_F1"].mean()
macro_f1_rc   = df_rc["F1_RC"].mean()

# Per-target comparison for ensemble section
ens_md_rows = df_compare[["F1_logreg","F1_lgbm","F1_ensemble","best_model"]].copy()
ens_test_md = df_ens_test[["Precision","Recall","Accuracy","F1"]].copy()

md = f"""# Stage 1 Classifier — Optimisation Results

**Script**: `code/model_optimise.py`  
**Date**: 2026-03-14  
**Two optimisations applied to the trained Stage 1 classifiers:**

1. **Soft-Vote Ensemble** — average LogReg + LightGBM predicted probabilities before threshold application
2. **Recall-Constrained Threshold Tuning** — find threshold that maximises Precision while keeping Recall ≥ {MIN_RECALL}

---

## Optimisation 1 — Soft-Vote Ensemble

### What it does
Instead of choosing *either* Logistic Regression *or* LightGBM (whichever had the higher validation F1), the ensemble averages their probability scores:

```
p_ensemble = (p_logreg + p_lgbm) / 2
```

A new threshold is then tuned on the validation set to maximise F1.

### Validation F1 Comparison (per target)

| Target | F1 LogReg | F1 LightGBM | F1 Ensemble | Best |
|---|:---:|:---:|:---:|:---|
"""

for t, row in ens_md_rows.iterrows():
    best_marker = lambda x, col: f"**{x:.4f}**" if col == row["best_model"] else f"{x:.4f}"
    md += (f"| {t} | {best_marker(row['F1_logreg'],'logreg')} | "
           f"{best_marker(row['F1_lgbm'],'lgbm')} | "
           f"{best_marker(row['F1_ensemble'],'ensemble')} | "
           f"{row['best_model']} |\n")

md += f"""
### Macro-Average F1 (Validation)

| Model | Macro F1 |
|---|:---:|
| Logistic Regression | {macro_f1_lr:.4f} |
| LightGBM | {macro_f1_lgb:.4f} |
| **Ensemble** | **{macro_f1_ens:.4f}** |

The ensemble is the best single model for **{wins} out of {len(OUTPUT_TARGETS)} targets**.

### Ensemble Test Set Results

| Target | Precision | Recall | Accuracy | F1 |
|---|:---:|:---:|:---:|:---:|
"""

for t, row in ens_test_md.sort_values("F1", ascending=False).iterrows():
    md += f"| {t} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['Accuracy']:.4f} | {row['F1']:.4f} |\n"

md += f"""
**Overall macro-averages (Test Set):**

| Metric | Value |
|---|:---:|
| Macro Precision | {df_ens_test['Precision'].mean():.4f} |
| Macro Recall | {df_ens_test['Recall'].mean():.4f} |
| Macro Accuracy | {df_ens_test['Accuracy'].mean():.4f} |
| **Macro F1** | **{df_ens_test['F1'].mean():.4f}** |

---

## Optimisation 2 — Recall-Constrained Threshold Tuning

### What it does
In a body-shop context, **missing a work step (false negative) is more costly than
a false alarm (false positive)**: an under-quoted job causes financial loss.

The standard F1-optimal threshold balances precision and recall equally.
This optimisation instead finds, for each target, the **lowest threshold at which
Recall ≥ {MIN_RECALL}** — then picks the one that gives the highest Precision
within that constraint.

### Comparison: F1-Optimal vs Recall-Constrained (Validation, Ensemble probabilities)

| Target | Thr F1 | Prec | Rec | F1 | Thr RC | Prec | Rec | F1 | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
"""

for t, row in df_rc.iterrows():
    md += (f"| {t} | {row['thr_F1']:.2f} | {row['Prec_F1']:.4f} | "
           f"{row['Rec_F1']:.4f} | {row['F1_F1']:.4f} | "
           f"{row['thr_RC']:.2f} | {row['Prec_RC']:.4f} | "
           f"{row['Rec_RC']:.4f} | {row['F1_RC']:.4f} | {row['status']} |\n")

md += f"""
### Macro-Average Summary

| Strategy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|
| F1-Optimal (ensemble) | {macro_prec_f1:.4f} | {macro_rec_f1:.4f} | {macro_f1_f1:.4f} |
| **Recall-Constrained (≥{MIN_RECALL})** | **{macro_prec_rc:.4f}** | **{macro_rec_rc:.4f}** | **{macro_f1_rc:.4f}** |

> **Trade-off**: The recall-constrained strategy raises macro Recall to ≥ {macro_rec_rc:.2f}
> at the cost of lower Precision ({macro_prec_rc:.4f} vs {macro_prec_f1:.4f}).
> Use this strategy when under-quoting is unacceptable.
"""

if unachievable:
    md += f"""
> [!WARNING]
> Recall ≥ {MIN_RECALL} is **not achievable** for: `{'`, `'.join(unachievable)}`  
> These targets have very few positive examples in the validation set.
> Consider gathering more training data or treating them as low-confidence predictions.
"""

md += f"""
---

## Saved Outputs

| File | Description |
|---|---|
| `model_plots/ensemble_vs_individual.csv` | F1 comparison across all three model types per target |
| `model_plots/ensemble_test_metrics.csv` | Full ensemble test metrics (Precision, Recall, Accuracy, F1) |
| `model_plots/recall_constrained_thresholds.csv` | F1-optimal vs Recall-constrained threshold comparison |

*Generated by `code/model_optimise.py`.*
"""

md_path = MD_DIR / "optimisation_results.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\n  [saved] {md_path}")

section("DONE")
print(f"""
  +------------------------------------------------------------+
  |  OPTIMISATION COMPLETE                                     |
  |  Ensemble best for {wins:>2}/{len(OUTPUT_TARGETS)} targets (val)                    |
  |  Ensemble macro F1 (val) : {macro_f1_ens:.4f}                      |
  |  Ensemble macro F1 (test): {df_ens_test['F1'].mean():.4f}                      |
  |  Recall-constrained macro recall: {macro_rec_rc:.4f}                 |
  +------------------------------------------------------------+
""")
