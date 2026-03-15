"""
model_oversample_recall.py
==========================
Combined Optimisation: Random Oversampling  +  Recall-Constrained Thresholds.

This script is the union of model_oversample.py (Opt-3) and model_optimise.py
(Opt-2).  It:

  1.  Loads the saved two-stage pipeline (models/two_stage_pipeline.pkl).
  2.  Rebuilds the full feature matrix from raw JSON (identical split to
      model_phase2.py).
  3.  Applies Random Oversampling on the training set so every target class
      has at least MIN_POS_OVERSAMPLE positive examples.
  4.  Retrains LogisticRegression and LightGBM on the oversampled data.
  5.  Builds a Soft-Vote Ensemble over the two retrained classifiers.
  6.  Tunes thresholds on the (un-oversampled) validation set in two ways:
        a. F1-optimal  — maximise F1 (reference baseline)
        b. Recall-Constrained — maximise Precision subject to Recall >= MIN_RECALL
  7.  Evaluates both strategies on the held-out test set.
  8.  Writes a Markdown report to markdowns/oversample_recall_results.md.

Outputs
-------
  Console         : full comparison tables
  CSV             : model_plots/oversample_recall_metrics.csv
  Markdown        : markdowns/oversample_recall_results.md
"""

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent  # repo root
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
MODEL_DIR = ROOT / "models"
PLOT_DIR  = ROOT / "model_plots"
MD_DIR    = ROOT / "markdowns"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
MIN_POS_OVERSAMPLE = 200   # Oversample positives up to this count per target
MIN_RECALL         = 0.90  # Recall constraint for Opt-2 threshold tuning

SECTION_BAR = "=" * 70

def section(title):
    print(f"\n{SECTION_BAR}")
    print(f"  {title}")
    print(SECTION_BAR)

def subsection(title):
    print(f"\n  -- {title} --")


# ============================================================================
# STEP 1: Load pipeline + Rebuild Features (identical to model_phase2.py)
# ============================================================================
section("STEP 1 - Load saved pipeline & Rebuild Feature Matrix")

pkl_path = MODEL_DIR / "two_stage_pipeline.pkl"
with open(pkl_path, "rb") as f:
    pipeline = pickle.load(f)

tfidf_word      = pipeline["tfidf_word"]
tfidf_char      = pipeline["tfidf_char"]
base_clf_models = pipeline["clf_models"]
base_thresholds = pipeline["thresholds"]
OUTPUT_TARGETS  = pipeline["output_targets"]

clf_types = ["logreg"]
if HAS_LGB and "lgbm" in base_clf_models and len(base_clf_models["lgbm"]) > 0:
    clf_types.append("lgbm")

print(f"\n  Pipeline loaded from: {pkl_path}")
print(f"  Targets             : {len(OUTPUT_TARGETS)}")
print(f"  Clf types available : {clf_types}")

# ── Feature helpers (copied verbatim from model_phase2.py) ───────────────────
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
    records_target.append({t: float(r["output"].get(t, 0) or 0) for t in OUTPUT_TARGETS})

make_list   = [f["make_freq_raw"] for f in records_feats]
make_counts = pd.Series(make_list).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1)
    del f["make_freq_raw"]

df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)
n_records  = len(df_num)

X_word       = tfidf_word.transform(records_text)
X_char       = tfidf_char.transform(records_text)
X_num_sparse = sparse.csr_matrix(df_num.values)
X_all        = sparse.hstack([X_word, X_char, X_num_sparse], format="csr")

Y_binary = (df_targets > 0).astype(int)

# Reproduce the exact same split as model_phase2.py
n_active        = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])
indices = np.arange(n_records)
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
cb_trainval = complexity_band[idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=cb_trainval)

X_train = X_all[idx_train]
X_val   = X_all[idx_val]
X_test  = X_all[idx_test]
Y_train = Y_binary.iloc[idx_train]
Y_val   = Y_binary.iloc[idx_val]
Y_test  = Y_binary.iloc[idx_test]

print(f"  Feature matrix: {X_all.shape}  |  train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")


# ============================================================================
# STEP 2: Random Oversampling + Retraining
# ============================================================================
section(f"STEP 2 - Random Oversampling (target count = {MIN_POS_OVERSAMPLE}) + Retraining")

os_clf_models = {"logreg": {}, "lgbm": {}}

for t in OUTPUT_TARGETS:
    y_tr    = Y_train[t].values
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_pos   = len(pos_idx)

    if n_pos < MIN_POS_OVERSAMPLE and n_pos > 0:
        n_extra = MIN_POS_OVERSAMPLE - n_pos
        np.random.seed(RANDOM_STATE)
        extra_pos_idx = np.random.choice(pos_idx, size=n_extra, replace=True)
        idx_resampled = np.concatenate([neg_idx, pos_idx, extra_pos_idx])
        X_tr_os = X_train[idx_resampled]
        y_tr_os = y_tr[idx_resampled]
        print(f"  [Oversampled] {t:<22}: {n_pos:>4} -> {MIN_POS_OVERSAMPLE:>4} positives")
    else:
        X_tr_os = X_train
        y_tr_os = y_tr
        print(f"  [Kept As-Is ] {t:<22}: {n_pos:>4} positives")

    new_n_pos = y_tr_os.sum()
    new_n_neg = len(y_tr_os) - new_n_pos
    w = {0: 1.0, 1: max(1.0, new_n_neg / (new_n_pos + 1e-6))}

    # Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                            class_weight=w, random_state=RANDOM_STATE)
    lr.fit(X_tr_os, y_tr_os)
    os_clf_models["logreg"][t] = lr

    # LightGBM
    if "lgbm" in clf_types:
        scale = max(1.0, new_n_neg / (new_n_pos + 1e-6))
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=scale, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=1
        )
        lgb_clf.fit(X_tr_os, y_tr_os)
        os_clf_models["lgbm"][t] = lgb_clf


# ============================================================================
# STEP 3: Build Soft-Vote Ensemble over oversampled models
# ============================================================================
section("STEP 3 - Soft-Vote Ensemble (oversampled LR + LGBM)")


def get_proba(models, X, target, mtype):
    return models[mtype][target].predict_proba(X)[:, 1]


def get_os_ensemble_proba(X, target):
    """Average oversampled LogReg and LGBM probabilities."""
    if "lgbm" in clf_types:
        p_lr  = get_proba(os_clf_models, X, target, "logreg")
        p_lgb = get_proba(os_clf_models, X, target, "lgbm")
        return (p_lr + p_lgb) / 2.0
    else:
        return get_proba(os_clf_models, X, target, "logreg")


# ============================================================================
# STEP 4: Tune thresholds on validation set — F1-optimal & Recall-Constrained
# ============================================================================
section(f"STEP 4 - Threshold Tuning on Validation (F1-optimal vs Recall >= {MIN_RECALL})")

results_rows = []

for t in OUTPUT_TARGETS:
    y_val_true = Y_val[t].values
    y_prob_val = get_os_ensemble_proba(X_val, t)

    # --- F1-optimal threshold ---
    best_f1, best_thr_f1 = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob_val >= thr).astype(int)
        f1 = f1_score(y_val_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr

    y_f1 = (y_prob_val >= best_thr_f1).astype(int)

    # --- Recall-Constrained threshold ---
    best_prec_rc, best_thr_rc = -1, None
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob_val >= thr).astype(int)
        rec  = recall_score(y_val_true, y_pred, zero_division=0)
        prec = precision_score(y_val_true, y_pred, zero_division=0)
        if rec >= MIN_RECALL and prec > best_prec_rc:
            best_prec_rc = prec
            best_thr_rc  = thr

    status = "OK"
    if best_thr_rc is None:
        best_thr_rc = 0.01   # fallback: predict almost everything positive
        status = "MIN_RECALL_UNACHIEVABLE"

    y_rc = (y_prob_val >= best_thr_rc).astype(int)

    results_rows.append({
        "target":   t,
        # validation metrics
        "thr_F1":   round(best_thr_f1, 2),
        "ValPrec_F1": round(precision_score(y_val_true, y_f1, zero_division=0), 4),
        "ValRec_F1":  round(recall_score(y_val_true, y_f1, zero_division=0), 4),
        "ValF1_F1":   round(f1_score(y_val_true, y_f1, zero_division=0), 4),
        "thr_RC":   round(best_thr_rc, 2),
        "ValPrec_RC": round(precision_score(y_val_true, y_rc, zero_division=0), 4),
        "ValRec_RC":  round(recall_score(y_val_true, y_rc, zero_division=0), 4),
        "ValF1_RC":   round(f1_score(y_val_true, y_rc, zero_division=0), 4),
        "status":   status,
        # store thresholds for test evaluation
        "_thr_f1_val": best_thr_f1,
        "_thr_rc_val": best_thr_rc,
    })

df_val = pd.DataFrame(results_rows).set_index("target")
display_cols = ["thr_F1","ValPrec_F1","ValRec_F1","ValF1_F1",
                "thr_RC","ValPrec_RC","ValRec_RC","ValF1_RC","status"]
print("\n" + df_val[display_cols].to_string())

unachievable = df_val[df_val["status"] == "MIN_RECALL_UNACHIEVABLE"].index.tolist()
if unachievable:
    print(f"\n  [WARN] Recall >= {MIN_RECALL} not achievable for: {unachievable}")


# ============================================================================
# STEP 5: Test Set Evaluation
# ============================================================================
section("STEP 5 - Test Set Evaluation")

test_rows = []
for t in OUTPUT_TARGETS:
    y_test_true = Y_test[t].values
    y_prob_test = get_os_ensemble_proba(X_test, t)

    thr_f1 = df_val.loc[t, "_thr_f1_val"]
    thr_rc = df_val.loc[t, "_thr_rc_val"]

    y_f1 = (y_prob_test >= thr_f1).astype(int)
    y_rc = (y_prob_test >= thr_rc).astype(int)

    test_rows.append({
        "target":       t,
        "n_pos_test":   int(y_test_true.sum()),
        # F1-optimal
        "Prec_F1":  round(precision_score(y_test_true, y_f1, zero_division=0), 4),
        "Rec_F1":   round(recall_score(y_test_true, y_f1, zero_division=0), 4),
        "Acc_F1":   round(accuracy_score(y_test_true, y_f1), 4),
        "F1_F1":    round(f1_score(y_test_true, y_f1, zero_division=0), 4),
        # Recall-constrained
        "Prec_RC":  round(precision_score(y_test_true, y_rc, zero_division=0), 4),
        "Rec_RC":   round(recall_score(y_test_true, y_rc, zero_division=0), 4),
        "Acc_RC":   round(accuracy_score(y_test_true, y_rc), 4),
        "F1_RC":    round(f1_score(y_test_true, y_rc, zero_division=0), 4),
        "status":   df_val.loc[t, "status"],
    })

df_test = pd.DataFrame(test_rows).set_index("target")
print("\n" + df_test[["Prec_F1","Rec_F1","F1_F1","Prec_RC","Rec_RC","F1_RC"]].to_string())

# Macro averages
for strat, prec_col, rec_col, f1_col in [
    ("F1-optimal  ", "Prec_F1", "Rec_F1", "F1_F1"),
    ("Recall>=.90 ", "Prec_RC", "Rec_RC", "F1_RC"),
]:
    print(f"\n  Test Macro [{strat}] Prec={df_test[prec_col].mean():.4f}  "
          f"Rec={df_test[rec_col].mean():.4f}  F1={df_test[f1_col].mean():.4f}")

# Save CSV
csv_path = PLOT_DIR / "oversample_recall_metrics.csv"
df_test.to_csv(csv_path)
print(f"\n  [saved] {csv_path}")


# ============================================================================
# STEP 6: Markdown Report
# ============================================================================
section("STEP 6 - Writing Markdown Report")

mac_prec_f1 = df_test["Prec_F1"].mean()
mac_rec_f1  = df_test["Rec_F1"].mean()
mac_f1_f1   = df_test["F1_F1"].mean()
mac_prec_rc = df_test["Prec_RC"].mean()
mac_rec_rc  = df_test["Rec_RC"].mean()
mac_f1_rc   = df_test["F1_RC"].mean()

md = f"""# Combined Optimisation — Oversampling + Recall-Constrained Thresholds

**Script**: `code/model_oversample_recall.py`  
**Date**: 2026-03-14  

## Approach

This experiment stacks two optimisations that were previously run independently:

| Step | Technique | Purpose |
|---|---|---|
| 1 | **Random Oversampling** (target ≥ {MIN_POS_OVERSAMPLE} positives) | Help rare-class classifiers learn better decision boundaries |
| 2 | **Soft-Vote Ensemble** (oversampled LR + LGBM) | Reduce variance across model families |
| 3 | **Recall-Constrained Threshold Tuning** (Recall ≥ {MIN_RECALL}) | Guarantee high recall in production (missing a work step = under-quote) |

The thresholds are tuned on an un-oversampled validation set to avoid
data-leakage artefacts, then evaluated on the held-out test set.

---

## Test Set Results

### F1-Optimal Thresholds

| Target | Test Pos | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
"""

for t, row in df_test.iterrows():
    md += f"| {t} | {row['n_pos_test']} | {row['Prec_F1']:.4f} | {row['Rec_F1']:.4f} | {row['F1_F1']:.4f} |\n"

md += f"""
**Macro-Average (F1-optimal):**  Precision = {mac_prec_f1:.4f} | Recall = {mac_rec_f1:.4f} | F1 = {mac_f1_f1:.4f}

---

### Recall-Constrained Thresholds (Recall ≥ {MIN_RECALL})

| Target | Test Pos | Precision | Recall | F1 | Status |
|---|:---:|:---:|:---:|:---:|:---:|
"""

for t, row in df_test.iterrows():
    md += (f"| {t} | {row['n_pos_test']} | {row['Prec_RC']:.4f} | "
           f"{row['Rec_RC']:.4f} | {row['F1_RC']:.4f} | {row['status']} |\n")

md += f"""
**Macro-Average (Recall-constrained):**  Precision = {mac_prec_rc:.4f} | Recall = {mac_rec_rc:.4f} | F1 = {mac_f1_rc:.4f}

---

## Summary Table

| Strategy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|
| Oversampled — F1-Optimal | {mac_prec_f1:.4f} | {mac_rec_f1:.4f} | {mac_f1_f1:.4f} |
| **Oversampled — Recall ≥ {MIN_RECALL}** | **{mac_prec_rc:.4f}** | **{mac_rec_rc:.4f}** | **{mac_f1_rc:.4f}** |

> **Trade-off**: The recall-constrained strategy raises macro Recall
> at the cost of lower Precision.  Use it whenever missing a work step
> is more costly than a false positive prediction.
"""

if unachievable:
    md += f"""
> [!WARNING]
> Recall ≥ {MIN_RECALL} is **not achievable** for: `{'`, `'.join(unachievable)}`  
> These targets have too few positive validation examples.  
> Consider gathering more data or treating them as low-confidence predictions.
"""

md += f"""
---

## Saved Outputs

| File | Description |
|---|---|
| `model_plots/oversample_recall_metrics.csv` | Full per-target test metrics for both threshold strategies |

*Generated by `code/model_oversample_recall.py`.*
"""

md_path = MD_DIR / "oversample_recall_results.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\n  [saved] {md_path}")

section("DONE")
print(f"""
  +--------------------------------------------------------------+
  |  COMBINED OPTIMISATION COMPLETE                              |
  |  Oversampling:  >= {MIN_POS_OVERSAMPLE} positives per target               |
  |  Recall constraint: >= {MIN_RECALL}                                 |
  |  Macro F1  (F1-optimal)       : {mac_f1_f1:.4f}                   |
  |  Macro F1  (Recall-constr.)   : {mac_f1_rc:.4f}                   |
  |  Macro Recall (Recall-constr.): {mac_rec_rc:.4f}                   |
  +--------------------------------------------------------------+
""")
