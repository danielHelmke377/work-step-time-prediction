"""
model_oversample.py
===================
Optimisation 3: Random Oversampling for Minority Classes.

This script rebuilds the training data from the raw JSON, applies the
same feature extraction pipeline as model_phase2.py, and then implements
Random Oversampling for any target with sparse positive examples in the
training set.

It trains new LogisticRegression and LightGBM models on the oversampled
data, tunes thresholds on the validation set, and reports Precision, Recall,
Accuracy, and F1 on the test set — comparing against the baseline models.
"""

import json
import pickle
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
BASE      = Path(r"C:\Users\Administrator\baum")
DATA_PATH = BASE / "data" / "orders_simplified_sample.json"
MODEL_DIR = BASE / "models"
PLOT_DIR  = BASE / "model_plots"
MD_DIR    = BASE / "markdowns"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
MIN_POS_OVERSAMPLE = 200  # Number of positive examples to reach via oversampling
                          # (if a class naturally has more, it is not oversampled)

SECTION_BAR = "=" * 70

def section(title):
    print(f"\n{SECTION_BAR}")
    print(f"  {title}")
    print(SECTION_BAR)

def subsection(title):
    print(f"\n  -- {title} --")

# ============================================================================
# STEP 1: Load pipeline + Rebuild Features (Identical to phase 2)
# ============================================================================
section("STEP 1 - Load pipeline & Rebuild Feature Matrix")

pkl_path = MODEL_DIR / "two_stage_pipeline.pkl"
with open(pkl_path, "rb") as f:
    pipeline = pickle.load(f)

tfidf_word          = pipeline["tfidf_word"]
tfidf_char          = pipeline["tfidf_char"]
base_clf_models     = pipeline["clf_models"]          # The original trained models
base_thresholds     = pipeline["thresholds"]
OUTPUT_TARGETS      = pipeline["output_targets"]

# We only need 'lgbm' if it trained successfully
clf_types = ["logreg"]
if HAS_LGB and "lgbm" in base_clf_models and len(base_clf_models["lgbm"]) > 0:
    clf_types.append("lgbm")

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
        cleaned.append({"text": text, "totalPrice": price, "totalTime": time_, "genericCostCenter": cc or "unknown_cc"})
    return cleaned

def build_order_text(positions):
    return " ".join(p["text"].strip().lower() for p in positions if p["text"].strip() and (p["totalPrice"] > 0 or p["totalTime"] > 0))

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
    feats["ratio_material_price"] = feats["price_material"] / feats["total_price"] if feats["total_price"] > 0 else 0

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

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

valid = [r for r in raw if "input" in r and "output" in r and "calculatedPositions" in r["input"]]
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

# Exact same split definition
n_active       = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100], labels=["low", "medium", "high", "very_high"])
indices = np.arange(n_records)
idx_trainval, idx_test = train_test_split(indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
cb_trainval = complexity_band[idx_trainval]
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=cb_trainval)

X_train = X_all[idx_train]
X_val   = X_all[idx_val]
X_test  = X_all[idx_test]

Y_train = Y_binary.iloc[idx_train]
Y_val   = Y_binary.iloc[idx_val]
Y_test  = Y_binary.iloc[idx_test]

print(f"  Feature matrix: {X_all.shape}  |  train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")


# ============================================================================
# STEP 2: Pre-calculate Baseline Classifiers performance on Validation
# ============================================================================
section("STEP 2 - Re-evaluating Baseline Models on Validation")

def get_proba(models, X, target, mtype):
    return models[mtype][target].predict_proba(X)[:, 1]

# Recompute baseline F1 on Validation so we can compare "best model" choices
base_val_results = {"logreg": {}, "lgbm": {}}
for mtype in clf_types:
    for t in OUTPUT_TARGETS:
        y_true = Y_val[t].values
        y_prob = get_proba(base_clf_models, X_val, t, mtype)
        thr = base_thresholds[mtype][t]
        y_pred = (y_prob >= thr).astype(int)
        base_val_results[mtype][t] = f1_score(y_true, y_pred, zero_division=0)

# ============================================================================
# STEP 3: Random Oversampling & Re-training
# ============================================================================
section(f"STEP 3 - Random Oversampling (Target count = {MIN_POS_OVERSAMPLE})")

os_clf_models = {"logreg": {}, "lgbm": {}}
os_thresholds = {"logreg": {}, "lgbm": {}}
os_val_results = {"logreg": {}, "lgbm": {}}

for t in OUTPUT_TARGETS:
    y_tr = Y_train[t].values
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_pos = len(pos_idx)

    # 1. Random Oversampling Logic
    if n_pos < MIN_POS_OVERSAMPLE and n_pos > 0:
        # Calculate how many extra random samples we need
        n_extra = MIN_POS_OVERSAMPLE - n_pos
        np.random.seed(RANDOM_STATE)
        extra_pos_idx = np.random.choice(pos_idx, size=n_extra, replace=True)

        idx_resampled = np.concatenate([neg_idx, pos_idx, extra_pos_idx])
        X_tr_os = X_train[idx_resampled]
        y_tr_os = y_tr[idx_resampled]

        print(f"  [Oversampled] {t:<20}: {n_pos:>4} -> {MIN_POS_OVERSAMPLE:>4} positives")
    else:
        # No oversampling needed
        X_tr_os = X_train
        y_tr_os = y_tr
        print(f"  [Kept Custom] {t:<20}: {n_pos:>4} positives")

    # 2. Train Models
    # Logistic Regression
    new_n_pos = y_tr_os.sum()
    new_n_neg = len(y_tr_os) - new_n_pos
    w = {0: 1.0, 1: max(1.0, new_n_neg / (new_n_pos + 1e-6))}

    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight=w, random_state=RANDOM_STATE)
    lr.fit(X_tr_os, y_tr_os)
    os_clf_models["logreg"][t] = lr

    # LightGBM
    if "lgbm" in clf_types:
        scale = max(1.0, new_n_neg / (new_n_pos + 1e-6))
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, scale_pos_weight=scale,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=1
        )
        lgb_clf.fit(X_tr_os, y_tr_os)
        os_clf_models["lgbm"][t] = lgb_clf

    # 3. Tune Thresholds on UN-oversampled Validation Set
    y_val_true = Y_val[t].values
    for mtype in clf_types:
        y_prob = get_proba(os_clf_models, X_val, t, mtype)
        best_f1, best_thr = -1, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_val_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        os_thresholds[mtype][t] = best_thr
        os_val_results[mtype][t] = best_f1


# ============================================================================
# STEP 4: Test Set Final Evaluation (Baseline vs Oversampled)
# ============================================================================
section("STEP 4 - Test Set Final Evaluation")

test_rows = []

for t in OUTPUT_TARGETS:
    y_test_true = Y_test[t].values

    # Determine winning model type for Baseline
    base_scores = {m: base_val_results[m][t] for m in clf_types}
    base_mtype = max(base_scores, key=base_scores.get)

    # Determine winning model type for Oversampled
    os_scores = {m: os_val_results[m][t] for m in clf_types}
    os_mtype = max(os_scores, key=os_scores.get)

    # ----- Evaluate Baseline -----
    p_base = get_proba(base_clf_models, X_test, t, base_mtype)
    thr_base = base_thresholds[base_mtype][t]
    y_base = (p_base >= thr_base).astype(int)

    base_prec = precision_score(y_test_true, y_base, zero_division=0)
    base_rec = recall_score(y_test_true, y_base, zero_division=0)
    base_acc = accuracy_score(y_test_true, y_base)
    base_f1 = f1_score(y_test_true, y_base, zero_division=0)

    # ----- Evaluate Oversampled -----
    p_os = get_proba(os_clf_models, X_test, t, os_mtype)
    thr_os = os_thresholds[os_mtype][t]
    y_os = (p_os >= thr_os).astype(int)

    os_prec = precision_score(y_test_true, y_os, zero_division=0)
    os_rec = recall_score(y_test_true, y_os, zero_division=0)
    os_acc = accuracy_score(y_test_true, y_os)
    os_f1 = f1_score(y_test_true, y_os, zero_division=0)

    # Add to results
    test_rows.append({
        "target": t,
        "n_pos_test": y_test_true.sum(),
        "Base_Model": base_mtype,
        "Base_Prec": round(base_prec, 4),
        "Base_Rec": round(base_rec, 4),
        "Base_Acc": round(base_acc, 4),
        "Base_F1": round(base_f1, 4),
        "OS_Model": os_mtype,
        "OS_Prec": round(os_prec, 4),
        "OS_Rec": round(os_rec, 4),
        "OS_Acc": round(os_acc, 4),
        "OS_F1": round(os_f1, 4),
        "Recall_Delta": round(os_rec - base_rec, 4),
        "F1_Delta": round(os_f1 - base_f1, 4)
    })

df_test = pd.DataFrame(test_rows).set_index("target")
print("\n" + df_test[["Base_F1", "OS_F1", "F1_Delta", "Base_Rec", "OS_Rec", "Recall_Delta"]].to_string())

# Dump CSV
csv_out = PLOT_DIR / "oversampling_metrics.csv"
df_test.to_csv(csv_out)
print(f"\n  [saved] {csv_out}")


# ============================================================================
# STEP 5: Markdown Report Write
# ============================================================================
section("STEP 5 - Build Markdown Report")

md = f"""# Optmisation 3 — Random Oversampling for Minority Classes

**Script**: `code/model_oversample.py`  
**Date**: 2026-03-14  

## Rationale
Some work steps are incredibly rare (e.g., `allTiresService`, `bodymeasurement`). The original Logistic Regression & LightGBM classifiers struggled to learn the text boundaries for these classes.

In this experiment, we applied **Random Oversampling** to the training matrix (duplicating positive rows until every class had at least `{MIN_POS_OVERSAMPLE}` positive examples), retrained both classifiers, and tuned new optimal F1 thresholds. We then compare the best oversampled classifier per-target against the best baseline classifier.

---

## Test Set Results 

*(Comparing the best Baseline classifier vs the best Oversampled classifier per target)*

| Target | Test Pos Count | Baseline Recall | OS Recall | Recall Δ | Baseline Precision | OS Precision | Prec Δ | Baseline F1 | OS F1 | F1 Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for t, row in df_test.iterrows():
    r_delta_fmt = f"**{row['Recall_Delta']:+.4f}**" if row["Recall_Delta"] > 0 else f"{row['Recall_Delta']:+.4f}"
    p_delta_fmt = f"**{row['OS_Prec'] - row['Base_Prec']:+.4f}**" if (row['OS_Prec'] - row['Base_Prec']) > 0 else f"{row['OS_Prec'] - row['Base_Prec']:+.4f}"
    f1_delta_fmt = f"**{row['F1_Delta']:+.4f}**" if row["F1_Delta"] > 0 else f"{row['F1_Delta']:+.4f}"
    
    md += (f"| {t} | {row['n_pos_test']} | {row['Base_Rec']:.4f} | {row['OS_Rec']:.4f} | {r_delta_fmt} | "
           f"{row['Base_Prec']:.4f} | {row['OS_Prec']:.4f} | {p_delta_fmt} | "
           f"{row['Base_F1']:.4f} | {row['OS_F1']:.4f} | {f1_delta_fmt} |\n")


b_mac_prec = df_test["Base_Prec"].mean()
b_mac_rec = df_test["Base_Rec"].mean()
b_mac_f1 = df_test["Base_F1"].mean()
os_mac_prec = df_test["OS_Prec"].mean()
os_mac_rec = df_test["OS_Rec"].mean()
os_mac_f1 = df_test["OS_F1"].mean()

md += f"""
### Macro-Average Test Set Performance

| Metric | Baseline | Oversampled | Delta |
|---|:---:|:---:|:---:|
| **Macro Precision** | {b_mac_prec:.4f} | {os_mac_prec:.4f} | {os_mac_prec - b_mac_prec:+.4f} |
| **Macro Recall** | {b_mac_rec:.4f} | {os_mac_rec:.4f} | {os_mac_rec - b_mac_rec:+.4f} |
| **Macro F1** | {b_mac_f1:.4f} | {os_mac_f1:.4f} | {os_mac_f1 - b_mac_f1:+.4f} |

---

## Conclusion
"""

if os_mac_f1 > b_mac_f1:
    md += f"Random oversampling successfully improved the overall model performance! The macro F1 improved by {os_mac_f1 - b_mac_f1:+.4f}.\n"
else:
    md += f"Random oversampling marginally degraded or did not significantly change the overall macro F1 ({os_mac_f1 - b_mac_f1:+.4f}). "
    md += "While creating synthetic duplicates helps tree models learn split rules for rare variants, the effect on TF-IDF matrices is often a slight increase in Recall at the expensive cost of Precision, leading to lower net F1s."

out_md = MD_DIR / "oversampling_results.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write(md)
print(f"  [saved] {out_md}\nDONE.")
