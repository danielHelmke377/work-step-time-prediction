"""
model_fully_mixed.py
====================
Fully mixed pipeline — best-per-target classifier AND best-per-target regressor.

Stage 1 — Classifiers (NO oversampling, class weights only):
  Both LogReg and LightGBM are trained per target.
  Thresholds are F1-tuned on the validation set.
  The model with higher val F1 is used for test-set predictions.

Stage 2 — Regressors (same as model_best_per_target_reg.py):
  Winner from markdowns/regressor_selection.md per target.
  hailrepair: plain Ridge vs Ridge+winsorise, chosen by val MAE.
  Fallback = mean of positives if n_pos < 5.

Outputs:
  combined_best/markdowns/fully_mixed_results.md
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
MD_DIR    = ROOT / "combined_best" / "markdowns"
MD_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
RIDGE_ALPHA   = 10.0
WINSORISE_PCT = 95

BEST_REG = {
    "calibration":         "lgbm",
    "wheelmeasurement":    "lgbm",
    "bodymeasurement":     "lgbm",
    "dismounting":         "lgbm",
    "bodyrepair":          "ridge",
    "assembly":            "ridge",
    "plasticrepair":       "lgbm",
    "cleaning":            "lgbm",
    "paintingPreparation": "ridge",
    "paintingSpraying":    "lgbm",
    "paintingFinish":      "ridge",
    "hailrepair":          "ridge_auto",
    "glas":                "lgbm",
    "allTiresService":     "lgbm",
}

OUTPUT_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement", "dismounting",
    "bodyrepair", "assembly", "plasticrepair", "allTiresService",
    "cleaning", "paintingPreparation", "paintingSpraying", "paintingFinish",
    "hailrepair", "glas",
]

BAR = "=" * 70
def section(t): print(f"\n{BAR}\n  {t}\n{BAR}")

# ============================================================================
# STEP 1 — Feature engineering
# ============================================================================
section("STEP 1 — Feature engineering")

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
        feats[f"make_{m.replace('-', '_')}"] = int(make_clean == m)
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
    records_target.append(
        {t: float(r["output"].get(t, 0) or 0) for t in OUTPUT_TARGETS})

make_counts = pd.Series([f["make_freq_raw"] for f in records_feats]).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1)
    del f["make_freq_raw"]

df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)
Y_binary   = (df_targets > 0).astype(int)
Y_reg      = df_targets.copy()

n_active        = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])
indices = np.arange(len(df_num))
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE,
    stratify=complexity_band[idx_trainval])

tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=2,
                              sublinear_tf=True, analyzer="word",
                              token_pattern=r"(?u)\b[\w\+]+\b")
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), max_features=1000, min_df=3,
                              sublinear_tf=True, analyzer="char_wb")
X_word = tfidf_word.fit_transform(records_text)
X_char = tfidf_char.fit_transform(records_text)
X_all  = sparse.hstack([X_word, X_char, sparse.csr_matrix(df_num.values)], format="csr")

X_train = X_all[idx_train]
X_val   = X_all[idx_val]
X_test  = X_all[idx_test]
Y_bin_train = Y_binary.iloc[idx_train]
Y_bin_val   = Y_binary.iloc[idx_val]
Y_bin_test  = Y_binary.iloc[idx_test]
Y_reg_train = Y_reg.iloc[idx_train]
Y_reg_val   = Y_reg.iloc[idx_val]
Y_reg_test  = Y_reg.iloc[idx_test]

print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

# ============================================================================
# STEP 2 — Best-per-target classifiers
# ============================================================================
section("STEP 2 — Best-per-target classifiers (no oversampling)")

clf_best   = {}  # target -> fitted model
clf_thr    = {}  # target -> threshold
clf_choice = {}  # target -> 'logreg' | 'lgbm'

for t in OUTPUT_TARGETS:
    y_tr = Y_bin_train[t].values
    y_va = Y_bin_val[t].values
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    w = max(1.0, n_neg / (n_pos + 1e-9))

    # Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight={0: 1.0, 1: w},
                            random_state=RANDOM_STATE)
    lr.fit(X_train, y_tr)

    # LightGBM classifier
    lc = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             scale_pos_weight=w, random_state=RANDOM_STATE,
                             verbose=-1, n_jobs=1)
    lc.fit(X_train, y_tr)

    # F1-tune thresholds on val set, pick best model
    def best_f1_thr(model, Xv, yv):
        probs = model.predict_proba(Xv)[:, 1]
        best_f1, best_t = -1.0, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            s = f1_score(yv, (probs >= thr).astype(int), zero_division=0)
            if s > best_f1:
                best_f1, best_t = s, thr
        return best_f1, best_t

    f1_lr, thr_lr = best_f1_thr(lr, X_val, y_va)
    f1_lc, thr_lc = best_f1_thr(lc, X_val, y_va)

    if f1_lr >= f1_lc:
        clf_best[t], clf_thr[t], clf_choice[t] = lr, thr_lr, "logreg"
    else:
        clf_best[t], clf_thr[t], clf_choice[t] = lc, thr_lc, "lgbm"

    print(f"  {t:<22}  logreg F1={f1_lr:.4f}  lgbm F1={f1_lc:.4f}  → {clf_choice[t]}")

# ============================================================================
# STEP 3 — Best-per-target regressors
# ============================================================================
section("STEP 3 — Best-per-target regressors")


def make_lgbm_reg(n_pos):
    return lgb.LGBMRegressor(
        objective="regression", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=max(5, n_pos // 10),
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    )


def make_ridge():
    return Ridge(alpha=RIDGE_ALPHA)


def val_mae_score(model, Xv_pos, yv_pos):
    if len(yv_pos) == 0:
        return float("inf")
    return mean_absolute_error(yv_pos, np.maximum(0, model.predict(Xv_pos)))


reg_models = {}
reg_choice = {}

for t in OUTPUT_TARGETS:
    mask_tr = Y_bin_train[t].values == 1
    n_pos   = mask_tr.sum()
    rtype   = BEST_REG.get(t, "lgbm")

    if n_pos < 5:
        y_avail = Y_reg_train.loc[Y_bin_train[t] == 1, t]
        reg_models[t] = ("fallback", float(y_avail.mean()))
        reg_choice[t] = "fallback"
        print(f"  [fallback ] {t:<22}  n_pos={n_pos}")
        continue

    X_pos = X_train[mask_tr]
    y_pos = Y_reg_train.loc[Y_bin_train[t] == 1, t].values

    if rtype == "lgbm":
        m = make_lgbm_reg(n_pos)
        m.fit(X_pos, y_pos)
        reg_models[t] = m
        reg_choice[t] = "lgbm"
        print(f"  [lgbm     ] {t:<22}  n_pos={n_pos}")

    elif rtype == "ridge":
        m = make_ridge()
        m.fit(X_pos, y_pos)
        reg_models[t] = m
        reg_choice[t] = "ridge"
        print(f"  [ridge    ] {t:<22}  n_pos={n_pos}")

    elif rtype == "ridge_auto":
        cap    = np.percentile(y_pos, WINSORISE_PCT)
        y_wins = np.clip(y_pos, 0, cap)
        m_plain = make_ridge()
        m_wins  = make_ridge()
        m_plain.fit(X_pos, y_pos)
        m_wins.fit(X_pos, y_wins)

        mask_va   = Y_bin_val[t].values == 1
        X_va_pos  = X_val[mask_va]
        y_va_pos  = Y_reg_val.loc[Y_bin_val[t] == 1, t].values
        mae_plain = val_mae_score(m_plain, X_va_pos, y_va_pos)
        mae_wins  = val_mae_score(m_wins,  X_va_pos, y_va_pos)

        if mae_wins < mae_plain:
            reg_models[t] = m_wins
            reg_choice[t] = f"ridge+wins(cap={cap:.0f})"
            print(f"  [ridge+win] {t:<22}  plain={mae_plain:.2f} wins={mae_wins:.2f} → winsorise")
        else:
            reg_models[t] = m_plain
            reg_choice[t] = "ridge"
            print(f"  [ridge    ] {t:<22}  plain={mae_plain:.2f} wins={mae_wins:.2f} → plain")

# ============================================================================
# STEP 4 — Evaluate
# ============================================================================
section("STEP 4 — Evaluate: fully mixed CLF + fully mixed REG")

freq_weights = {t: float(Y_bin_test[t].mean()) for t in OUTPUT_TARGETS}
total_fw     = sum(freq_weights.values())

REF = {
    "baseline_fw_f1": 0.9387, "baseline_fw_mae": 2.78,
    "optimal_fw_f1":  0.9347, "optimal_fw_mae":  1.88,
    "bptr_fw_f1":     0.9347, "bptr_fw_mae":     0.96,
}

rows = []
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_true_reg = Y_reg_test[t].values

    probs  = clf_best[t].predict_proba(X_test)[:, 1]
    y_bin  = (probs >= clf_thr[t]).astype(int)

    m = reg_models[t]
    y_dur     = np.full(X_test.shape[0], m[1]) if isinstance(m, tuple) else np.maximum(0.0, m.predict(X_test))
    y_reg_hat = np.where(y_bin == 1, y_dur, 0.0)

    rows.append({
        "target":   t,
        "clf":      clf_choice[t],
        "reg":      reg_choice[t],
        "freq_w":   round(freq_weights[t], 3),
        "n_pos":    int(y_true_bin.sum()),
        "Accuracy": round(accuracy_score(y_true_bin, y_bin), 4),
        "F1":       round(f1_score(y_true_bin, y_bin, zero_division=0), 4),
        "Recall":   round(recall_score(y_true_bin, y_bin, zero_division=0), 4),
        "Prec":     round(precision_score(y_true_bin, y_bin, zero_division=0), 4),
        "MAE":      round(mean_absolute_error(y_true_reg, y_reg_hat), 2),
    })

df = pd.DataFrame(rows).set_index("target")

macro_f1  = round(df["F1"].mean(), 4)
macro_rec = round(df["Recall"].mean(), 4)
macro_acc = round(df["Accuracy"].mean(), 4)
macro_mae = round(df["MAE"].mean(), 2)
fw_f1  = round(sum(freq_weights[t] * df.loc[t, "F1"]      for t in OUTPUT_TARGETS) / total_fw, 4)
fw_rec = round(sum(freq_weights[t] * df.loc[t, "Recall"]   for t in OUTPUT_TARGETS) / total_fw, 4)
fw_acc = round(sum(freq_weights[t] * df.loc[t, "Accuracy"] for t in OUTPUT_TARGETS) / total_fw, 4)
fw_mae = round(sum(freq_weights[t] * df.loc[t, "MAE"]      for t in OUTPUT_TARGETS) / total_fw, 2)

print(f"\n  {'Target':<22} {'clf':>6} {'reg':>18} {'freq_w':>6} {'Acc':>7} {'F1':>7} {'Rec':>7} {'MAE':>7}")
print("  " + "-" * 83)
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    print(f"  {t:<22} {r['clf']:>6} {r['reg']:>18} {r['freq_w']:>6.3f} "
          f"{r['Accuracy']:>7.4f} {r['F1']:>7.4f} {r['Recall']:>7.4f} {r['MAE']:>7.2f}")

print(f"\n  {'Metric':<28} {'Baseline':>10} {'TrueOpt':>10} {'BestReg':>10} {'FullMix':>10}")
print(f"  {'-'*60}")
print(f"  {'Macro F1':<28} {'0.8372':>10} {'0.8372':>10} {'0.8372':>10} {macro_f1:>10.4f}")
print(f"  {'Macro Recall':<28} {'0.779':>10} {'0.8594':>10} {'0.8594':>10} {macro_rec:>10.4f}")
print(f"  {'Macro Accuracy':<28} {'0.9380':>10} {'0.9380':>10} {'0.9380':>10} {macro_acc:>10.4f}")
print(f"  {'Freq-weighted F1':<28} {REF['baseline_fw_f1']:>10.4f} {REF['optimal_fw_f1']:>10.4f} {REF['bptr_fw_f1']:>10.4f} {fw_f1:>10.4f}")
print(f"  {'Freq-weighted Recall':<28} {'0.9460':>10} {'0.9412':>10} {'0.9412':>10} {fw_rec:>10.4f}")
print(f"  {'Freq-weighted Accuracy':<28} {'0.9433':>10} {'0.9433':>10} {'0.9433':>10} {fw_acc:>10.4f}")
print(f"  {'Macro MAE (hrs)':<28} {'3.51':>10} {'2.14':>10} {'0.70':>10} {macro_mae:>10.2f}")
print(f"  {'Freq-weighted MAE (hrs)':<28} {REF['baseline_fw_mae']:>10.2f} {REF['optimal_fw_mae']:>10.2f} {REF['bptr_fw_mae']:>10.2f} {fw_mae:>10.2f}")

# ============================================================================
# STEP 5 — Write markdown
# ============================================================================
section("STEP 5 — Writing markdown report")

tbl_rows = []
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    tbl_rows.append(
        f"| `{t}` | {r['clf']} | {r['reg']} | {r['n_pos']} | {r['freq_w']:.3f} "
        f"| {r['Accuracy']:.4f} | {r['F1']:.4f} | {r['Recall']:.4f} | {r['MAE']:.2f} |"
    )

clf_summary = {}
for t in OUTPUT_TARGETS:
    clf_summary[clf_choice[t]] = clf_summary.get(clf_choice[t], 0) + 1

md = f"""# Fully Mixed Pipeline — Best-Per-Target CLF + Best-Per-Target REG

**Date:** {date.today()}

**Goal:** Use the best classifier and the best regressor independently per target,
both chosen by validation-set comparison. No oversampling, class weights only.

## Classifier Summary

| Model | n targets |
|---|---|
| Logistic Regression | {clf_summary.get('logreg', 0)} |
| LightGBM | {clf_summary.get('lgbm', 0)} |

## Comparison vs Previous Experiments

| Metric | Baseline | True Optimal | Best-per-REG | **Fully Mixed** |
|---|---|---|---|---|
| Macro F1 | 0.8372 | 0.8372 | 0.8372 | **{macro_f1}** |
| Macro Recall | 0.779 | 0.8594 | 0.8594 | **{macro_rec}** |
| Macro Accuracy | 0.9380 | 0.9380 | 0.9380 | **{macro_acc}** |
| Freq-weighted F1 | {REF['baseline_fw_f1']} | {REF['optimal_fw_f1']} | {REF['bptr_fw_f1']} | **{fw_f1}** |
| Freq-weighted Recall | 0.9460 | 0.9412 | 0.9412 | **{fw_rec}** |
| Freq-weighted Accuracy | 0.9433 | 0.9433 | 0.9433 | **{fw_acc}** |
| Macro MAE | 3.51 hrs | 2.14 hrs | 0.70 hrs | **{macro_mae} hrs** |
| **Freq-weighted MAE** | {REF['baseline_fw_mae']} hrs | {REF['optimal_fw_mae']} hrs | {REF['bptr_fw_mae']} hrs | **{fw_mae} hrs** |

## Per-target Results

| Target | clf | reg | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----|-----|-------|--------|----------|----|--------|-----|
{chr(10).join(tbl_rows)}
| **MACRO** | | | | | **{macro_acc}** | **{macro_f1}** | **{macro_rec}** | **{macro_mae}** |
| **FREQ-WT** | | | | | **{fw_acc}** | **{fw_f1}** | **{fw_rec}** | **{fw_mae}** |
"""

md_path = MD_DIR / "fully_mixed_results.md"
md_path.write_text(md, encoding="utf-8")
print(f"\n  [saved] {md_path}")
print("  Done.")
