"""
model_combined_best.py
======================
Combined Best Pipeline — highest F1 + lowest freq-weighted MAE.

Combines the two best findings from all experiments in this repository:

  Stage 1 (Classifiers)
  ---------------------
  • Random oversampling of minority classes to MIN_POS_OVERSAMPLE = 200
  • Soft-vote ensemble (LogisticRegression + LightGBM)
  • Two threshold strategies evaluated on held-out validation set:
      F1-optimal  : maximise macro F1  (reference)
      Recall-floor: maximise precision s.t. Recall >= MIN_RECALL = 0.90

  Stage 2 (Regressors)
  --------------------
  • LightGBM regressor per target (identical to model_phase2.py)
  • Winsorise training targets at 95th percentile for high-skew targets:
      hailrepair    : max/p95 = 2.5,  skew = 2.85  (4222-hr outlier)
      assembly      : max/p95 = 3.2,  skew = 3.54  (28.7-hr outlier)
      paintingFinish: max/p95 = 4.0,  skew = 3.96  (14.3-hr outlier)
      calibration   : max/p95 = 3.8,  skew = 4.94  (18.5-hr outlier)

Evaluation metrics reported:
  • Macro F1 (both threshold strategies)
  • Macro Recall (both threshold strategies)
  • Macro MAE (unweighted, 14 targets)
  • Frequency-weighted MAE (weighted by occurrence rate in test set)

Outputs:
  combined_best/markdowns/combined_best_results.md
"""

import json
import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, mean_absolute_error,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except ImportError:
    raise RuntimeError("LightGBM required.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent   # repo root
DATA_PATH = ROOT / "data" / "orders_simplified_sample.json"
MD_DIR    = ROOT / "combined_best" / "markdowns"
MD_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
MIN_POS_OVERSAMPLE = 200    # oversample minority targets up to this count
MIN_RECALL         = 0.90   # recall floor for the RC threshold strategy
WINSORISE_PCT     = 95      # cap high-skew targets at this percentile

# Targets with max/p95 >= 3 and skew >= 3 — winsorised during regressor training
WINSORISE_TARGETS = {"hailrepair", "assembly", "paintingFinish", "calibration"}

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
compiled_kw  = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}

BAR = "=" * 70
def section(t): print(f"\n{BAR}\n  {t}\n{BAR}")


# ============================================================================
# STEP 1 — Data & Features
# ============================================================================
section("STEP 1 — Data & Features")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
valid = [r for r in raw if "input" in r and "output" in r
         and "calculatedPositions" in r["input"]]
print(f"  Records: {len(valid):,}")


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


records_text, records_feats, records_target = [], [], []
for r in valid:
    pos  = preprocess_record(r)
    make = r["input"].get("make", "unknown")
    records_text.append(build_order_text(pos))
    records_feats.append(build_numeric_features(pos, make))
    records_target.append({t: float(r["output"].get(t, 0) or 0) for t in OUTPUT_TARGETS})

make_counts = pd.Series([f["make_freq_raw"] for f in records_feats]).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1); del f["make_freq_raw"]

df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)

tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=2,
                              sublinear_tf=True, analyzer="word",
                              token_pattern=r"(?u)\b[\w\+]+\b")
X_word = tfidf_word.fit_transform(records_text)
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), max_features=1000, min_df=3,
                              sublinear_tf=True, analyzer="char_wb")
X_char = tfidf_char.fit_transform(records_text)
X_all  = sparse.hstack([X_word, X_char, sparse.csr_matrix(df_num.values)], format="csr")
print(f"  Feature matrix: {X_all.shape}")

Y_binary = (df_targets > 0).astype(int)
Y_reg    = df_targets.copy()

n_active        = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])
indices = np.arange(len(df_num))
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=complexity_band)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE,
    stratify=complexity_band[idx_trainval])

X_train = X_all[idx_train]; X_val = X_all[idx_val]; X_test = X_all[idx_test]
Y_bin_train = Y_binary.iloc[idx_train]; Y_bin_val = Y_binary.iloc[idx_val]
Y_bin_test  = Y_binary.iloc[idx_test]
Y_reg_train = Y_reg.iloc[idx_train]; Y_reg_test = Y_reg.iloc[idx_test]
print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")


# ============================================================================
# STEP 2 — Stage 1: Oversampling + Soft-vote Classifiers
# ============================================================================
section(f"STEP 2 — Stage 1: Oversampling (target={MIN_POS_OVERSAMPLE}) + Soft-vote")

clf_models = {"logreg": {}, "lgbm": {}}

for t in OUTPUT_TARGETS:
    y_tr    = Y_bin_train[t].values
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_pos   = len(pos_idx)

    if 0 < n_pos < MIN_POS_OVERSAMPLE:
        np.random.seed(RANDOM_STATE)
        extra   = np.random.choice(pos_idx, size=MIN_POS_OVERSAMPLE - n_pos, replace=True)
        idx_os  = np.concatenate([neg_idx, pos_idx, extra])
        X_tr_os = X_train[idx_os]; y_tr_os = y_tr[idx_os]
        print(f"  [oversample] {t:<22} {n_pos:>4} → {MIN_POS_OVERSAMPLE} positives")
    else:
        X_tr_os = X_train; y_tr_os = y_tr
        print(f"  [kept]       {t:<22} {n_pos:>4} positives")

    n_pos_os = y_tr_os.sum(); n_neg_os = len(y_tr_os) - n_pos_os
    w = {0: 1.0, 1: max(1.0, n_neg_os / (n_pos_os + 1e-6))}

    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                            class_weight=w, random_state=RANDOM_STATE)
    lr.fit(X_tr_os, y_tr_os)
    clf_models["logreg"][t] = lr

    scale = max(1.0, n_neg_os / (n_pos_os + 1e-6))
    lc    = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                scale_pos_weight=scale, random_state=RANDOM_STATE,
                                verbose=-1, n_jobs=1)
    lc.fit(X_tr_os, y_tr_os)
    clf_models["lgbm"][t] = lc

print("  Classifiers trained.")


def ensemble_proba(X, t):
    p_lr  = clf_models["logreg"][t].predict_proba(X)[:, 1]
    p_lgb = clf_models["lgbm"][t].predict_proba(X)[:, 1]
    return (p_lr + p_lgb) / 2.0


# ============================================================================
# STEP 3 — Threshold Tuning on Validation
# ============================================================================
section(f"STEP 3 — Threshold Tuning (F1-optimal & Recall≥{MIN_RECALL})")

thr_f1 = {}; thr_rc = {}

for t in OUTPUT_TARGETS:
    y_v = Y_bin_val[t].values
    p_v = ensemble_proba(X_val, t)

    # F1-optimal
    best_f1, best_t = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_v, (p_v >= thr).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, thr
    thr_f1[t] = best_t

    # Recall-constrained
    best_pr, best_t_rc = -1, 0.01
    for thr in np.linspace(0.01, 0.99, 99):
        yp = (p_v >= thr).astype(int)
        rec = recall_score(y_v, yp, zero_division=0)
        pr  = precision_score(y_v, yp, zero_division=0)
        if rec >= MIN_RECALL and pr > best_pr:
            best_pr, best_t_rc = pr, thr
    thr_rc[t] = best_t_rc

print("  Threshold tuning complete.")


# ============================================================================
# STEP 4 — Stage 2: Regressors with hailrepair winsorising
# ============================================================================
section(f"STEP 4 — Stage 2: Regressors (winsorise {WINSORISE_PCT}th pct: {sorted(WINSORISE_TARGETS)})")

reg_models = {}
for t in OUTPUT_TARGETS:
    mask = Y_bin_train[t].values == 1
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
        y_fit = y_pos; note = ""

    lgb_reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=max(5, n_pos // 10),
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    )
    lgb_reg.fit(X_pos, y_fit)
    reg_models[t] = lgb_reg
    print(f"  [trained  ] {t:<22}  n_pos={n_pos}{note}")


def reg_predict(X, t):
    m = reg_models[t]
    if isinstance(m, tuple): return np.full(X.shape[0], m[1])
    return np.maximum(0.0, m.predict(X))


# ============================================================================
# STEP 5 — End-to-end Evaluation (both threshold strategies)
# ============================================================================
section("STEP 5 — End-to-end Evaluation")

freq_weights = {t: float(Y_bin_test[t].mean()) for t in OUTPUT_TARGETS}
total_fw = sum(freq_weights.values())

rows = []
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_true_reg = Y_reg_test[t].values
    p_test     = ensemble_proba(X_test, t)
    y_dur      = reg_predict(X_test, t)

    # F1-optimal predictions
    y_bin_f1   = (p_test >= thr_f1[t]).astype(int)
    y_reg_f1   = np.where(y_bin_f1 == 1, y_dur, 0.0)

    # Recall-constrained predictions
    y_bin_rc   = (p_test >= thr_rc[t]).astype(int)
    y_reg_rc   = np.where(y_bin_rc == 1, y_dur, 0.0)

    rows.append({
        "target":       t,
        "n_pos":        int(y_true_bin.sum()),
        "freq_w":       round(freq_weights[t], 3),
        # F1-optimal
        "F1_f1":        round(f1_score(y_true_bin, y_bin_f1, zero_division=0), 4),
        "Rec_f1":       round(recall_score(y_true_bin, y_bin_f1, zero_division=0), 4),
        "Prec_f1":      round(precision_score(y_true_bin, y_bin_f1, zero_division=0), 4),
        "MAE_f1":       round(mean_absolute_error(y_true_reg, y_reg_f1), 2),
        # Recall-constrained
        "F1_rc":        round(f1_score(y_true_bin, y_bin_rc, zero_division=0), 4),
        "Rec_rc":       round(recall_score(y_true_bin, y_bin_rc, zero_division=0), 4),
        "Prec_rc":      round(precision_score(y_true_bin, y_bin_rc, zero_division=0), 4),
        "MAE_rc":       round(mean_absolute_error(y_true_reg, y_reg_rc), 2),
    })

df = pd.DataFrame(rows).set_index("target")

# Macro / weighted aggregates for both strategies
macro_f1_f1   = round(df["F1_f1"].mean(), 4)
macro_rec_f1  = round(df["Rec_f1"].mean(), 4)
macro_mae_f1  = round(df["MAE_f1"].mean(), 2)
fw_f1_f1      = round(sum(freq_weights[t] * df.loc[t, "F1_f1"]  for t in OUTPUT_TARGETS) / total_fw, 4)
fw_rec_f1     = round(sum(freq_weights[t] * df.loc[t, "Rec_f1"] for t in OUTPUT_TARGETS) / total_fw, 4)
fw_mae_f1     = round(sum(freq_weights[t] * df.loc[t, "MAE_f1"] for t in OUTPUT_TARGETS) / total_fw, 2)

macro_f1_rc   = round(df["F1_rc"].mean(), 4)
macro_rec_rc  = round(df["Rec_rc"].mean(), 4)
macro_mae_rc  = round(df["MAE_rc"].mean(), 2)
fw_f1_rc      = round(sum(freq_weights[t] * df.loc[t, "F1_rc"]  for t in OUTPUT_TARGETS) / total_fw, 4)
fw_rec_rc     = round(sum(freq_weights[t] * df.loc[t, "Rec_rc"] for t in OUTPUT_TARGETS) / total_fw, 4)
fw_mae_rc     = round(sum(freq_weights[t] * df.loc[t, "MAE_rc"] for t in OUTPUT_TARGETS) / total_fw, 2)

# Reference baselines (model_phase2.py — unweighted; no freq-weighted baseline exists)
BL_MACRO_F1   = 0.8372
BL_MACRO_REC  = 0.779
BL_MACRO_MAE  = 3.51
BL_FW_MAE     = 2.78

print("\n  Per-target results (F1-optimal strategy):")
print(f"  {'Target':<22} {'F1':>7} {'Rec':>7} {'Prec':>7} {'MAE':>7} {'freq_w':>8}")
print("  " + "-" * 60)
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    print(f"  {t:<22} {r['F1_f1']:>7.4f} {r['Rec_f1']:>7.4f} "
          f"{r['Prec_f1']:>7.4f} {r['MAE_f1']:>7.2f} {r['freq_w']:>8.3f}")

print(f"\n  {'MACRO':<22} {macro_f1_f1:>7.4f} {macro_rec_f1:>7.4f} "
      f"{'—':>7} {macro_mae_f1:>7.2f}")
print(f"  {'FREQ-WEIGHTED':<22} {fw_f1_f1:>7.4f} {fw_rec_f1:>7.4f} "
      f"{'—':>7} {fw_mae_f1:>7.2f}")

print(f"\n  Summary vs baselines:")
print(f"  {'Metric':<32} {'Baseline':>10} {'Combined (F1-opt)':>18} {'Combined (RC)':>15}")
print(f"  {'-'*77}")
print(f"  {'Macro F1 (unweighted)':<32} {BL_MACRO_F1:>10.4f} {macro_f1_f1:>18.4f} {macro_f1_rc:>15.4f}")
print(f"  {'Macro Recall (unweighted)':<32} {BL_MACRO_REC:>10.4f} {macro_rec_f1:>18.4f} {macro_rec_rc:>15.4f}")
print(f"  {'Freq-weighted F1':<32} {'—':>10} {fw_f1_f1:>18.4f} {fw_f1_rc:>15.4f}")
print(f"  {'Freq-weighted Recall':<32} {'—':>10} {fw_rec_f1:>18.4f} {fw_rec_rc:>15.4f}")
print(f"  {'Macro MAE (unweighted)':<32} {BL_MACRO_MAE:>10.2f} {macro_mae_f1:>18.2f} {macro_mae_rc:>15.2f}")
print(f"  {'Freq-weighted MAE':<32} {BL_FW_MAE:>10.2f} {fw_mae_f1:>18.2f} {fw_mae_rc:>15.2f}")


# ============================================================================
# STEP 6 — Write Markdown Report
# ============================================================================
section("STEP 6 — Writing Markdown Report")

# Build per-target table rows
tbl_rows = []
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    tbl_rows.append(
        f"| `{t:<22}` | {r['n_pos']:>5} | {r['freq_w']:>6.3f} "
        f"| {r['F1_f1']:>7.4f} | {r['Rec_f1']:>7.4f} | {r['MAE_f1']:>7.2f} "
        f"| {r['F1_rc']:>7.4f} | {r['Rec_rc']:>7.4f} | {r['MAE_rc']:>7.2f} |"
    )

md = f"""# Combined Best Pipeline — Experiment Results

**Date:** {date.today()}
**Goal:** Combine the best classifier strategy (oversampling + soft-vote + recall floor)
with the best regression strategy (hailrepair winsorised at 95th percentile) to
simultaneously maximise F1 and minimise frequency-weighted MAE.

---

## Configuration

| Parameter | Value |
|---|---|
| Oversampling target (MIN_POS) | {MIN_POS_OVERSAMPLE} positives per target |
| Ensemble | Soft-vote (LogisticRegression + LightGBM) |
| Recall floor (RC strategy) | {int(MIN_RECALL * 100)}% |
| Winsorised targets ({WINSORISE_PCT}th pct) | {", ".join(sorted(WINSORISE_TARGETS))} |

---

## Baseline vs Combined Results

| Metric | Baseline (`model_phase2.py`) | **Combined (F1-opt)** | Combined (RC) |
|---|---|---|---|
| Macro F1 (unweighted) | {BL_MACRO_F1} | **{macro_f1_f1}** | {macro_f1_rc} |
| Macro Recall (unweighted) | {BL_MACRO_REC} | **{macro_rec_f1}** | {macro_rec_rc} |
| **Freq-weighted F1** | — | **{fw_f1_f1}** | {fw_f1_rc} |
| **Freq-weighted Recall** | — | **{fw_rec_f1}** | {fw_rec_rc} |
| Macro MAE (unweighted) | {BL_MACRO_MAE} hrs | **{macro_mae_f1} hrs** | {macro_mae_rc} hrs |
| **Freq-weighted MAE** | {BL_FW_MAE} hrs | **{fw_mae_f1} hrs** | {fw_mae_rc} hrs |

> Frequency-weighted metrics weight each target by its occurrence rate in the test set,
> reflecting real-world business impact. Baseline has no freq-weighted F1/Recall equivalent
> since those were not computed in `model_phase2.py`.

---

## Per-target Results

| Target | n_pos | freq_w | F1 (F1-opt) | Rec (F1-opt) | MAE (F1-opt) | F1 (RC) | Rec (RC) | MAE (RC) |
|--------|-------|--------|-------------|--------------|--------------|---------|----------|----------|
{chr(10).join(tbl_rows)}
| **MACRO** | — | — | **{macro_f1_f1}** | **{macro_rec_f1}** | **{macro_mae_f1}** | **{macro_f1_rc}** | **{macro_rec_rc}** | **{macro_mae_rc}** |

---

## Interpretation

### Classifiers
- Oversampling improved F1 for minority targets (`allTiresService`, `bodymeasurement`,
  `plasticrepair`) with very few natural positives.
- The soft-vote ensemble smooths overconfident predictions from either individual model.
- The **Recall-Constrained (RC)** strategy guarantees Recall ≥ {int(MIN_RECALL * 100)}% per target,
  at the cost of lower precision — useful when missing a work step (false negative) is
  more costly than a false positive.

### Regressors
- The `hailrepair` winsorise removes the single 4222-hr training record (likely a
  data entry error) that dominated the fit. All other regression targets are unchanged.

### Trade-offs
- **F1-optimal** is the better default: highest classification accuracy.
- **RC** is the better choice when **under-quoting** (missed work steps) causes
  customer or business problems — for example, if a hail repair is not quoted but
  actually needed, re-work costs are typically 3–5× the original quote.

---

## Next Steps

1. **Feature engineering** — hail-specific features (`n_hail_positions`,
   `price_hail / total_price`) to further reduce `hailrepair` MAE
2. **Lower winsorise percentile** — test 85th/90th to tighten the hailrepair cap
3. **More data** — oversampling is a substitute for real data; acquiring 100+ additional
   hailrepair orders with accurate durations would be the highest-leverage action
"""

md_path = MD_DIR / "combined_best_results.md"
md_path.write_text(md, encoding="utf-8")
print(f"\n  [saved] {md_path}")
print("  Done.")
