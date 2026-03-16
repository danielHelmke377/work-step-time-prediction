"""
model_best_per_target_reg.py
============================
Best-per-target regressor experiment.

Uses the SAVED baseline classifiers from models/two_stage_pipeline.pkl
(unchanged) and trains the best-performing regressor per target as
documented in markdowns/regressor_selection.md:

  LightGBM: calibration, wheelmeasurement, bodymeasurement, dismounting,
             plasticrepair, cleaning, paintingSpraying, glas
  Ridge:     bodyrepair, assembly, paintingPreparation, paintingFinish
  hailrepair:  Ridge, but validated against Ridge+winsorise(95th pct)
               on the val set — only winsorise if MAE improves.
  allTiresService: fallback (mean) — fewer than 5 positive training examples.

Outputs:
  combined_best/markdowns/best_per_target_reg_results.md
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
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, precision_score,
    recall_score,
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
RANDOM_STATE  = 42
RIDGE_ALPHA   = 10.0
WINSORISE_PCT = 95

# Best regressor per target from regressor_selection.md
# 'tie' → LightGBM (primary choice)
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
    "hailrepair":          "ridge_auto",   # ridge vs ridge+winsorise on val set
    "glas":                "lgbm",
    "allTiresService":     "lgbm",
}

BAR = "=" * 70
def section(t): print(f"\n{BAR}\n  {t}\n{BAR}")

# ============================================================================
# STEP 1 — Load saved pipeline
# ============================================================================
section("STEP 1 — Load saved pipeline")

with open(PKL_PATH, "rb") as f:
    pipeline = pickle.load(f)

tfidf_word     = pipeline["tfidf_word"]
tfidf_char     = pipeline["tfidf_char"]
clf_models     = pipeline["clf_models"]
thresholds     = pipeline["thresholds"]
best_clf       = pipeline["best_clf_per_target"]
OUTPUT_TARGETS = pipeline["output_targets"]

print(f"  Targets: {len(OUTPUT_TARGETS)}")

# ============================================================================
# STEP 2 — Rebuild features (no TF-IDF refitting)
# ============================================================================
section("STEP 2 — Rebuild features")

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
COST_CENTERS = ["bodywork","painting","paintmaterial","material","others","hail"]
TOP_MAKES    = ["VOLKSWAGEN","MERCEDES-BENZ","BMW","FORD","SKODA","AUDI","OPEL","TESLA"]
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

X_word = tfidf_word.transform(records_text)
X_char = tfidf_char.transform(records_text)
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


def make_lgbm_reg(n_pos):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=max(5, n_pos // 10),
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    )


def make_ridge():
    return Ridge(alpha=RIDGE_ALPHA)


def val_mae(model, X_val_pos, y_val_pos):
    return mean_absolute_error(y_val_pos, np.maximum(0, model.predict(X_val_pos)))


# ============================================================================
# STEP 3 — Train best-per-target regressors
# ============================================================================
section("STEP 3 — Train best-per-target regressors")

reg_models = {}
reg_choice = {}

for t in OUTPUT_TARGETS:
    mask_tr   = Y_bin_train[t].values == 1
    n_pos     = mask_tr.sum()
    rtype     = BEST_REG.get(t, "lgbm")

    if n_pos < 5:
        y_avail       = Y_reg_train.loc[Y_bin_train[t] == 1, t]
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
        # Fit both plain Ridge and Ridge+winsorise on training set,
        # choose whichever has lower validation MAE.
        cap    = np.percentile(y_pos, WINSORISE_PCT)
        y_wins = np.clip(y_pos, 0, cap)

        m_plain = make_ridge();   m_plain.fit(X_pos, y_pos)
        m_wins  = make_ridge();   m_wins.fit(X_pos, y_wins)

        # Evaluate on val-set positives only
        mask_va     = Y_bin_val[t].values == 1
        X_val_pos   = X_val[mask_va]
        y_val_pos   = Y_reg_val.loc[Y_bin_val[t] == 1, t].values

        mae_plain = val_mae(m_plain, X_val_pos, y_val_pos) if len(y_val_pos) > 0 else float("inf")
        mae_wins  = val_mae(m_wins,  X_val_pos, y_val_pos) if len(y_val_pos) > 0 else float("inf")

        if mae_wins < mae_plain:
            reg_models[t] = m_wins
            reg_choice[t] = f"ridge+winsorise(cap={cap:.0f})"
            print(f"  [ridge+win] {t:<22}  n_pos={n_pos}  val_MAE plain={mae_plain:.2f} winsorise={mae_wins:.2f} → winsorise wins")
        else:
            reg_models[t] = m_plain
            reg_choice[t] = "ridge"
            print(f"  [ridge    ] {t:<22}  n_pos={n_pos}  val_MAE plain={mae_plain:.2f} winsorise={mae_wins:.2f} → plain wins")


# ============================================================================
# STEP 4 — Evaluate
# ============================================================================
section("STEP 4 — Evaluate: saved CLF + best-per-target REG")

freq_weights = {t: float(Y_bin_test[t].mean()) for t in OUTPUT_TARGETS}
total_fw     = sum(freq_weights.values())

# Reference numbers from model_true_optimal.py (all-LightGBM + winsorise)
REF = {"mf1": 0.8372, "mrec": 0.8594, "fw_f1": 0.9347,
       "fw_rec": 0.9412, "mmae": 2.14, "fw_mae": 1.88}

rows = []
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_true_reg = Y_reg_test[t].values

    mtype  = best_clf[t]
    thr    = thresholds[mtype][t]
    p_test = clf_models[mtype][t].predict_proba(X_test)[:, 1]
    y_bin  = (p_test >= thr).astype(int)

    m = reg_models[t]
    if isinstance(m, tuple):
        y_dur = np.full(X_test.shape[0], m[1])
    else:
        y_dur = np.maximum(0.0, m.predict(X_test))
    y_reg_hat = np.where(y_bin == 1, y_dur, 0.0)

    rows.append({
        "target":   t,
        "reg_type": reg_choice[t],
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

fw_f1  = round(sum(freq_weights[t] * df.loc[t, "F1"]       for t in OUTPUT_TARGETS) / total_fw, 4)
fw_rec = round(sum(freq_weights[t] * df.loc[t, "Recall"]    for t in OUTPUT_TARGETS) / total_fw, 4)
fw_acc = round(sum(freq_weights[t] * df.loc[t, "Accuracy"]  for t in OUTPUT_TARGETS) / total_fw, 4)
fw_mae = round(sum(freq_weights[t] * df.loc[t, "MAE"]       for t in OUTPUT_TARGETS) / total_fw, 2)

# Console summary
print(f"\n  {'Target':<22} {'reg':>18} {'f_w':>5} {'Acc':>7} {'F1':>7} {'Rec':>7} {'MAE':>7}")
print("  " + "-" * 78)
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    print(f"  {t:<22} {r['reg_type']:>18} {r['freq_w']:>5.3f} {r['Accuracy']:>7.4f} "
          f"{r['F1']:>7.4f} {r['Recall']:>7.4f} {r['MAE']:>7.2f}")

print(f"\n  {'Metric':<32} {'True Optimal':>14} {'Best-per-REG':>14} {'Delta':>8}")
print(f"  {'-'*70}")
print(f"  {'Macro F1':<32} {REF['mf1']:>14.4f} {macro_f1:>14.4f} {macro_f1-REF['mf1']:>+8.4f}")
print(f"  {'Macro Recall':<32} {REF['mrec']:>14.4f} {macro_rec:>14.4f} {macro_rec-REF['mrec']:>+8.4f}")
print(f"  {'Macro Accuracy':<32} {'—':>14} {macro_acc:>14.4f} {'—':>8}")
print(f"  {'Freq-weighted F1':<32} {REF['fw_f1']:>14.4f} {fw_f1:>14.4f} {fw_f1-REF['fw_f1']:>+8.4f}")
print(f"  {'Freq-weighted Recall':<32} {REF['fw_rec']:>14.4f} {fw_rec:>14.4f} {fw_rec-REF['fw_rec']:>+8.4f}")
print(f"  {'Freq-weighted Accuracy':<32} {'—':>14} {fw_acc:>14.4f} {'—':>8}")
print(f"  {'Macro MAE (hrs)':<32} {REF['mmae']:>14.2f} {macro_mae:>14.2f} {macro_mae-REF['mmae']:>+8.2f}")
print(f"  {'Freq-weighted MAE (hrs)':<32} {REF['fw_mae']:>14.2f} {fw_mae:>14.2f} {fw_mae-REF['fw_mae']:>+8.2f}")

# ============================================================================
# STEP 5 — Write markdown report
# ============================================================================
section("STEP 5 — Writing markdown report")

tbl_rows = []
for t in OUTPUT_TARGETS:
    r = df.loc[t]
    tbl_rows.append(
        f"| `{t}` | {r['reg_type']} | {r['n_pos']} | {r['freq_w']:.3f} "
        f"| {r['Accuracy']:.4f} | {r['F1']:.4f} | {r['Recall']:.4f} | {r['MAE']:.2f} |"
    )

md = f"""# Best-Per-Target Regressor — Saved CLF + Mixed REG

**Date:** {date.today()}

**Goal:** Use the saved baseline classifiers from `models/two_stage_pipeline.pkl`
and train the best-performing regressor per target as documented in
`markdowns/regressor_selection.md`. For `hailrepair`, Ridge vs Ridge+winsorise
is selected automatically based on validation MAE.

## Regressor Assignments

| Target | Regressor | Reason |
|---|---|---|
| `calibration` | LightGBM | Lower val MAE (0.95 vs 1.88 hrs)  |
| `wheelmeasurement` | LightGBM | Lower val MAE (0.51 vs 0.55 hrs) |
| `bodymeasurement` | LightGBM | Lower val MAE (0.46 vs 5.65 hrs) |
| `dismounting` | LightGBM | Lower val MAE (1.14 vs 1.18 hrs) |
| `bodyrepair` | Ridge | Lower val MAE (2.96 vs 3.27 hrs) |
| `assembly` | Ridge | Lower val MAE (1.41 vs 1.43 hrs) |
| `plasticrepair` | LightGBM | Lower val MAE (0.67 vs 1.13 hrs) |
| `cleaning` | LightGBM | Tie — LightGBM primary choice |
| `paintingPreparation` | Ridge | Lower val MAE (1.62 vs 2.13 hrs) |
| `paintingSpraying` | LightGBM | Lower val MAE (0.64 vs 0.73 hrs) |
| `paintingFinish` | Ridge | Lower val MAE (0.93 vs 1.10 hrs) |
| `hailrepair` | {reg_choice.get('hailrepair','ridge')} | Chosen by val MAE comparison |
| `glas` | LightGBM | Lower val MAE (0.89 vs 1.59 hrs) |
| `allTiresService` | Fallback (mean) | n_pos < 5 |

## Comparison vs True Optimal (all-LightGBM + winsorise)

| Metric | True Optimal | **Best-per-REG** | Delta |
|---|---|---|---|
| Macro F1 | {REF['mf1']} | **{macro_f1}** | {macro_f1-REF['mf1']:+.4f} |
| Macro Recall | {REF['mrec']} | **{macro_rec}** | {macro_rec-REF['mrec']:+.4f} |
| Macro Accuracy | — | **{macro_acc}** | — |
| Freq-weighted F1 | {REF['fw_f1']} | **{fw_f1}** | {fw_f1-REF['fw_f1']:+.4f} |
| Freq-weighted Recall | {REF['fw_rec']} | **{fw_rec}** | {fw_rec-REF['fw_rec']:+.4f} |
| Freq-weighted Accuracy | — | **{fw_acc}** | — |
| Macro MAE | {REF['mmae']} hrs | **{macro_mae} hrs** | {macro_mae-REF['mmae']:+.2f} hrs |
| **Freq-weighted MAE** | {REF['fw_mae']} hrs | **{fw_mae} hrs** | {fw_mae-REF['fw_mae']:+.2f} hrs |

## Per-target Results

| Target | Regressor | n_pos | freq_w | Accuracy | F1 | Recall | MAE |
|--------|-----------|-------|--------|----------|----|--------|-----|
{chr(10).join(tbl_rows)}
| **MACRO** | — | — | — | **{macro_acc}** | **{macro_f1}** | **{macro_rec}** | **{macro_mae}** |
| **FREQ-WT** | — | — | — | **{fw_acc}** | **{fw_f1}** | **{fw_rec}** | **{fw_mae}** |
"""

md_path = MD_DIR / "best_per_target_reg_results.md"
md_path.write_text(md, encoding="utf-8")
print(f"\n  [saved] {md_path}")
print("  Done.")
