"""
model_gbert.py
==============
Two-Stage Work Step Time Prediction Pipeline — BERT Feature Version
==================================================================

Identical architecture to model_phase2.py, but the TF-IDF text features are
replaced by pre-computed gbert-base sentence embeddings (768-dim, L2-normalised).
Numeric / keyword / make features are kept unchanged.

Pre-requisite:
    Run extract_bert_embeddings.py first to produce:
        gbert_base/data/bert_embeddings.npy
        gbert_base/data/bert_order_index.json

Usage:
    python gbert_base/code/model_gbert.py
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — LightGBM models will be skipped.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = Path(r"C:\Users\Administrator\baum\data\orders_simplified_sample.json")
EMB_PATH  = ROOT / "data" / "bert_embeddings.npy"
PLOT_DIR  = ROOT / "model_plots"
MODEL_DIR = ROOT / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
COLORS    = sns.color_palette("muted", n_colors=14)
SECTION_BAR = "=" * 70

compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}


def section(title):
    print(f"\n{SECTION_BAR}\n  {title}\n{SECTION_BAR}")

def subsection(title):
    print(f"\n  -- {title} --")

def save_fig(name):
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ============================================================================
# STEP 1: LOAD & PREPROCESS
# ============================================================================
section("STEP 1 — Load & Preprocess")

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
        cc    = p.get("genericCostCenter") or "unknown_cc"
        if not text and price == 0 and time_ == 0:
            continue
        cleaned.append({"text": text, "totalPrice": price,
                         "totalTime": time_, "genericCostCenter": cc})
    return cleaned


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
section("STEP 2 — Feature Engineering")

# ── 2a. Load pre-computed BERT embeddings ─────────────────────────────────────
if not EMB_PATH.exists():
    raise FileNotFoundError(
        f"\n  [ERROR] BERT embeddings not found at:\n  {EMB_PATH}\n\n"
        "  Run first:\n    python gbert_base/code/extract_bert_embeddings.py\n"
    )
bert_emb = np.load(EMB_PATH)  # shape: (N, 768), float32, L2-normalised
print(f"\n  BERT embeddings loaded: {bert_emb.shape}  (dim=768, L2-normalised)")
assert bert_emb.shape[0] == len(valid), (
    f"Embedding count ({bert_emb.shape[0]}) != record count ({len(valid)}). "
    "Re-run extract_bert_embeddings.py."
)


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
    feats["ratio_material_price"] = feats["price_material"]  / feats["total_price"] \
                                    if feats["total_price"] > 0 else 0

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


print("\n  Building numeric features per order…")
records_feats  = []
records_target = []

for r in valid:
    positions = preprocess_record(r)
    make      = r["input"].get("make", "unknown")
    feats     = build_numeric_features(positions, make)
    records_feats.append(feats)
    records_target.append({t: float(r["output"].get(t, 0) or 0)
                           for t in OUTPUT_TARGETS})

# Frequency-encode make
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

# ── 2b. Combine: dense BERT (768) + numeric (as sparse) ──────────────────────
X_bert_sparse = sparse.csr_matrix(bert_emb)           # (N, 768)
X_num_sparse  = sparse.csr_matrix(df_num.values)      # (N, n_numeric)
X_all         = sparse.hstack([X_bert_sparse, X_num_sparse], format="csr")

print(f"  Final feature matrix: {X_all.shape[0]} records × {X_all.shape[1]} features")
print(f"    - BERT embeddings : {X_bert_sparse.shape[1]}")
print(f"    - Numeric/kw/make : {X_num_sparse.shape[1]}")

Y_binary = (df_targets > 0).astype(int)
Y_reg    = df_targets.copy()


# ============================================================================
# STEP 3: TRAIN / VAL / TEST SPLIT  (60 / 20 / 20)
# ============================================================================
section("STEP 3 — Train / Val / Test Split")

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
Y_bin_train = Y_binary.iloc[idx_train]
Y_bin_val   = Y_binary.iloc[idx_val]
Y_bin_test  = Y_binary.iloc[idx_test]
Y_reg_train = Y_reg.iloc[idx_train]
Y_reg_val   = Y_reg.iloc[idx_val]
Y_reg_test  = Y_reg.iloc[idx_test]

print(f"\n  Train : {len(idx_train):>3} records")
print(f"  Val   : {len(idx_val):>3} records")
print(f"  Test  : {len(idx_test):>3} records")


# ============================================================================
# STEP 4: STAGE 1 — CLASSIFICATION
# ============================================================================
section("STEP 4 — Stage 1: Multi-Label Classification")


def train_classifiers(X_tr, Y_tr):
    models = {"logreg": {}, "lgbm": {}}
    for t in OUTPUT_TARGETS:
        y    = Y_tr[t].values
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        w    = {0: 1.0, 1: max(1.0, n_neg / (n_pos + 1e-6))}
        lr   = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                  class_weight=w, random_state=RANDOM_STATE)
        lr.fit(X_tr, y)
        models["logreg"][t] = lr

        if HAS_LGB:
            scale = max(1.0, n_neg / (n_pos + 1e-6))
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                scale_pos_weight=scale, random_state=RANDOM_STATE,
                verbose=-1, n_jobs=1,
            )
            lgb_clf.fit(X_tr, y)
            models["lgbm"][t] = lgb_clf
    return models


print("\n  Training classifiers on train set…")
clf_models = train_classifiers(X_train, Y_bin_train)
clf_types  = [k for k in clf_models if clf_models[k]]


def get_proba(models, X, target, model_type):
    return models[model_type][target].predict_proba(X)[:, 1]


# ── 4a. Threshold tuning ──────────────────────────────────────────────────────
subsection("4a. Per-Target Threshold Tuning on Validation Set")

best_thresholds   = {}
val_clf_results   = {}

for mtype in clf_types:
    best_thresholds[mtype] = {}
    val_clf_results[mtype] = {}
    for t in OUTPUT_TARGETS:
        y_true = Y_bin_val[t].values
        y_prob = get_proba(clf_models, X_val, t, mtype)
        best_f1, best_thr = -1, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        best_thresholds[mtype][t] = best_thr
        y_best = (y_prob >= best_thr).astype(int)
        val_clf_results[mtype][t] = {
            "threshold": round(best_thr, 2),
            "F1":        round(f1_score(y_true, y_best, zero_division=0), 4),
            "Precision": round(precision_score(y_true, y_best, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_best, zero_division=0), 4),
            "AP":        round(average_precision_score(y_true, y_prob)
                               if y_true.sum() > 0 else 0, 4),
        }

for mtype in clf_types:
    print(f"\n  [{mtype.upper()}] Val Classification Results:")
    df_clf = pd.DataFrame(val_clf_results[mtype]).T.sort_values("F1", ascending=False)
    print(df_clf.to_string())

best_clf_type_per_target = {}
for t in OUTPUT_TARGETS:
    scores = {mtype: val_clf_results[mtype][t]["F1"] for mtype in clf_types}
    best_clf_type_per_target[t] = max(scores, key=scores.get)

print("\n  Best classifier per target:")
for t, m in best_clf_type_per_target.items():
    f1 = val_clf_results[m][t]["F1"]
    print(f"    {t:<25} -> {m:<8}  F1={f1:.4f}")

# ── 4b. LogReg precision / recall / accuracy ──────────────────────────────────
subsection("4b. Logistic Regression: Precision / Recall / Accuracy")


def logreg_metrics(X, Y_bin, split_name):
    rows = []
    for t in OUTPUT_TARGETS:
        y_true = Y_bin[t].values
        y_prob = get_proba(clf_models, X, t, "logreg")
        thr    = best_thresholds["logreg"][t]
        y_pred = (y_prob >= thr).astype(int)
        rows.append({
            "target":    t,
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Threshold": round(thr, 2),
        })
    df_out = pd.DataFrame(rows).set_index("target").sort_values("Recall", ascending=False)
    print(f"\n  [Logistic Regression — {split_name}]")
    print(df_out.to_string())
    return df_out


df_lr_val  = logreg_metrics(X_val,  Y_bin_val,  "Validation Set")
df_lr_test = logreg_metrics(X_test, Y_bin_test, "Test Set")

df_lr_val["split"]  = "val"
df_lr_test["split"] = "test"
pd.concat([df_lr_val, df_lr_test]).to_csv(PLOT_DIR / "gbert_logreg_clf_metrics.csv")
print(f"  [saved] {PLOT_DIR / 'gbert_logreg_clf_metrics.csv'}")

# ── F1 comparison plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(OUTPUT_TARGETS))
width = 0.35
for i, mtype in enumerate(clf_types):
    f1s = [val_clf_results[mtype][t]["F1"] for t in OUTPUT_TARGETS]
    ax.bar(x + i * width, f1s, width, label=mtype.upper(), alpha=0.85)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(OUTPUT_TARGETS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1 (occurrence)")
ax.set_ylim(0, 1.05)
ax.axhline(0.736, color="orange", linestyle="--", linewidth=1.5,
           label="TF-IDF baseline (0.736)")
ax.set_title("Stage 1: Classification F1 by Target — gbert-base (Validation Set)")
ax.legend()
plt.tight_layout()
save_fig("gbert_01_clf_f1_comparison")


# ============================================================================
# STEP 5: STAGE 2 — REGRESSION
# ============================================================================
section("STEP 5 — Stage 2: Conditional Regression (Positive Subset)")


def train_regressors(X_tr, Y_tr_reg, Y_tr_bin):
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
            print(f"  [WARN] {t}: only {n_pos} positive train records — using mean fallback")
            continue

        X_pos = X_tr[mask_pos]
        y_pos = Y_tr_reg.loc[Y_tr_bin[t] == 1, t].values
        train_stats[t]["fallback"] = False

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_pos, y_pos)
        models["ridge"][t] = ridge

        if HAS_LGB:
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=max(5, n_pos // 10),
                random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
            )
            lgb_reg.fit(X_pos, y_pos)
            models["lgbm"][t] = lgb_reg

    return models, train_stats


print("\n  Training regressors on positive train subsets…")
reg_models, reg_train_stats = train_regressors(X_train, Y_reg_train, Y_bin_train)


def reg_predict(models, X, target, model_type):
    m = models[model_type][target]
    if isinstance(m, tuple) and m[0] == "mean_fallback":
        return np.full(X.shape[0], m[1])
    return np.maximum(0.0, m.predict(X))


subsection("5a. Regression Validation (Positive Subset MAE)")
val_reg_results = {}
for mtype in (["ridge"] + (["lgbm"] if HAS_LGB else [])):
    val_reg_results[mtype] = {}
    for t in OUTPUT_TARGETS:
        mask_pos = Y_bin_val[t].values == 1
        n_pos = mask_pos.sum()
        if n_pos < 2:
            val_reg_results[mtype][t] = {"n_pos": n_pos, "MAE": float("nan"),
                                          "RMSE": float("nan"), "MAE_pct": float("nan")}
            continue
        X_pos  = X_val[mask_pos]
        y_true = Y_reg_val.loc[Y_bin_val[t] == 1, t].values
        y_pred = reg_predict(reg_models, X_pos, t, mtype)
        mae    = mean_absolute_error(y_true, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_pct = mae / (y_true.mean() + 1e-8) * 100
        val_reg_results[mtype][t] = {"n_pos": n_pos,
                                      "MAE": round(mae, 2),
                                      "RMSE": round(rmse, 2),
                                      "MAE_pct": round(mae_pct, 1)}

for mtype in val_reg_results:
    print(f"\n  [{mtype.upper()}] Val Regression Results (positive subset):")
    print(pd.DataFrame(val_reg_results[mtype]).T.to_string())


# ============================================================================
# STEP 6: FULL END-TO-END EVALUATION ON TEST SET
# ============================================================================
section("STEP 6 — Full End-to-End Evaluation on Test Set")


def end_to_end_predict(X, clf_models, reg_models, thresholds, clf_type_map):
    pred_bin = {}; pred_reg = {}
    for t in OUTPUT_TARGETS:
        mtype  = clf_type_map[t]
        y_prob = get_proba(clf_models, X, t, mtype)
        thr    = thresholds[mtype][t]
        y_bin  = (y_prob >= thr).astype(int)
        y_dur  = reg_predict(reg_models, X, t, "lgbm" if HAS_LGB else "ridge")
        pred_bin[t] = y_bin
        pred_reg[t] = np.where(y_bin == 1, y_dur, 0.0)
    return pd.DataFrame(pred_bin), pd.DataFrame(pred_reg)


print("\n  Running end-to-end predictions on test set…")
Y_pred_bin, Y_pred_reg = end_to_end_predict(
    X_test, clf_models, reg_models, best_thresholds, best_clf_type_per_target)
Y_pred_bin.index = Y_bin_test.index
Y_pred_reg.index = Y_reg_test.index

subsection("6a. Per-Target Test Metrics")
test_results = {}
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_pred_b   = Y_pred_bin[t].values
    y_true_reg = Y_reg_test[t].values
    y_pred_r   = Y_pred_reg[t].values
    f1   = f1_score(y_true_bin, y_pred_b, zero_division=0)
    prec = precision_score(y_true_bin, y_pred_b, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_b, zero_division=0)
    mae_e2e = mean_absolute_error(y_true_reg, y_pred_r)
    tp_mask = y_true_bin == 1
    mae_pos = (mean_absolute_error(y_true_reg[tp_mask], y_pred_r[tp_mask])
               if tp_mask.sum() >= 2 else float("nan"))
    test_results[t] = {
        "F1": round(f1, 4), "Precision": round(prec, 4), "Recall": round(rec, 4),
        "MAE_e2e": round(mae_e2e, 2),
        "MAE_pos": round(mae_pos, 2) if not np.isnan(mae_pos) else float("nan"),
        "n_true_pos": int(tp_mask.sum()),
    }

tdf = pd.DataFrame(test_results).T.sort_values("F1", ascending=False)
print(f"\n{tdf.to_string()}")

all_f1  = [v["F1"]      for v in test_results.values()]
all_mae = [v["MAE_e2e"] for v in test_results.values()]
print(f"\n  Macro-avg F1 (occurrence)  : {np.nanmean(all_f1):.4f}")
print(f"  Mean MAE end-to-end        : {np.nanmean(all_mae):.4f} hrs")
print(f"\n  TF-IDF baseline comparison : F1=0.703, Macro-Recall=0.779")

# Total repair time
subsection("6b. Total Repair Time MAE per Order")
total_true = Y_reg_test.sum(axis=1).values
total_pred = Y_pred_reg.sum(axis=1).values
mae_total  = mean_absolute_error(total_true, total_pred)
print(f"\n  Total repair time MAE (per order): {mae_total:.2f} hrs")
print(f"  Median true total time           : {np.median(total_true):.2f} hrs")
print(f"  Median pred total time           : {np.median(total_pred):.2f} hrs")

# ── Comparison bar chart ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sorted_by_f1 = tdf.sort_values("F1", ascending=True)
axes[0].barh(sorted_by_f1.index, sorted_by_f1["F1"], color=COLORS[0], alpha=0.85)
axes[0].axvline(0.703, color="orange", linestyle="--", linewidth=1.5,
                label="TF-IDF baseline (0.703)")
axes[0].set_xlabel("F1 (occurrence)")
axes[0].set_title("Test F1 per Target — gbert-base")
axes[0].legend(fontsize=8)
axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
sorted_by_mae = tdf.sort_values("MAE_e2e", ascending=False)
axes[1].barh(sorted_by_mae.index, sorted_by_mae["MAE_e2e"].clip(upper=300),
             color=COLORS[1], alpha=0.85)
axes[1].set_xlabel("MAE end-to-end (hrs, clipped at 300)")
axes[1].set_title("Test MAE per Target — gbert-base")
plt.tight_layout()
save_fig("gbert_02_test_metrics")


# ============================================================================
# STEP 7: FEATURE IMPORTANCE (numeric/keyword only — BERT dims not named)
# ============================================================================
section("STEP 7 — Feature Importance (Numeric/Keyword Features)")

BERT_DIM = bert_emb.shape[1]  # 768

if HAS_LGB:
    focus_imp = [t for t in OUTPUT_TARGETS
                 if reg_train_stats[t]["n_pos_train"] >= 20][:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, t in enumerate(focus_imp):
        m = reg_models["lgbm"].get(t)
        if m is None or isinstance(m, tuple):
            continue
        imp     = m.feature_importances_
        num_imp = imp[BERT_DIM:]   # skip the 768 BERT dims
        top_idx = np.argsort(num_imp)[::-1][:15]
        top_names = [numeric_feature_names[j] for j in top_idx]
        sns.barplot(x=num_imp[top_idx], y=top_names, ax=axes[i], palette="Blues_r")
        axes[i].set_title(f"LightGBM Regressor: {t}\n(numeric features)")
        axes[i].set_xlabel("Importance")
    for j in range(len(focus_imp), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Stage 2 LightGBM Regressor — Numeric Feature Importance", y=1.01)
    plt.tight_layout()
    save_fig("gbert_03_regressor_importance")


# ============================================================================
# STEP 8: SAVE PIPELINE
# ============================================================================
section("STEP 8 — Save Pipeline")

pipeline = {
    "bert_model_id":          "deepset/gbert-base",
    "bert_embedding_dim":     BERT_DIM,
    "numeric_features":       numeric_feature_names,
    "clf_models":             clf_models,
    "reg_models":             reg_models,
    "thresholds":             best_thresholds,
    "best_clf_per_target":    best_clf_type_per_target,
    "output_targets":         OUTPUT_TARGETS,
    "make_freq_lookup":       make_counts,
}
pkl_path = MODEL_DIR / "gbert_two_stage_pipeline.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\n  Pipeline saved to: {pkl_path}")

print(f"""
  +-------------------------------------------------------------------+
  |  gbert-base PIPELINE COMPLETE — SUMMARY                         |
  +-------------------------------------------------------------------+
  |  Records : {len(valid):>3} total / {len(idx_train):>3} train / {len(idx_val):>3} val / {len(idx_test):>3} test         |
  |  Features: {X_all.shape[1]:>5} ({BERT_DIM} BERT + {X_num_sparse.shape[1]} numeric)              |
  |  Macro-avg F1 (test)  : {np.nanmean(all_f1):.4f}                          |
  |  Mean MAE e2e (test)  : {np.nanmean(all_mae):.4f} hrs                      |
  |  TF-IDF baseline      : F1=0.703, Macro-Recall=0.779            |
  |  All plots saved to   : {str(PLOT_DIR):<35}  |
  +-------------------------------------------------------------------+
""")
