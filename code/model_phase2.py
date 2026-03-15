"""
Phase 2 - Two-Stage Model Pipeline
====================================
orders_simplified_sample.json
Multi-target Repair Order Prediction

Architecture:
  Stage 1 - Multi-label classifier (one per target): does work step occur?
  Stage 2 - Conditional regressor (one per target): how long does it take?

Units: input position-level `totalTime` fields are in MINUTES; output regression targets (per work step duration) are in HOURS.

Steps:
  1. Preprocessing
  2. Feature Engineering (numeric aggregates + TF-IDF word/char + keyword flags + make)
  3. Train / Val / Test split (60/20/20, stratified by repair complexity)
  4. Stage 1 - Logistic Regression + LightGBM classifiers, per-target threshold tuning
  5. Stage 2 - Ridge + LightGBM regressors (trained on positive subsets only)
  6. Full evaluation report (F1 + MAE per target + summaries)
  7. Feature importance + error analysis
  8. Sample prediction output

Outputs:
  - Console report
  - Plots saved to ./model_plots/
  - Best models saved to ./models/ (pickle)
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
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed - LightGBM models will be skipped.")

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent  # repo root
DATA_PATH  = ROOT / "data" / "orders_simplified_sample.json"
PLOT_DIR   = ROOT / "model_plots"
MODEL_DIR  = ROOT / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

OUTPUT_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas",
]

# Keyword binary flag patterns (domain-tuned German body shop terms)
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

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
COLORS = sns.color_palette("muted", n_colors=14)

SECTION_BAR = "=" * 70

def section(title):
    print(f"\n{SECTION_BAR}")
    print(f"  {title}")
    print(SECTION_BAR)

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
section("STEP 1 - Load & Preprocess")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

valid = [r for r in raw
         if "input" in r and "output" in r
         and "calculatedPositions" in r["input"]]
print(f"\n  Records loaded: {len(valid):,}")

def preprocess_record(record):
    """
    Clean a single record's calculatedPositions.
    Returns cleaned list of position dicts.
    """
    cleaned = []
    for p in record["input"]["calculatedPositions"]:
        text  = (p.get("text") or "").strip()
        price = float(p.get("totalPrice") or 0)
        time_ = float(p.get("totalTime")  or 0)
        cc    = p.get("genericCostCenter", None)

        # Clip negative prices (discount/correction rows)
        price = max(0.0, price)

        # Skip pure empty rows (no text, no price, no time)
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
# STEP 2: FEATURE ENGINEERING
# ============================================================================
section("STEP 2 - Feature Engineering")

compiled_keywords = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_FLAGS.items()}

def build_order_text(positions):
    """Concatenate all non-empty, useful position texts."""
    parts = []
    for p in positions:
        t = p["text"].strip()
        if t and (p["totalPrice"] > 0 or p["totalTime"] > 0):
            parts.append(t.lower())
    return " ".join(parts)

def build_numeric_features(positions, make):
    """Build aggregated numeric + keyword flag + make features for one order."""
    feats = {}

    df = pd.DataFrame(positions)
    df["totalPrice"] = df["totalPrice"].astype(float)
    df["totalTime"]  = df["totalTime"].astype(float)

    useful = df[(df["totalPrice"] > 0) | (df["totalTime"] > 0)]

    # ---- Count features ----
    feats["n_positions"]     = len(df)
    feats["n_useful"]        = len(useful)
    feats["n_zero_time"]     = (df["totalTime"] == 0).sum()
    feats["n_zero_price"]    = (df["totalPrice"] == 0).sum()

    # ---- Aggregate time/price ----
    feats["total_time"]      = df["totalTime"].sum()
    feats["total_price"]     = df["totalPrice"].sum()
    feats["max_time"]        = df["totalTime"].max()
    feats["mean_time"]       = df["totalTime"].mean()
    feats["median_time"]     = df["totalTime"].median()
    feats["max_price"]       = df["totalPrice"].max()
    feats["mean_price"]      = df["totalPrice"].mean()
    feats["std_time"]        = df["totalTime"].std(ddof=0)
    feats["std_price"]       = df["totalPrice"].std(ddof=0)

    # ---- Per cost-center sums/counts ----
    for cc in COST_CENTERS:
        sub = df[df["genericCostCenter"] == cc]
        feats[f"time_{cc}"]  = sub["totalTime"].sum()
        feats[f"price_{cc}"] = sub["totalPrice"].sum()
        feats[f"n_{cc}"]     = len(sub)

    # ---- Ratios ----
    tt = feats["total_time"]
    feats["ratio_painting_time"]  = feats["time_painting"]  / tt if tt > 0 else 0
    feats["ratio_bodywork_time"]  = feats["time_bodywork"]   / tt if tt > 0 else 0
    feats["ratio_hail_time"]      = feats["time_hail"]       / tt if tt > 0 else 0
    feats["ratio_material_price"] = feats["price_material"]  / feats["total_price"] \
                                    if feats["total_price"] > 0 else 0

    # ---- Boolean indicators ----
    feats["has_hail_cc"]     = int(feats["n_hail"] > 0)
    feats["has_painting_cc"] = int(feats["n_painting"] > 0)
    feats["has_bodywork_cc"] = int(feats["n_bodywork"] > 0)

    # ---- Keyword binary flags ----
    combined_text = " ".join(p["text"].lower() for p in positions)
    for kw, pattern in compiled_keywords.items():
        feats[kw] = int(bool(pattern.search(combined_text)))

    # ---- Make encoding ----
    make_clean = (make or "unknown").strip().upper()
    feats["make_freq_raw"] = make_clean   # will be encoded below
    for m in TOP_MAKES:
        feats[f"make_{m.replace('-','_')}"] = int(make_clean == m)
    feats["make_other"] = int(make_clean not in TOP_MAKES)

    return feats

# Build raw feature list (before TF-IDF)
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

# Frequency-encode make
make_list   = [f["make_freq_raw"] for f in records_feats]
make_counts = pd.Series(make_list).value_counts().to_dict()
for f in records_feats:
    f["make_freq"] = make_counts.get(f["make_freq_raw"], 1)
    del f["make_freq_raw"]

# Assemble numeric DataFrame
df_num     = pd.DataFrame(records_feats).fillna(0).astype(float)
df_targets = pd.DataFrame(records_target)
n_records  = len(df_num)

numeric_feature_names = list(df_num.columns)
print(f"  Numeric/keyword/make features: {len(numeric_feature_names)}")

# TF-IDF: word n-grams (1-2)
print("  Fitting TF-IDF word n-grams (1-2)...")
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=2000,
    min_df=2,
    sublinear_tf=True,
    analyzer="word",
    token_pattern=r"(?u)\b[\w\+]+\b",  # keep A+E etc.
)
X_word = tfidf_word.fit_transform(records_text)

# TF-IDF: character n-grams (3-5)
print("  Fitting TF-IDF char n-grams (3-5)...")
tfidf_char = TfidfVectorizer(
    ngram_range=(3, 5),
    max_features=1000,
    min_df=3,
    sublinear_tf=True,
    analyzer="char_wb",
)
X_char = tfidf_char.fit_transform(records_text)

# Combine: sparse TF-IDF + dense numeric (as sparse)
X_num_sparse = sparse.csr_matrix(df_num.values)
X_all        = sparse.hstack([X_word, X_char, X_num_sparse], format="csr")

print(f"  Final feature matrix: {X_all.shape[0]} records x {X_all.shape[1]} features")
print(f"    - Word TF-IDF : {X_word.shape[1]}")
print(f"    - Char TF-IDF : {X_char.shape[1]}")
print(f"    - Numeric/kw  : {X_num_sparse.shape[1]}")

# Binary occurrence targets
Y_binary = (df_targets > 0).astype(int)
# Regression targets (minutes, raw)
Y_reg    = df_targets.copy()


# ============================================================================
# STEP 3: TRAIN / VAL / TEST SPLIT
# ============================================================================
section("STEP 3 - Train / Val / Test Split")

# Stratify by repair complexity band (number of active targets)
n_active = Y_binary.sum(axis=1).values
complexity_band = pd.cut(n_active, bins=[-1, 3, 6, 9, 100],
                         labels=["low", "medium", "high", "very_high"])

indices = np.arange(n_records)
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE,
    stratify=complexity_band
)
# Recompute band for trainval subset for second split
cb_trainval = complexity_band[idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=RANDOM_STATE,
    stratify=cb_trainval
)

X_train = X_all[idx_train]
X_val   = X_all[idx_val]
X_test  = X_all[idx_test]

Y_bin_train = Y_binary.iloc[idx_train]
Y_bin_val   = Y_binary.iloc[idx_val]
Y_bin_test  = Y_binary.iloc[idx_test]

Y_reg_train = Y_reg.iloc[idx_train]
Y_reg_val   = Y_reg.iloc[idx_val]
Y_reg_test  = Y_reg.iloc[idx_test]

print(f"\n  Train : {len(idx_train):>3} records")
print(f"  Val   : {len(idx_val):>3} records")
print(f"  Test  : {len(idx_test):>3} records")
print(f"\n  Complexity distribution (train):")
print(pd.Series(n_active[idx_train]).value_counts().sort_index().to_string())


# ============================================================================
# STEP 4: STAGE 1 - CLASSIFICATION (occurrence)
# ============================================================================
section("STEP 4 - Stage 1: Multi-Label Classification")

def train_classifiers(X_tr, Y_tr):
    """Train one classifier per target. Returns dict of fitted models."""
    models = {"logreg": {}, "lgbm": {}}

    for t in OUTPUT_TARGETS:
        y = Y_tr[t].values
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        # Logistic Regression
        w = {0: 1.0, 1: max(1.0, n_neg / (n_pos + 1e-6))}
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight=w, random_state=RANDOM_STATE
        )
        lr.fit(X_tr, y)
        models["logreg"][t] = lr

        # LightGBM
        if HAS_LGB:
            scale = max(1.0, n_neg / (n_pos + 1e-6))
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            lgb_clf.fit(X_tr, y)
            models["lgbm"][t] = lgb_clf

    return models

print("\n  Training classifiers on train set...")
clf_models = train_classifiers(X_train, Y_bin_train)
clf_types  = [k for k in clf_models if clf_models[k]]

def get_proba(models, X, target, model_type):
    m = models[model_type][target]
    return m.predict_proba(X)[:, 1]

# ---- Threshold tuning on validation set ----
subsection("4a. Per-Target Threshold Tuning on Validation Set")

best_thresholds = {}   # {model_type: {target: threshold}}
val_clf_results = {}   # {model_type: {target: {metric: value}}}

for mtype in clf_types:
    best_thresholds[mtype] = {}
    val_clf_results[mtype] = {}

    for t in OUTPUT_TARGETS:
        y_true = Y_bin_val[t].values
        y_prob = get_proba(clf_models, X_val, t, mtype)

        # Sweep thresholds, pick best F1
        best_f1, best_thr = -1, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1   = f1
                best_thr  = thr

        best_thresholds[mtype][t] = best_thr
        y_best = (y_prob >= best_thr).astype(int)
        val_clf_results[mtype][t] = {
            "threshold":  round(best_thr, 2),
            "F1":         round(f1_score(y_true, y_best, zero_division=0), 4),
            "Precision":  round(precision_score(y_true, y_best, zero_division=0), 4),
            "Recall":     round(recall_score(y_true, y_best, zero_division=0), 4),
            "AP":         round(average_precision_score(y_true, y_prob)
                                if y_true.sum() > 0 else 0, 4),
        }

# Print comparison table
for mtype in clf_types:
    print(f"\n  [{mtype.upper()}] Val Classification Results:")
    df_clf = pd.DataFrame(val_clf_results[mtype]).T
    df_clf = df_clf.sort_values("F1", ascending=False)
    print(df_clf.to_string())

# ---- Choose best classifier per target ----
best_clf_type_per_target = {}
for t in OUTPUT_TARGETS:
    scores = {mtype: val_clf_results[mtype][t]["F1"] for mtype in clf_types}
    best_clf_type_per_target[t] = max(scores, key=scores.get)

print("\n  Best classifier per target:")
for t, m in best_clf_type_per_target.items():
    f1 = val_clf_results[m][t]["F1"]
    print(f"    {t:<25} -> {m:<8}  F1={f1:.4f}")

# ---- 4b. Logistic Regression standalone: Precision / Recall / Accuracy ----
subsection("4b. Logistic Regression: Precision / Recall / Accuracy")

def logreg_metrics(X, Y_bin, split_name):
    """Compute precision, recall, accuracy for LogReg per target on a given split."""
    rows = []
    for t in OUTPUT_TARGETS:
        y_true = Y_bin[t].values
        y_prob  = get_proba(clf_models, X, t, "logreg")
        thr     = best_thresholds["logreg"][t]
        y_pred  = (y_prob >= thr).astype(int)
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

# Save combined table
df_lr_val["split"]  = "val"
df_lr_test["split"] = "test"
pd.concat([df_lr_val, df_lr_test]).to_csv(
    PLOT_DIR / "logreg_clf_metrics.csv"
)
print(f"  [saved] {PLOT_DIR / 'logreg_clf_metrics.csv'}")

# ---- Plot: Val F1 comparison ----
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
           label="Rule baseline (0.736)")
ax.set_title("Stage 1: Classification F1 by Target (Validation Set, Tuned Threshold)")
ax.legend()
plt.tight_layout()
save_fig("15_clf_f1_comparison")

# ---- Precision-Recall curves (best model per target) ----
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, t in enumerate(OUTPUT_TARGETS):
    ax = axes[i]
    mtype = best_clf_type_per_target[t]
    y_true = Y_bin_val[t].values
    y_prob = get_proba(clf_models, X_val, t, mtype)
    if y_true.sum() < 2:
        ax.text(0.5, 0.5, "Too sparse", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(t, fontsize=8)
        continue
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, color=COLORS[i], lw=1.5)
    ax.fill_between(rec, prec, alpha=0.15, color=COLORS[i])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title(f"{t}\nAP={ap:.2f}", fontsize=8)
    ax.set_xlabel("Recall", fontsize=7)
    ax.set_ylabel("Precision", fontsize=7)
for j in range(len(OUTPUT_TARGETS), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Stage 1: Precision-Recall Curves per Target (Validation Set)", y=1.01)
plt.tight_layout()
save_fig("16_pr_curves")


# ============================================================================
# STEP 5: STAGE 2 - REGRESSION (duration, on positive subset)
# ============================================================================
section("STEP 5 - Stage 2: Conditional Regression (Positive Subset)")

def train_regressors(X_tr, Y_tr_reg, Y_tr_bin):
    """
    Train one regressor per target, only on records where target > 0.
    Returns dict of fitted models and fit stats.
    """
    models = {"ridge": {}, "lgbm": {}}
    train_stats = {}

    for t in OUTPUT_TARGETS:
        mask_pos = Y_tr_bin[t].values == 1
        n_pos    = mask_pos.sum()

        train_stats[t] = {"n_pos_train": n_pos}

        if n_pos < 5:
            # Too few positive examples: fallback to mean predictor
            mean_val = Y_tr_reg.loc[Y_tr_bin[t] == 1, t].mean()
            models["ridge"][t] = ("mean_fallback", mean_val)
            models["lgbm"][t]  = ("mean_fallback", mean_val)
            train_stats[t]["fallback"] = True
            print(f"  [WARN] {t}: only {n_pos} positive train records - using mean fallback")
            continue

        X_pos = X_tr[mask_pos]
        y_pos = Y_tr_reg.loc[Y_tr_bin[t] == 1, t].values
        train_stats[t]["fallback"] = False

        # Ridge
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_pos, y_pos)
        models["ridge"][t] = ridge

        # LightGBM
        if HAS_LGB:
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=max(5, n_pos // 10),
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            )
            lgb_reg.fit(X_pos, y_pos)
            models["lgbm"][t] = lgb_reg

    return models, train_stats

print("\n  Training regressors on positive train subsets...")
reg_models, reg_train_stats = train_regressors(X_train, Y_reg_train, Y_bin_train)

def reg_predict(models, X, target, model_type):
    m = models[model_type][target]
    if isinstance(m, tuple) and m[0] == "mean_fallback":
        return np.full(X.shape[0], m[1])
    return np.maximum(0.0, m.predict(X))

# ---- Evaluate regressors on validation set (positive subset only) ----
subsection("5a. Regression Validation (Positive Subset MAE)")
val_reg_results = {}
for mtype in ["ridge"] + (["lgbm"] if HAS_LGB else []):
    val_reg_results[mtype] = {}
    for t in OUTPUT_TARGETS:
        mask_pos = Y_bin_val[t].values == 1
        n_pos    = mask_pos.sum()
        if n_pos < 2:
            val_reg_results[mtype][t] = {"n_pos": n_pos, "MAE": float("nan"),
                                          "RMSE": float("nan"), "MAE_pct": float("nan")}
            continue
        X_pos    = X_val[mask_pos]
        y_true   = Y_reg_val.loc[Y_bin_val[t] == 1, t].values
        y_pred   = reg_predict(reg_models, X_pos, t, mtype)
        mae      = mean_absolute_error(y_true, y_pred)
        rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_pct  = mae / (y_true.mean() + 1e-8) * 100
        val_reg_results[mtype][t] = {
            "n_pos":   n_pos,
            "MAE":     round(mae, 2),
            "RMSE":    round(rmse, 2),
            "MAE_pct": round(mae_pct, 1),
        }

for mtype in val_reg_results:
    print(f"\n  [{mtype.upper()}] Val Regression Results (positive subset):")
    rdf = pd.DataFrame(val_reg_results[mtype]).T
    print(rdf.to_string())


# ============================================================================
# STEP 6: FULL EVALUATION ON TEST SET
# ============================================================================
section("STEP 6 - Full End-to-End Evaluation on Test Set")

def end_to_end_predict(X, clf_models, reg_models, thresholds, clf_type_map):
    """
    Full two-stage prediction.
    Returns (Y_pred_binary, Y_pred_reg) DataFrames.
    """
    pred_bin = {}
    pred_reg = {}
    for t in OUTPUT_TARGETS:
        mtype    = clf_type_map[t]
        y_prob   = get_proba(clf_models, X, t, mtype)
        thr      = thresholds[mtype][t]
        y_bin    = (y_prob >= thr).astype(int)
        y_dur    = reg_predict(reg_models, X, t, "lgbm" if HAS_LGB else "ridge")

        pred_bin[t] = y_bin
        pred_reg[t] = np.where(y_bin == 1, y_dur, 0.0)

    return pd.DataFrame(pred_bin), pd.DataFrame(pred_reg)

print("\n  Running end-to-end predictions on test set...")
Y_pred_bin, Y_pred_reg = end_to_end_predict(
    X_test, clf_models, reg_models,
    best_thresholds, best_clf_type_per_target
)

# Align indices
Y_pred_bin.index = Y_bin_test.index
Y_pred_reg.index = Y_reg_test.index

# ---- Per-target metrics ----
subsection("6a. Per-Target Test Metrics")
test_results = {}
for t in OUTPUT_TARGETS:
    y_true_bin = Y_bin_test[t].values
    y_pred_b   = Y_pred_bin[t].values
    y_true_reg = Y_reg_test[t].values
    y_pred_r   = Y_pred_reg[t].values

    f1    = f1_score(y_true_bin, y_pred_b, zero_division=0)
    prec  = precision_score(y_true_bin, y_pred_b, zero_division=0)
    rec   = recall_score(y_true_bin, y_pred_b, zero_division=0)

    # MAE end-to-end (on all records)
    mae_e2e = mean_absolute_error(y_true_reg, y_pred_r)

    # MAE on true-positive records only
    tp_mask = y_true_bin == 1
    if tp_mask.sum() >= 2:
        mae_pos = mean_absolute_error(y_true_reg[tp_mask], y_pred_r[tp_mask])
    else:
        mae_pos = float("nan")

    test_results[t] = {
        "F1":         round(f1, 4),
        "Precision":  round(prec, 4),
        "Recall":     round(rec, 4),
        "MAE_e2e":    round(mae_e2e, 2),
        "MAE_pos":    round(mae_pos, 2) if not np.isnan(mae_pos) else float("nan"),
        "n_true_pos": int(tp_mask.sum()),
    }

tdf = pd.DataFrame(test_results).T.sort_values("F1", ascending=False)
print(f"\n{tdf.to_string()}")

# Summary
all_f1 = [v["F1"] for v in test_results.values()]
all_mae = [v["MAE_e2e"] for v in test_results.values()]
print(f"\n  Macro-avg F1 (occurrence)  : {np.nanmean(all_f1):.4f}")
print(f"  Mean MAE end-to-end        : {np.nanmean(all_mae):.4f} min")
print(f"\n  Rule baseline comparison   : F1=0.737, MAE=20.5 min")

# ---- Total repair time per order ----
subsection("6b. Total Repair Time MAE per Order")
total_true = Y_reg_test.sum(axis=1).values
total_pred = Y_pred_reg.sum(axis=1).values
mae_total  = mean_absolute_error(total_true, total_pred)
print(f"\n  Total repair time MAE (per order): {mae_total:.2f} min")
print(f"  Median true total time           : {np.median(total_true):.2f} min")
print(f"  Median pred total time           : {np.median(total_pred):.2f} min")

# ---- Plot: F1 and MAE comparison ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sorted_by_f1 = tdf.sort_values("F1", ascending=True)

# F1 vs rule baseline
axes[0].barh(sorted_by_f1.index, sorted_by_f1["F1"], color=COLORS[0], alpha=0.85)
axes[0].axvline(0.736, color="orange", linestyle="--", linewidth=1.5,
                label="Rule baseline (0.736)")
axes[0].set_xlabel("F1 (occurrence)")
axes[0].set_title("Test F1 per Target (Two-Stage Model)")
axes[0].legend(fontsize=8)
axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

# MAE end-to-end
sorted_by_mae = tdf.sort_values("MAE_e2e", ascending=False)
axes[1].barh(sorted_by_mae.index, sorted_by_mae["MAE_e2e"].clip(upper=300),
             color=COLORS[1], alpha=0.85)
axes[1].set_xlabel("MAE end-to-end (hours, clipped at 300)")
axes[1].set_title("Test MAE per Target")
plt.tight_layout()
save_fig("17_test_metrics")

# ---- True vs Predicted scatter for top targets ----
focus_targets = ["cleaning", "dismounting", "assembly",
                 "paintingSpraying", "paintingPreparation"]
fig, axes = plt.subplots(1, len(focus_targets), figsize=(18, 4))
for i, t in enumerate(focus_targets):
    y_t = Y_reg_test[t].values
    y_p = Y_pred_reg[t].values
    axes[i].scatter(y_t, y_p, alpha=0.5, s=15, color=COLORS[i])
    lim = max(y_t.max(), y_p.max()) * 1.05
    axes[i].plot([0, lim], [0, lim], "r--", linewidth=1)
    axes[i].set_xlabel("True (min)")
    axes[i].set_ylabel("Predicted (min)")
    mae_val = test_results[t]["MAE_e2e"]
    axes[i].set_title(f"{t}\nMAE={mae_val:.1f}min", fontsize=9)
plt.suptitle("True vs Predicted Duration (Test Set)", y=1.02)
plt.tight_layout()
save_fig("18_true_vs_pred_scatter")


# ============================================================================
# STEP 7: FEATURE IMPORTANCE & EXPLAINABILITY
# ============================================================================
section("STEP 7 - Feature Importance & Explainability")

feature_names_word = tfidf_word.get_feature_names_out().tolist()
feature_names_char = [f"char_{c}" for c in tfidf_char.get_feature_names_out()]
feature_names_num  = numeric_feature_names
all_feature_names  = feature_names_word + feature_names_char + feature_names_num

subsection("7a. Top TF-IDF Terms per Target (Logistic Regression)")
logreg_importance = {}
for t in OUTPUT_TARGETS:
    m = clf_models["logreg"][t]
    if not hasattr(m, "coef_"):
        continue
    coef = m.coef_[0]
    # Only look at word-ngram features for interpretability
    word_coef = coef[:len(feature_names_word)]
    top_idx   = np.argsort(word_coef)[::-1][:10]
    logreg_importance[t] = [(feature_names_word[i], round(word_coef[i], 4))
                             for i in top_idx]

print(f"\n  Top 5 positive TF-IDF tokens per target (Logistic Regression):")
print(f"  {'Target':<25} {'Top tokens'}")
print("  " + "-" * 70)
for t in OUTPUT_TARGETS:
    if t not in logreg_importance:
        continue
    top = [f[0] for f in logreg_importance[t][:5]]
    print(f"  {t:<25} {', '.join(top)}")

subsection("7b. LightGBM Feature Importance (Top Targets)")
if HAS_LGB:
    # Pick targets with enough positive examples
    focus_imp = [t for t in OUTPUT_TARGETS
                 if reg_train_stats[t]["n_pos_train"] >= 20][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, t in enumerate(focus_imp):
        m = reg_models["lgbm"].get(t)
        if m is None or isinstance(m, tuple):
            continue
        imp = m.feature_importances_
        n_tok = len(feature_names_word) + len(feature_names_char)
        # Only numeric/keyword features for legibility
        num_imp  = imp[n_tok:]
        top_idx  = np.argsort(num_imp)[::-1][:15]
        top_names = [feature_names_num[j] for j in top_idx]
        top_vals  = num_imp[top_idx]

        ax = axes[i]
        sns.barplot(x=top_vals, y=top_names, ax=ax, palette="Blues_r")
        ax.set_title(f"LightGBM Regressor: {t}\n(numeric feature importance)")
        ax.set_xlabel("Importance")
    for j in range(len(focus_imp), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Stage 2: LightGBM Regressor Feature Importance (Numeric Features)", y=1.01)
    plt.tight_layout()
    save_fig("19_lgbm_regressor_importance")

    # Classifier importance for top 6 targets
    focus_clf_imp = [t for t in OUTPUT_TARGETS
                     if Y_bin_test[t].sum() >= 5][:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, t in enumerate(focus_clf_imp):
        m = clf_models["lgbm"].get(t)
        if m is None:
            continue
        imp = m.feature_importances_
        n_tok = len(feature_names_word) + len(feature_names_char)
        num_imp  = imp[n_tok:]
        top_idx  = np.argsort(num_imp)[::-1][:15]
        top_names = [feature_names_num[j] for j in top_idx]
        top_vals  = num_imp[top_idx]
        ax = axes[i]
        sns.barplot(x=top_vals, y=top_names, ax=ax, palette="Greens_r")
        ax.set_title(f"LightGBM Classifier: {t}")
        ax.set_xlabel("Importance")
    for j in range(len(focus_clf_imp), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Stage 1: LightGBM Classifier Feature Importance (Numeric Features)", y=1.01)
    plt.tight_layout()
    save_fig("20_lgbm_classifier_importance")

# ---- Error analysis ----
subsection("7c. Error Analysis - Worst False Positives & False Negatives")

for t in ["calibration", "glas", "assembly"]:
    y_true_b = Y_bin_test[t].values
    y_pred_b = Y_pred_bin[t].values
    y_true_r = Y_reg_test[t].values
    y_pred_r = Y_pred_reg[t].values
    local_idx = Y_bin_test.index.values

    fp_mask = (y_pred_b == 1) & (y_true_b == 0)
    fn_mask = (y_pred_b == 0) & (y_true_b == 1)
    tp_mask = (y_pred_b == 1) & (y_true_b == 1)

    print(f"\n  [{t}] FP={fp_mask.sum()}  FN={fn_mask.sum()}  TP={tp_mask.sum()}")
    if tp_mask.sum() >= 2:
        tp_errors = np.abs(y_pred_r[tp_mask] - y_true_r[tp_mask])
        worst_tp  = np.argsort(tp_errors)[::-1][:5]
        print(f"    Worst TP duration errors (true | pred | error in min):")
        for w in worst_tp:
            tr = y_true_r[tp_mask][w]
            pr = y_pred_r[tp_mask][w]
            print(f"      true={tr:.1f}  pred={pr:.1f}  err={abs(tr-pr):.1f}")


# ============================================================================
# STEP 8: SAMPLE PREDICTION OUTPUT
# ============================================================================
section("STEP 8 - Sample Prediction Output")

print(f"\n  Showing predictions for 5 test records:\n")
print(f"  {'Order':<6} {'Make':<18} {'Positions':>9} "
      f"{'True Active':>12} {'Pred Active':>12} {'Total True':>11} {'Total Pred':>11}")
print("  " + "-" * 80)

for k, idx in enumerate(idx_test[:5]):
    r       = valid[idx]
    make    = r["input"].get("make", "unknown")
    n_pos   = len(r["input"]["calculatedPositions"])
    t_true  = int(Y_binary.iloc[idx].sum())
    t_pred  = int(Y_pred_bin.iloc[k].sum())
    tot_t   = Y_reg_test.iloc[k].sum()
    tot_p   = Y_pred_reg.iloc[k].sum()
    print(f"  {k:<6} {make:<18} {n_pos:>9} {t_true:>12} {t_pred:>12} "
          f"{tot_t:>11.1f} {tot_p:>11.1f}")

# Detailed view for first test record
print(f"\n  Detailed view - Test Record 0:")
print(f"\n  {'Target':<25} {'True (min)':>12} {'Pred (min)':>12} {'Error':>10} {'Correct?':>9}")
print("  " + "-" * 65)
for t in OUTPUT_TARGETS:
    tr = Y_reg_test.iloc[0][t]
    pr = Y_pred_reg.iloc[0][t]
    err = abs(tr - pr)
    correct = "YES" if (tr > 0) == (pr > 0) else "NO"
    print(f"  {t:<25} {tr:>12.2f} {pr:>12.2f} {err:>10.2f} {correct:>9}")

print(f"\n  Total true  : {Y_reg_test.iloc[0].sum():.2f} min")
print(f"  Total pred  : {Y_pred_reg.iloc[0].sum():.2f} min")


# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================
section("STEP 9 - Save Pipeline")

pipeline = {
    "tfidf_word":        tfidf_word,
    "tfidf_char":        tfidf_char,
    "numeric_features":  numeric_feature_names,
    "clf_models":        clf_models,
    "reg_models":        reg_models,
    "thresholds":        best_thresholds,
    "best_clf_per_target": best_clf_type_per_target,
    "output_targets":    OUTPUT_TARGETS,
}
pkl_path = MODEL_DIR / "two_stage_pipeline.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"\n  Pipeline saved to: {pkl_path}")

print(f"""
  +-------------------------------------------------------------------+
  |  PHASE 2 COMPLETE - SUMMARY                                      |
  +-------------------------------------------------------------------+
  |  Records: {len(valid):>3} total / {len(idx_train):>3} train / {len(idx_val):>3} val / {len(idx_test):>3} test          |
  |  Features: {X_all.shape[1]:>5} ({X_word.shape[1]} word TF + {X_char.shape[1]} char TF + {X_num_sparse.shape[1]} num)     |
  |  Macro-avg F1 (test)  : {np.nanmean(all_f1):.4f}                          |
  |  Mean MAE e2e (test)  : {np.nanmean(all_mae):.2f} min                       |
  |  Rule baseline        : F1=0.736, MAE=20.5 min                  |
  |  All plots saved to   : {str(PLOT_DIR):<35}  |
  +-------------------------------------------------------------------+
""")
