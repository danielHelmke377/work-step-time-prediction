# -*- coding: utf-8 -*-
"""
Phase 1 — Data Analysis Script
================================
orders_simplified_sample.json
EDA for Multi-Target Repair Order Prediction

Steps:
  1. Load & Structural Validation
  2. Target Variable Analysis (sparsity, distributions, co-occurrence)
  3. Input Field Quality Audit (make, cost centers, text, numerics)
  4. Numeric Consistency & Unit Verification
  5. Feature Engineering Candidate Assessment
  6. Rule-Based Baseline
  7. Stakeholder Questions Summary

Outputs:
  - Console report with all key stats
  - Plots saved to ./eda_plots/
"""

import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Global Config ────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent  # repo root
DATA_PATH   = ROOT / "data" / "orders_simplified_sample.json"
PLOT_DIR    = ROOT / "eda_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas",
]

# Cost-center → output-target semantic mapping (for unit cross-validation)
CC_TO_TARGET = {
    "bodywork":      ["dismounting", "assembly", "bodyrepair", "bodymeasurement",
                      "wheelmeasurement", "calibration", "cleaning"],
    "painting":      ["paintingPreparation", "paintingSpraying", "paintingFinish"],
    "paintmaterial": [],
    "material":      [],
    "others":        [],
}

# Keyword → target mapping for rule-based baseline
KEYWORD_RULES = {
    "calibration":        r"kalibrier|adas|fas|kamera|radar|sensor.*ausrich",
    "wheelmeasurement":   r"fahrwerksvermess|kinematik|spur|sturzwinkel|radvermess",
    "bodymeasurement":    r"karosserievermess|3d.*vermess|richtrahmen|vermess.*kaross",
    "dismounting":        r"\ba\+e\b|de-?/montage|ausgebaut|ausbauen|einbauen|demontage",
    "bodyrepair":         r"blech|beul|dent|richt|karosserie.*repar|schweiss",
    "assembly":           r"montage(?!.*de)|zusammenbau|verbindung.*teil|einbau(?!en)",
    "plasticrepair":      r"plastik|kunststoff|stoßfänger.*repar|bumper.*repar",
    "allTiresService":    r"reifen|rad(?:wechsel|service|montage)|felge",
    "cleaning":           r"reinigung|wäsche|polier|entstörung",
    "paintingPreparation":r"vorbereitung.*lack|grundier|füller|vorbereit.*lack",
    "paintingSpraying":   r"lackier|neu.*lackier|spritz",
    "paintingFinish":     r"finish|polier|lack.*fertig|abschlusspolit",
    "hailrepair":         r"hagel|dellen(?!.*repar)|pdr|smart.*repar",
    "glas":               r"glas|scheibe|frontscheibe|windschutz|heckscheibe",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
COLORS       = sns.color_palette("muted", n_colors=14)
SECTION_BAR  = "=" * 70

def section(title):
    print(f"\n{SECTION_BAR}")
    print(f"  {title}")
    print(SECTION_BAR)

def subsection(title):
    print(f"\n  ── {title} ──")

def save_fig(name):
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & STRUCTURAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Load & Structural Validation")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

n_records = len(raw)
print(f"\n  Total records: {n_records:,}")
print(f"  File size    : {DATA_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# Check structural integrity
missing_input        = sum(1 for r in raw if "input"  not in r)
missing_output       = sum(1 for r in raw if "output" not in r)
missing_positions    = sum(1 for r in raw
                           if "input" in r and
                           "calculatedPositions" not in r.get("input", {}))
missing_make         = sum(1 for r in raw
                           if "input" in r and "make" not in r.get("input", {}))

print(f"\n  Schema Completeness:")
print(f"    Missing 'input'              : {missing_input}")
print(f"    Missing 'output'             : {missing_output}")
print(f"    Missing 'calculatedPositions': {missing_positions}")
print(f"    Missing 'make'               : {missing_make}")

# Check output keys
missing_target_keys = {}
for t in OUTPUT_TARGETS:
    c = sum(1 for r in raw if "output" in r and t not in r["output"])
    if c:
        missing_target_keys[t] = c

if missing_target_keys:
    print(f"\n  ⚠ Missing output keys per target:")
    for k, v in missing_target_keys.items():
        print(f"    {k}: {v} records missing")
else:
    print(f"\n  [OK] All 14 output targets present in every record")

# Build clean DataFrame of records that have both input and output
valid   = [r for r in raw if "input" in r and "output" in r
           and "calculatedPositions" in r["input"]]
invalid = n_records - len(valid)
print(f"\n  Valid records (both input + output): {len(valid):,}")
if invalid:
    print(f"  [WARN] Skipped {invalid} malformed records")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TARGET VARIABLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Target Variable Analysis")

# Build targets dataframe
df_targets = pd.DataFrame([r["output"] for r in valid])[OUTPUT_TARGETS].fillna(0.0)

# ── 2a. Sparsity ──────────────────────────────────────────────────────────────
subsection("2a. Sparsity per Target")

sparsity = {}
for t in OUTPUT_TARGETS:
    n_zero    = (df_targets[t] == 0).sum()
    n_nonzero = (df_targets[t] >  0).sum()
    sparsity[t] = {
        "n_zero":    n_zero,
        "n_nonzero": n_nonzero,
        "pct_zero":  100 * n_zero / len(df_targets),
    }

sdf = pd.DataFrame(sparsity).T.sort_values("pct_zero", ascending=False)
print(f"\n  {'Target':<25} {'# Zero':>8} {'# Active':>9} {'% Zero':>8}")
print("  " + "-" * 54)
for t, row in sdf.iterrows():
    bar = "|" * int(row["pct_zero"] / 5)
    print(f"  {t:<25} {int(row['n_zero']):>8,} {int(row['n_nonzero']):>9,}  "
          f"{row['pct_zero']:>5.1f}%  {bar}")

fig, ax = plt.subplots(figsize=(10, 5))
sorted_targets = sdf.index.tolist()
sorted_pct     = sdf["pct_zero"].values
bars = ax.barh(sorted_targets, sorted_pct, color=COLORS[:len(sorted_targets)])
ax.set_xlabel("% of records where target = 0")
ax.set_title("Target Sparsity — % Zero Values per Work Step")
ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50% threshold")
ax.legend()
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
for bar, pct in zip(bars, sorted_pct):
    ax.text(min(pct + 0.5, 99), bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=8)
plt.tight_layout()
save_fig("01_target_sparsity")

# ── 2b. Distribution of non-zero values ───────────────────────────────────────
subsection("2b. Non-Zero Target Distributions")

nonzero_stats = {}
for t in OUTPUT_TARGETS:
    vals = df_targets.loc[df_targets[t] > 0, t]
    if len(vals) < 2:
        continue
    nonzero_stats[t] = {
        "count":  len(vals),
        "mean":   vals.mean(),
        "median": vals.median(),
        "std":    vals.std(),
        "max":    vals.max(),
        "skew":   vals.skew(),
        "kurt":   vals.kurt(),
    }

st_df = pd.DataFrame(nonzero_stats).T.round(3)
print(f"\n{st_df.to_string()}")

# KDE / histogram grid for non-zero targets with enough data
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, t in enumerate(OUTPUT_TARGETS):
    ax  = axes[i]
    vals = df_targets.loc[df_targets[t] > 0, t]
    if len(vals) < 2:
        ax.text(0.5, 0.5, "Too sparse\n(< 2 records)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(t, fontsize=9)
        continue
    sns.histplot(vals, ax=ax, bins=30, kde=True, color=COLORS[i], log_scale=(True, False))
    ax.set_title(f"{t}\n(n={len(vals):,}, med={vals.median():.2f})", fontsize=8)
    ax.set_xlabel("Hours (est.)", fontsize=7)
for j in range(len(OUTPUT_TARGETS), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Non-Zero Target Distributions (log x-scale)", y=1.01, fontsize=12)
plt.tight_layout()
save_fig("02_target_distributions")

# ── 2c. Target co-occurrence heatmap ─────────────────────────────────────────
subsection("2c. Target Co-Occurrence")

binary = (df_targets > 0).astype(int)
cooc   = binary.T.dot(binary)
cooc_arr = cooc.values.copy()  # .copy() needed — pandas returns read-only view
np.fill_diagonal(cooc_arr, 0)
cooc = pd.DataFrame(cooc_arr, index=cooc.index, columns=cooc.columns)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(cooc, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(cooc, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, mask=mask, cbar_kws={"label": "# Orders (both active)"})
ax.set_title("Target Co-Occurrence — How Often Two Work Steps Appear Together")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
save_fig("03_target_cooccurrence")

# ── 2d. Active targets per order ─────────────────────────────────────────────
subsection("2d. Repair Complexity — Active Targets per Order")

n_active = binary.sum(axis=1)
print(f"\n  Active work steps per order:")
print(f"    Mean   : {n_active.mean():.2f}")
print(f"    Median : {n_active.median():.1f}")
print(f"    Max    : {n_active.max()}")
print(f"    Min    : {n_active.min()}")
print(f"\n  Distribution of active step counts:")
print(n_active.value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(n_active, bins=range(0, int(n_active.max()) + 2), ax=ax,
             color=COLORS[0], discrete=True)
ax.set_xlabel("Number of Active Work Steps per Order")
ax.set_ylabel("Count of Orders")
ax.set_title("Repair Complexity — Distribution of Active Work Step Count")
plt.tight_layout()
save_fig("04_active_targets_per_order")

# Total predicted time
total_time = df_targets.sum(axis=1)
subsection("2e. Total Predicted Time per Order")
print(f"\n  Total predicted time (sum of all targets) per order:")
print(f"    Mean   : {total_time.mean():.2f} hrs")
print(f"    Median : {total_time.median():.2f} hrs")
print(f"    Max    : {total_time.max():.2f} hrs")
print(f"    Min    : {total_time.min():.2f} hrs")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(total_time[total_time > 0], bins=50, kde=True, ax=ax, color=COLORS[1])
ax.set_xlabel("Total Predicted Hours per Order")
ax.set_title("Distribution of Total Predicted Repair Time")
plt.tight_layout()
save_fig("05_total_predicted_time")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — INPUT FIELD QUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Input Field Quality Audit")

# Flatten all positions into one dataframe
pos_rows = []
for idx, r in enumerate(valid):
    make = r["input"].get("make", None)
    for p in r["input"]["calculatedPositions"]:
        pos_rows.append({
            "order_idx":         idx,
            "make":              make,
            "text":              p.get("text", ""),
            "totalPrice":        p.get("totalPrice", 0) or 0,
            "totalTime":         p.get("totalTime",  0) or 0,
            "genericCostCenter": p.get("genericCostCenter", None),
        })

df_pos = pd.DataFrame(pos_rows)
print(f"\n  Total positions (all orders combined): {len(df_pos):,}")

# ── 3a. Make ─────────────────────────────────────────────────────────────────
subsection("3a. Make / Brand Distribution")
makes = pd.Series([r["input"].get("make", None) for r in valid])
make_counts = makes.value_counts(dropna=False)
print(f"\n  Top makes:\n{make_counts.head(20).to_string()}")
print(f"\n  Unknown / null makes : {(makes.isin(['unknown', 'Unknown', None])).sum()}")

fig, ax = plt.subplots(figsize=(10, 5))
top_makes = make_counts.head(20)
sns.barplot(x=top_makes.values, y=top_makes.index, ax=ax, palette="muted")
ax.set_xlabel("Order Count")
ax.set_title("Top 20 Vehicle Makes")
plt.tight_layout()
save_fig("06_make_distribution")

# ── 3b. Positions per order ───────────────────────────────────────────────────
subsection("3b. Positions per Order")
pos_per_order = df_pos.groupby("order_idx").size()
print(f"\n  Positions per order:")
print(f"    Mean   : {pos_per_order.mean():.1f}")
print(f"    Median : {pos_per_order.median():.0f}")
print(f"    Min    : {pos_per_order.min()}")
print(f"    Max    : {pos_per_order.max()}")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(pos_per_order, bins=40, kde=True, ax=ax, color=COLORS[2])
ax.set_xlabel("Positions per Order")
ax.set_title("Distribution of Line-Item Count per Repair Order")
plt.tight_layout()
save_fig("07_positions_per_order")

# ── 3c. Position type breakdown ───────────────────────────────────────────────
subsection("3c. Position Type Breakdown")
n_empty_text  = (df_pos["text"].str.strip() == "").sum()
n_header_row  = ((df_pos["totalPrice"] == 0) & (df_pos["totalTime"] == 0)
                 & (df_pos["text"].str.strip() != "")).sum()
n_useful      = ((df_pos["totalPrice"] > 0) | (df_pos["totalTime"] > 0)).sum()
n_null_cc     = df_pos["genericCostCenter"].isna().sum()

print(f"\n  Position breakdown:")
print(f"    Empty text rows              : {n_empty_text:,}  ({100*n_empty_text/len(df_pos):.1f}%)")
print(f"    Header/separator rows (0,0) : {n_header_row:,}  ({100*n_header_row/len(df_pos):.1f}%)")
print(f"    Rows with price or time > 0 : {n_useful:,}  ({100*n_useful/len(df_pos):.1f}%)")
print(f"    Null genericCostCenter       : {n_null_cc:,}  ({100*n_null_cc/len(df_pos):.1f}%)")

# ── 3d. Cost center distribution ─────────────────────────────────────────────
subsection("3d. Cost Center Distribution")
cc_counts = df_pos["genericCostCenter"].value_counts(dropna=False)
print(f"\n{cc_counts.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=cc_counts.values, y=cc_counts.index.astype(str),
            ax=axes[0], palette="muted")
axes[0].set_title("Position Count by Cost Center")
axes[0].set_xlabel("Count")

# Time by cost center (non-zero only)
cc_time = df_pos[df_pos["totalTime"] > 0].groupby("genericCostCenter")["totalTime"].sum()
sns.barplot(x=cc_time.values, y=cc_time.index.astype(str), ax=axes[1], palette="muted")
axes[1].set_title("Total Input Time by Cost Center")
axes[1].set_xlabel("Sum totalTime (non-zero positions)")
plt.tight_layout()
save_fig("08_cost_center_distribution")

# ── 3e. Text quality ─────────────────────────────────────────────────────────
subsection("3e. Text Quality")

useful_text = df_pos.loc[
    (df_pos["text"].str.strip() != "") &
    ((df_pos["totalPrice"] > 0) | (df_pos["totalTime"] > 0)),
    "text"
]

# Umlaut check
umlaut_pattern = re.compile(r"[äöüÄÖÜß]")
ae_pattern     = re.compile(r"\b(AE|OE|UE|SS)\b")
n_umlaut = useful_text.apply(lambda t: bool(umlaut_pattern.search(t))).sum()
n_ae     = useful_text.apply(lambda t: bool(ae_pattern.search(t))).sum()
print(f"\n  Text encoding check (useful positions only, n={len(useful_text):,}):")
print(f"    Records with real Umlauts (ä/ö/ü/ß): {n_umlaut:,}  ({100*n_umlaut/max(len(useful_text),1):.1f}%)")
print(f"    Records with AE/OE/UE/SS substitute : {n_ae:,}  ({100*n_ae/max(len(useful_text),1):.1f}%)")

char_lengths = useful_text.str.len()
print(f"\n  Text length (characters, non-empty useful positions):")
print(f"    Mean   : {char_lengths.mean():.1f}")
print(f"    Median : {char_lengths.median():.1f}")
print(f"    Max    : {char_lengths.max()}")

# Top tokens
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\+]", " ", text)
    return text.split()

all_tokens   = []
for t in useful_text:
    all_tokens.extend(tokenize(t))
token_counts = Counter(all_tokens)
print(f"\n  Total unique tokens: {len(token_counts):,}")
print(f"\n  Top 30 tokens:")
for tok, cnt in token_counts.most_common(30):
    print(f"    {tok:<30} {cnt:>6,}")

fig, ax = plt.subplots(figsize=(10, 6))
top_toks = pd.Series(dict(token_counts.most_common(20)))
sns.barplot(x=top_toks.values, y=top_toks.index, ax=ax, palette="muted")
ax.set_title("Top 20 Most Frequent Tokens in Position Descriptions")
ax.set_xlabel("Frequency")
plt.tight_layout()
save_fig("09_top_tokens")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — NUMERIC CONSISTENCY & UNIT VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Numeric Consistency & Unit Verification")

subsection("4a. Price Distribution (non-zero positions)")
nonzero_price = df_pos.loc[df_pos["totalPrice"] > 0, "totalPrice"]
print(f"\n  totalPrice (non-zero, n={len(nonzero_price):,}):")
print(f"    Mean   : {nonzero_price.mean():.2f}")
print(f"    Median : {nonzero_price.median():.2f}")
print(f"    Max    : {nonzero_price.max():.2f}")
print(f"    Min    : {nonzero_price.min():.4f}")
neg_price = (df_pos["totalPrice"] < 0).sum()
print(f"    Negative prices : {neg_price}")

subsection("4b. Input Time Distribution (non-zero positions)")
nonzero_time = df_pos.loc[df_pos["totalTime"] > 0, "totalTime"]
print(f"\n  totalTime (non-zero, n={len(nonzero_time):,}):")
print(f"    Mean   : {nonzero_time.mean():.2f}")
print(f"    Median : {nonzero_time.median():.2f}")
print(f"    Max    : {nonzero_time.max():.2f}")
print(f"    Min    : {nonzero_time.min():.4f}")
print(f"\n  NOTE: typical AW=5 minutes → divide by 12 to get hours")
print(f"        typical AW=6 minutes → divide by 10 to get hours")
print(f"        If values like 300 = 5h → 1AW = 1 minute → divide by 60")

subsection("4c. Implied Unit Conversion: Input Time vs Output Target")

# For each matching cost-center → target pair, compute per-order ratios
# and look for a consistent conversion factor
conversion_data = {}
for cc, targets in CC_TO_TARGET.items():
    if not targets:
        continue
    for t in targets:
        in_time = (df_pos[df_pos["genericCostCenter"] == cc]
                   .groupby("order_idx")["totalTime"].sum())
        out_time = df_targets[t]
        # Merge on order index
        merged = pd.DataFrame({"input": in_time, "output": out_time}).dropna()
        merged = merged[(merged["input"] > 0) & (merged["output"] > 0)]
        if len(merged) < 10:
            continue
        ratio = merged["output"] / merged["input"]
        conversion_data[f"{cc}→{t}"] = ratio
        med_ratio = ratio.median()
        print(f"  {cc} → {t}: n={len(merged):4d}  "
              f"median_ratio={med_ratio:.4f}  "
              f"(if input=min → output_unit factor≈{1/med_ratio:.1f} min/hr)")

# Plot price vs time scatter per cost center
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cc_list = [cc for cc in ["bodywork","painting","paintmaterial","material","others"]
           if cc in df_pos["genericCostCenter"].values]
axes = axes.flatten()
for i, cc in enumerate(cc_list[:6]):
    sub = df_pos[(df_pos["genericCostCenter"] == cc) &
                 (df_pos["totalPrice"] > 0) & (df_pos["totalTime"] > 0)]
    ax = axes[i]
    ax.scatter(sub["totalTime"], sub["totalPrice"], alpha=0.3, s=10,
               color=COLORS[i])
    ax.set_xlabel("totalTime (input)")
    ax.set_ylabel("totalPrice (EUR)")
    ax.set_title(f"Cost Center: {cc}\n(n={len(sub):,})")
    if len(sub) > 5:
        slope, intercept, r, p, _ = stats.linregress(sub["totalTime"], sub["totalPrice"])
        x_line = np.linspace(sub["totalTime"].min(), sub["totalTime"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1)
        ax.set_title(f"Cost Center: {cc}\n(n={len(sub):,}, r={r:.2f})")
for j in range(len(cc_list), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Input: Price vs Time by Cost Center", y=1.01, fontsize=12)
plt.tight_layout()
save_fig("10_price_vs_time_by_cc")

# Scatter input aggregate vs output target for painting (clearest semantic match)
subsection("4d. Input Painting Time vs Output Painting Targets")
paint_in = (df_pos[df_pos["genericCostCenter"] == "painting"]
            .groupby("order_idx")["totalTime"].sum())
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, t in enumerate(["paintingPreparation", "paintingSpraying", "paintingFinish"]):
    merged = pd.DataFrame({"input_paintTime": paint_in,
                           "output": df_targets[t]}).dropna()
    merged = merged[merged["output"] > 0]
    ax = axes[i]
    ax.scatter(merged["input_paintTime"], merged["output"],
               alpha=0.4, s=12, color=COLORS[i+9])
    ax.set_xlabel("Sum input painting totalTime")
    ax.set_ylabel(f"Output: {t} (hrs)")
    ax.set_title(f"Painting input → {t}")
    if len(merged) > 5:
        slope, _, r, _, _ = stats.linregress(merged["input_paintTime"],
                                              merged["output"])
        ax.set_title(f"{t}\n(n={len(merged)}, r={r:.2f})")
plt.suptitle("Input Painting Time vs Painting Output Targets", y=1.02)
plt.tight_layout()
save_fig("11_painting_input_vs_output")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENGINEERING CANDIDATE ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Feature Engineering Candidate Assessment")

# Build order-level feature matrix
order_feats = []
for idx, r in enumerate(valid):
    pos  = r["input"]["calculatedPositions"]
    make = r["input"].get("make", "unknown") or "unknown"

    df_p = pd.DataFrame(pos).fillna({"totalPrice": 0, "totalTime": 0, "text": ""})
    df_p["totalPrice"] = pd.to_numeric(df_p["totalPrice"], errors="coerce").fillna(0)
    df_p["totalTime"]  = pd.to_numeric(df_p["totalTime"],  errors="coerce").fillna(0)

    useful = df_p[(df_p["totalPrice"] > 0) | (df_p["totalTime"] > 0)]

    feat = {
        "order_idx":         idx,
        "make":              make,
        "n_positions":       len(df_p),
        "n_useful":          len(useful),
        "n_zero_time":       (df_p["totalTime"] == 0).sum(),
        "n_zero_price":      (df_p["totalPrice"] == 0).sum(),
        "total_time_input":  df_p["totalTime"].sum(),
        "total_price_input": df_p["totalPrice"].sum(),
        "max_time_pos":      df_p["totalTime"].max(),
        "mean_time_pos":     df_p["totalTime"].mean(),
        "max_price_pos":     df_p["totalPrice"].max(),
        "mean_price_pos":    df_p["totalPrice"].mean(),
    }
    # Per cost-center sums
    for cc in ["bodywork", "painting", "paintmaterial", "material", "others"]:
        sub = df_p[df_p.get("genericCostCenter", pd.Series()) == cc] \
              if "genericCostCenter" in df_p.columns else pd.DataFrame()
        # Safer approach:
        sub = df_p[df_p["genericCostCenter"].fillna("") == cc]
        feat[f"time_{cc}"]  = sub["totalTime"].sum()
        feat[f"price_{cc}"] = sub["totalPrice"].sum()
        feat[f"n_{cc}"]     = len(sub)

    # Painting ratio
    t_body  = feat.get("time_bodywork",  0)
    t_paint = feat.get("time_painting",  0)
    total_t = feat["total_time_input"]
    feat["ratio_painting_time"]  = t_paint / total_t if total_t > 0 else 0
    feat["ratio_bodywork_time"]  = t_body  / total_t if total_t > 0 else 0

    order_feats.append(feat)

df_feats = pd.DataFrame(order_feats).set_index("order_idx")
numeric_feats = [c for c in df_feats.columns if c != "make"]
df_feats_num  = df_feats[numeric_feats].astype(float)

# Correlation of each input feature with each output target
subsection("5a. Input Feature × Output Target Correlation")
corr_list = []
for feat_col in numeric_feats:
    for t in OUTPUT_TARGETS:
        mask = df_targets[t] > 0  # on non-zero only
        if mask.sum() < 10:
            continue
        r, p = stats.spearmanr(df_feats_num.loc[mask.values, feat_col],
                                df_targets.loc[mask.values, t])
        corr_list.append({"feature": feat_col, "target": t, "spearman_r": r, "p": p})

corr_df  = pd.DataFrame(corr_list)
pivot    = corr_df.pivot(index="feature", columns="target", values="spearman_r")

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Spearman r"})
ax.set_title("Spearman Correlation: Aggregated Input Features vs Output Targets\n(on non-zero target subset)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
save_fig("12_feature_target_correlation")

# Make vs target box plots
subsection("5b. Make vs Key Targets")
df_full = df_feats[["make"]].copy()
df_full = df_full.join(df_targets)
top_makes_list = makes.value_counts().head(8).index.tolist()
df_top_make    = df_full[df_full["make"].isin(top_makes_list)]

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
key_targets = ["bodymeasurement", "assembly", "paintingPreparation",
               "paintingSpraying", "glas", "calibration", "cleaning", "dismounting"]
for i, t in enumerate(key_targets):
    sub = df_top_make[df_top_make[t] > 0]
    if len(sub) < 5:
        axes[i].set_visible(False)
        continue
    sns.boxplot(data=sub, x="make", y=t, ax=axes[i],
                palette="muted", order=top_makes_list, showfliers=False)
    axes[i].set_title(f"{t}", fontsize=9)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)
plt.suptitle("Work Step Duration by Vehicle Make (non-zero records, no outliers)",
             y=1.01, fontsize=11)
plt.tight_layout()
save_fig("13_make_vs_targets")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RULE-BASED BASELINE
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Rule-Based Baseline")

def extract_combined_text(record):
    """Concatenate all meaningful position texts for one order."""
    texts = []
    for p in record["input"]["calculatedPositions"]:
        t = (p.get("text") or "").strip()
        if t:
            texts.append(t)
    return " ".join(texts).lower()

# Compile patterns
compiled_rules = {t: re.compile(pat, re.IGNORECASE)
                  for t, pat in KEYWORD_RULES.items()}

def rule_predict(record, input_time_conversion=10.0):
    """
    Simple rule-based predictor.
    Returns dict of {target: estimated_hours}.
    Estimates time = sum(matching position totalTime) / conversion_factor.
    """
    text      = extract_combined_text(record)
    all_time  = sum(p.get("totalTime", 0) or 0
                    for p in record["input"]["calculatedPositions"])
    pos       = record["input"]["calculatedPositions"]

    pred = {}
    for target, pattern in compiled_rules.items():
        if pattern.search(text):
            # Sum time from positions whose text matches
            matched_time = sum(
                (p.get("totalTime", 0) or 0)
                for p in pos
                if pattern.search((p.get("text") or "").lower())
            )
            pred[target] = matched_time / input_time_conversion if matched_time > 0 \
                           else all_time / input_time_conversion / len(compiled_rules)
        else:
            pred[target] = 0.0
    return pred

# Evaluate baseline — scan several conversion factors
subsection("6a. Conversion Factor Grid Search")
best_factor  = None
best_mae     = np.inf
factor_results = []

for factor in [6, 10, 12, 60, 1]:
    y_preds, y_trues = [], []
    occ_preds, occ_trues = [], []
    for i, r in enumerate(valid):
        p = rule_predict(r, input_time_conversion=factor)
        for t in OUTPUT_TARGETS:
            y_preds.append(p.get(t, 0))
            y_trues.append(df_targets.loc[i, t])
            occ_preds.append(1 if p.get(t, 0) > 0 else 0)
            occ_trues.append(1 if df_targets.loc[i, t] > 0 else 0)
    mae  = np.mean(np.abs(np.array(y_preds) - np.array(y_trues)))
    rmse = np.sqrt(np.mean((np.array(y_preds) - np.array(y_trues))**2))
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1   = f1_score(occ_trues, occ_preds, zero_division=0)
    prec = precision_score(occ_trues, occ_preds, zero_division=0)
    rec  = recall_score(occ_trues, occ_preds, zero_division=0)
    factor_results.append({
        "conversion_factor": factor,
        "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "F1_occurrence": round(f1, 4),
        "Precision":     round(prec, 4),
        "Recall":        round(rec, 4),
    })
    if mae < best_mae:
        best_mae    = mae
        best_factor = factor

fdf = pd.DataFrame(factor_results)
print(f"\n  Grid search over conversion factors:\n")
print(fdf.to_string(index=False))
print(f"\n  ★ Best conversion factor (lowest MAE): {best_factor}")

# Per-target baseline scores with best factor
subsection(f"6b. Per-Target Baseline Scores (factor={best_factor})")
target_scores = {}
for t in OUTPUT_TARGETS:
    y_pred = [rule_predict(r, best_factor).get(t, 0) for r in valid]
    y_true = df_targets[t].values
    occ_pred = [1 if v > 0 else 0 for v in y_pred]
    occ_true = [1 if v > 0 else 0 for v in y_true]
    from sklearn.metrics import f1_score
    mae = np.mean(np.abs(np.array(y_pred) - y_true))
    f1  = f1_score(occ_true, occ_pred, zero_division=0)
    target_scores[t] = {"MAE": round(mae, 4), "F1_occurrence": round(f1, 4)}

tdf = pd.DataFrame(target_scores).T.sort_values("F1_occurrence", ascending=False)
print(f"\n{tdf.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tdf_sorted_f1  = tdf.sort_values("F1_occurrence", ascending=True)
tdf_sorted_mae = tdf.sort_values("MAE", ascending=False)
sns.barplot(x="F1_occurrence", y=tdf_sorted_f1.index, data=tdf_sorted_f1,
            ax=axes[0], palette="Blues_r")
axes[0].set_title(f"Rule Baseline — F1 Occurrence (factor={best_factor})")
axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1)
sns.barplot(x="MAE", y=tdf_sorted_mae.index, data=tdf_sorted_mae,
            ax=axes[1], palette="Reds_r")
axes[1].set_title(f"Rule Baseline — MAE (factor={best_factor})")
plt.tight_layout()
save_fig("14_baseline_scores")

print(f"\n  ★ Rule baseline establishes floor — ML must beat these scores.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SUMMARY & OPEN QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — Summary & Open Questions")

print(f"""
  +-------------------------------------------------------------------+
  |  DATASET SUMMARY                                                |
  +-------------------------------------------------------------------+
  |  Records          : {n_records:>8,}                                  |
  |  Valid records    : {len(valid):>8,}                                  |
  |  Output targets   : 14 (multi-target regression + classification)|
  |  Avg positions/order      : {pos_per_order.mean():>5.1f}                         |
  |  Avg active targets/order : {n_active.mean():>5.2f}                         |
  |  Most sparse target  : {sdf.index[0]:<20} ({sdf['pct_zero'].iloc[0]:.1f}% zero)   |
  |  Least sparse target : {sdf.index[-1]:<20} ({sdf['pct_zero'].iloc[-1]:.1f}% zero)   |
  +-------------------------------------------------------------------+

  KEY FINDINGS:
  ------------
  1. All 14 targets are sparse — two-stage model (classify → regress) is justified.
  2. co-occurrence analysis reveals natural repair clusters (painting / bodywork / glass).
  3. Text is German technical abbreviations — TF-IDF with char n-grams recommended.
  4. Rule baseline shows keyword coverage — F1 scores reveal which targets are easiest.
  5. Unit conversion factor MUST be confirmed — grid search gives best empirical estimate.
  6. 'make' varies widely including 'unknown' — needs robust encoding strategy.

  OPEN QUESTIONS FOR STAKEHOLDERS:
  ---------------------------------
  1. Unit of input totalTime? (minutes / AW / hours)
  2. Unit of output targets? (hours assumed — confirm)
  3. Full dataset size? (affects TF-IDF vs. transformer embedding decision)
  4. Additional fields available? (spare parts, VIN, timestamps, workshop ID)
  5. Is 'make == unknown' fixable upstream, or must model handle it?
  6. Evaluation priority: occurrence F1 vs. duration MAE vs. total time error?
  7. Is output schema fixed (exactly 14 targets), or can new categories appear?

  All plots saved to: {PLOT_DIR}
""")
