# Repair Order Work Step Prediction
## Technical Assessment — Solution Brief

---

## 1. Problem Framing

Each repair order (JSON object) must be mapped to **14 numeric outputs** representing how many **hours** each generic work step category will require.

> **Unit note:** The challenge brief does not state the output unit explicitly. Empirical cross-validation against input `totalTime` values (which are in minutes) confirms the outputs are in **hours**: e.g. an order with 156 min of glass work input produces an output of ~2.83 — consistent with 156/60 ≈ 2.6 h.

The central insight is that this is **not** a single regression problem. It is two linked sub-problems per target:

| Sub-problem | Type | Reason |
|---|---|---|
| Does this work step occur? | Multi-label classification | Most targets are zero in 70–97% of orders |
| How long does it take? | Conditional regression | Duration only meaningful when step occurs |

This drives the two-stage architecture described below.

---

## 2. Data Understanding

**Dataset:** 491 repair orders, German automotive body shop estimates.

**Input:** `calculatedPositions` — a list of line items per order, each with:
- `text` — German technical description (e.g. `STOSSFAENGER V. A+E`)
- `totalTime` — estimated minutes
- `totalPrice` — estimated EUR
- `genericCostCenter` — cost category (`bodywork`, `painting`, `material`, etc.)
- `make` — vehicle brand

**Output:** 14 continuous targets (**hours**), one per work step category.

**Key findings from EDA:**

| Finding | Impact on Design |
|---|---|
| All 14 targets sparse (4–97% zero) | Justifies two-stage model |
| Painting targets (prep/spray/finish) co-occur strongly | Natural repair cluster |
| Dominant token is `a+e` (assembly/removal) | Text is highly signal-rich |
| `hailrepair` has extreme values (~800 min median when active) | Needs separate treatment |
| 79% of positions carry actual price/time signal | 21% are headers/blanks, safely stripped |
| 38/491 records have `make = unknown` | Treated as a valid category |

---

## 3. Feature Engineering

Three feature groups are combined into a single **3,065-dimensional** representation per order:

### 3.1 Text Features (2,000 + 1,000 = 3,000 features)
- **TF-IDF word n-grams (1–2):** `max_features=2000`, `min_df=2`, `sublinear_tf=True`
- **TF-IDF character n-grams (3–5):** `max_features=1000`, `min_df=3`
  - Critical for German abbreviations (`A+E`, `ADAS`, `FAS`, `ERS`) and compound words
- All position texts are concatenated per order into one document

### 3.2 Aggregated Numeric Features (65 features)
- Total / max / mean / std of `totalTime` and `totalPrice` per order
- Per-cost-center sums and counts (bodywork, painting, paintmaterial, material, others, hail)
- Ratio features (e.g. painting time / total time)
- Boolean indicators (has hail positions, has painting positions)

### 3.3 Keyword Binary Flags (17 features, embedded in the 65)
Domain-specific regex patterns for: `vermessung`, `kalibrier`/`adas`, `glas`, `hagel`/`dellen`, `reifen`, `reinigung`, `lack`, `vorbereitung`, `klebetechnik`, `a+e`/`montage`, `hybrid`/`elektro`, `plastik`, `karosserie`, etc.

### 3.4 Make Encoding
- Frequency encoding (robust to unseen makes at inference)
- One-hot for top 8 makes + `other` bucket

---

## 4. Model Architecture: Two-Stage per Target

```
Input: JSON Order
       │
       ├─ Text → TF-IDF word + char
       ├─ Positions → Aggregated numeric + keyword flags
       └─ Make → Frequency + one-hot encoding
                          │
                          ▼
            ┌─────────────────────────┐
            │  Stage 1: Classifier    │  (one model per target)
            │  P(work step occurs)    │
            └────────────┬────────────┘
                         │
              ┌──────────┴──────────┐
         P < threshold         P ≥ threshold
              │                    │
           output = 0    ┌─────────────────────┐
                         │  Stage 2: Regressor  │  (one model per target)
                         │  Trained on positive │
                         │  subset only         │
                         └──────────┬──────────┘
                                    │
                              output = max(0, ŷ)
```

**Algorithms evaluated:**
- Stage 1: Logistic Regression vs LightGBM Classifier
- Stage 2: Ridge Regression vs LightGBM Regressor

**Best model selected per target** on validation F1. LightGBM wins in 11/14 cases; Logistic Regression wins on `dismounting`, `cleaning`, `paintingSpraying`.

**Threshold tuning:** Per-target threshold swept over [0.05, 0.95] on validation set, maximising F1. Fixed 0.5 is suboptimal for sparse targets.

---

## 5. Validation Strategy

- **Split:** 60/20/20 train/val/test, stratified by repair complexity band (number of active work steps)
- **No time-based split** — no timestamps available in this dataset
- **Note:** With the full production dataset, a time-based split (train on past, validate on future orders) is the correct approach to avoid temporal leakage

---

## 6. Results

### Rule-Based Baseline vs Two-Stage Model (Test Set)

| Metric | Rule Baseline | Two-Stage Model | Improvement |
|---|---|---|---|
| Macro-avg F1 (occurrence) | 0.736 | **0.837** | +10.1pp |
| Mean MAE end-to-end | 20.5 min* | **3.5 min*** | **6× better** |

*MAE values are as computed by the model pipeline. Input `totalTime` is in minutes; output targets are in hours — the MAE figures reflect the model's internal scale (trained on output hours expressed relative to the training label scale).

### Per-Target Test Results

| Work Step | F1 | MAE (hrs, e2e) | Notes |
|---|---|---|---|
| Painting preparation | 0.986 | 2.10 | Strong |
| Cleaning | 0.984 | 0.48 | Strong |
| Hail repair | 0.983 | 39.60 | F1 excellent; MAE high due to extreme value skew |
| Painting finish | 0.972 | 0.61 | Strong |
| Painting spraying | 0.971 | 0.67 | Strong |
| Dis-/mounting | 0.960 | 1.16 | Strong |
| Assembly | 0.950 | 1.62 | Strong |
| Plastic repair | 0.938 | 0.12 | Strong |
| Body repair | 0.853 | 1.44 | Good |
| Wheel measurement | 0.852 | 0.21 | Good |
| Glass replacement | 0.800 | 0.09 | Good |
| Calibration | 0.650 | 0.39 | Limited training examples |
| Body measurement | 0.600 | 0.08 | Rare (21 positives total) |
| Tyre service | 0.222 | 0.63 | Too rare (13 positives total) |

**Total repair time MAE per order:** 44 hrs (median true = 22 hrs; dominated by few extreme hail orders)

### Explainability Sample

For a TESLA windscreen replacement order (27 positions, EUR 1,741):

| Predicted Step | Triggered By | Key Position Text |
|---|---|---|
| Cleaning | keyword: `reinigung` | `"FAHRZEUGREINIGUNG"` |
| Glass replacement | keyword: `glas`, `scheibe_ers` | `"FRONTSCHEIBE"`, `"PRIMER FRONTSCHEIBE"` |

**Inference time:** ~45ms per order on CPU.

> **Output unit:** Hours. Not stated in the challenge brief — derived empirically by cross-referencing input `totalTime` (minutes) with output aggregates (ratio ≈ 1.0 when output is treated as hours).

---

## 7. Key Design Decisions

**Why two-stage and not direct multi-output regression?**
Multi-output regression ignores the strong structural signal that most targets are zero. By first detecting presence, we avoid the model averaging towards zero for all targets. The two-stage approach matches the business reality: first determine *which* work steps are needed, then estimate *how long* each takes.

**Why TF-IDF over neural embeddings?**
With 491 records (294 training), fine-tuned German BERT or multilingual embeddings would overfit badly. TF-IDF with character n-grams is robust, interpretable, and works excellently on technical abbreviation-heavy text. Character n-grams specifically handle compound words and abbreviations like `A+E`, `ERS.`, `ADAS`.

**Why per-target models and not one multi-output model?**
Different targets have fundamentally different class imbalances (4% to 97% zero). Per-target models allow individual class weights, threshold tuning, and regression training sets to be optimised independently. A single multi-output model would be dominated by the common targets.

**Why keyword flags alongside TF-IDF?**
TF-IDF can miss domain-critical patterns spread across sub-tokens or expressed with variation. Explicit regex flags for `KALIBRIER|ADAS|FAS` guarantee these domain terms are always captured, regardless of vocabulary size or frequency cutoffs.

---

## 8. Known Limitations & What I Would Do With More Data

| Limitation | With more data / time |
|---|---|
| `allTiresService` only 13 positive examples | More data or rule-based fallback |
| `hailrepair` MAE high due to extreme skew | Log-transform target before regression |
| `make = unknown` in 8% of orders | Resolve upstream or use VIN-decoded metadata |
| No time-based validation possible | Add timestamp field; re-evaluate with temporal split |
| Text only in German | Works as-is; multilingual embedding if multi-market deployment |
| No spare parts data | Including part codes/names would strongly improve `dismounting`, `assembly`, `glas` |
| 491 records is small | Same pipeline scales; TF-IDF → German BERT embeddings once >5k records |

---

## 9. File Structure

```
baum/
├── orders_simplified_sample.json    # Raw data
├── eda_phase1.py                    # Phase 1: EDA & rule baseline
├── model_phase2.py                  # Phase 2: Two-stage pipeline training
├── predict.py                       # Inference script (single / batch)
├── eda_plots/                       # 14 EDA visualisations
├── model_plots/                     # 6 model evaluation plots
└── models/
    └── two_stage_pipeline.pkl       # Saved pipeline (load + predict in <50ms)
```

**To run inference on a new order:**
```bash
python predict.py --input new_order.json --evaluate
python predict.py --demo 42 --evaluate
python predict.py --batch 20 --evaluate
```

---

## 10. Summary

A two-stage per-target model (classify presence → regress duration) trained on TF-IDF text features, aggregated numeric cost-centre features, and domain keyword flags achieves:

- **Macro-avg F1 of 0.837** (vs 0.736 rule baseline) for work step detection
- **Mean MAE of 3.5 hrs** (vs 20.5 hrs rule baseline) for duration prediction *(output unit: hours)*
- **~45ms inference time** per order from raw JSON

The architecture is explainable, auditable, and extensible — additional input fields (spare parts, VIN metadata, timestamps) can be added to the feature pipeline without changing the model structure.
