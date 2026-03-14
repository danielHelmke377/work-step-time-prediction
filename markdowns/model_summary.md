# Repair Order Work Step Prediction — Model & Pipeline Summary

> **Audience:** Technical stakeholders, engineering peers, and assessors.  
> **Scope:** End-to-end ML pipeline for predicting which work steps occur in an automotive body shop repair order, and how long each takes (in **hours**).

> **Unit clarification:** The challenge brief does not explicitly state the output unit. Empirical cross-validation confirms the outputs are in **hours**: input `totalTime` values are in minutes, and dividing them by 60 aligns with the output magnitudes (ratio ≈ 1.0). For example, 156 min of glass work in the input → 2.83 hrs in the output.

---

## 1. Problem Framing

The task maps a structured repair order — a list of German-language line items (positions) with associated cost centres, prices, and times — to **14 continuous output targets**, each representing the predicted duration (**hours**) of a distinct work step category (e.g. painting preparation, hail repair, calibration).

The structural reality of the data drives the modelling approach: **all 14 targets are zero in 4–97% of orders** depending on the work step. Treating this as a direct multi-output regression problem would cause the model to systematically under-predict, pulled towards zero by the dominant zero class. The correct decomposition is:

| Sub-problem | Output type | Rationale |
|---|---|---|
| **Does this work step occur?** | Binary classification | Structural sparsity — most targets are absent |
| **How long does it take, given it occurs?** | Conditional regression | Duration only has meaning when the step is active |

This motivates a **two-stage per-target architecture**, described in Section 4.

---

## 2. Dataset

- **491 repair orders**, JSON format, from a German automotive body shop
- **Input:** `calculatedPositions` — a variable-length list of line items per order, each with:
  - `text`: German technical description (e.g. `STOSSFAENGER V. A+E`)
  - `totalTime`: estimated minutes per position
  - `totalPrice`: estimated EUR per position
  - `genericCostCenter`: cost category (`bodywork`, `painting`, `hail`, `material`, etc.)
  - `make`: vehicle manufacturer
- **Output:** 14 numeric targets (**hours**), one per work step category
- **Split:** 60 / 20 / 20 (train / val / test), stratified by **repair complexity band** (number of active work steps per order)

> **Note:** No timestamps are available. With a production dataset, a rolling time-based split (train on past orders, validate on future) is the correct evaluation protocol to prevent temporal leakage.

---

## 3. Feature Engineering

Three complementary feature groups are assembled per order into a single **3,065-dimensional** sparse matrix for model input.

### 3.1 TF-IDF Text Features (3,000 dimensions)

All position texts within an order are concatenated into a single document (lowercased, filtered to positions with actual price or time signal). Two TF-IDF vectorisers are fitted:

| Vectoriser | N-gram range | Max features | Analyser | Key role |
|---|---|---|---|---|
| Word TF-IDF | 1–2 grams | 2,000 | `word` | Captures lexical context; bigrams like `a+e montage` |
| Character TF-IDF | 3–5 grams | 1,000 | `char_wb` | Handles abbreviations, compound words, typos |

`sublinear_tf=True` is applied to compress the effect of high-frequency tokens.  
The character n-gram vectoriser is particularly important here: German automotive text is heavily abbreviation-driven (`A+E`, `ERS.`, `ADAS`, `FAS`), and character-level representation handles spelling variants and compound words that word-level tokenisation would split or miss entirely.

### 3.2 Aggregated Numeric Features (~48 dimensions)

Computed from the position list per order:
- **Count features:** total positions, useful positions (price or time > 0), zero-time and zero-price counts
- **Time/price aggregates:** sum, max, mean, median, std — for both `totalTime` and `totalPrice`
- **Per-cost-centre breakdowns:** sum and count of time/price for each of 6 cost centres (`bodywork`, `painting`, `paintmaterial`, `material`, `others`, `hail`)
- **Ratio features:** painting time as fraction of total time, hail time fraction, material price fraction
- **Boolean indicators:** binary flags for presence of hail, painting, or bodywork cost-centre positions

### 3.3 Domain Keyword Flags (17 dimensions)

Pre-compiled regex patterns capture domain-critical terms that may be underrepresented in the TF-IDF vocabulary due to low frequency or abbreviation variance:

```
kw_kalibrier  → kalibrier|adas|fas|kamera.*kalib|radar.*ausrich
kw_glas       → glas|scheibe|frontscheibe|windschutz|...
kw_hagel      → hagel|dellen|pdr|smart.*repar
kw_montage    → a\+e|montage|einbau|ausbau|demontage|ersatz
kw_lack       → lackier|lack(?!material)|oberflaech.*lack|...
... (17 total)
```

These flags are OR-ed into the numeric matrix and act as a hard, interpretable prior — ensuring high-signal terms always activate a feature regardless of their TF-IDF rank.

### 3.4 Make Encoding

- **Frequency encoding:** each vehicle maker is encoded as its training-set frequency count (robust to unseen makes at inference time)
- **One-hot encoding:** top 8 makes (VW, Mercedes-Benz, BMW, Ford, Skoda, Audi, Opel, Tesla) + `other` bucket

---

## 4. Model Architecture

```
Input: Raw JSON Order
       │
       ├─ Concatenated position texts  ──→  Word TF-IDF (2,000d)
       │                               ──→  Char TF-IDF (1,000d)
       ├─ Position list                ──→  Numeric aggregates + keyword flags (~65d)
       └─ make                         ──→  Frequency + one-hot encoding
                          │
                          ▼
              ┌──────────────────────────┐
              │  For each of 14 targets: │
              │                          │
              │  Stage 1: Classifier     │  (Logistic Regression OR LightGBM)
              │  P(work step occurs)     │  ← threshold-tuned on validation F1
              └────────────┬─────────────┘
                           │
                ┌──────────┴──────────┐
           P < τ_t                P ≥ τ_t
                │                    │
           output = 0   ┌──────────────────────────┐
                        │  Stage 2: Regressor       │
                        │  (Ridge OR LightGBM)      │
                        │  Trained on positive-only │
                        │  subset (y > 0 in train)  │
                        └────────────┬──────────────┘
                                     │
                               output = max(0, ŷ)
```

### Algorithm Selection & Threshold Tuning

Two candidate algorithms were evaluated at each stage:

- **Stage 1:** Logistic Regression (L2, `lbfgs`, class-weight-balanced) vs. LightGBM Classifier (`n_estimators=300`, `lr=0.05`, `num_leaves=31`, `scale_pos_weight` set per-target)
- **Stage 2:** Ridge Regression (α=10) vs. LightGBM Regressor (same hyperparams; `min_child_samples` scaled by positive subset size)

**Best model per target** is selected on validation F1 (Stage 1) and validation MAE (Stage 2) independently.  
LightGBM wins Stage 1 for 11/14 targets; Logistic Regression performs better on `dismounting`, `cleaning`, and `paintingSpraying` — where the signal is predominantly lexical and the linear decision boundary is sufficient.

**Per-target threshold tuning:** Classification thresholds are swept over [0.05, 0.95] in steps of 0.01 on the validation set, selecting the threshold that maximises F1 for each target. A fixed threshold of 0.5 is systematically suboptimal for sparse targets where the positive class probability is almost never near 0.5.

**Fallback strategy:** For targets with fewer than 5 positive training examples, the Stage 2 regressor falls back to predicting the training-set mean duration. This prevents degenerate fits on near-empty subsets.

---

## 5. Training Protocol

| Step | Detail |
|---|---|
| Data split | 60/20/20, stratified by complexity band (active target count) |
| Stage 1 training | Full training set (all records, binary labels) |
| Stage 2 training | Positive subset only — records where the target > 0 in training labels |
| Threshold selection | Validation set, F1-maximising sweep per target |
| Model selection | Validation F1 (classifiers) / Validation MAE on positive subset (regressors) |
| Test evaluation | Held-out 20% test set, end-to-end (Stage 1 → Stage 2 prediction chain) |

---

## 6. Evaluation Results

### 6.1 Overall Performance vs. Rule-Based Baseline (Test Set)

| Metric | Rule Baseline | **Two-Stage Model** | Improvement |
|---|---|---|---|
| Macro-avg F1 (occurrence) | 0.736 | **0.837** | +10.1 pp |
| Mean end-to-end MAE | 20.5 hrs | **3.5 hrs** | **6× lower** |

### 6.2 Per-Target Test Results

| Work Step | F1 | E2E MAE (hrs) | Notes |
|---|---|---|---|
| Painting preparation | 0.986 | 2.10 | Excellent |
| Cleaning | 0.984 | 0.48 | Excellent |
| Hail repair | 0.983 | 39.60 | F1 excellent; MAE high due to extreme skew (~70–70,000 min range) |
| Painting finish | 0.972 | 0.61 | Excellent |
| Painting spraying | 0.971 | 0.67 | Excellent |
| Dis-/mounting | 0.960 | 1.16 | Excellent |
| Assembly | 0.950 | 1.62 | Excellent |
| Plastic repair | 0.938 | 0.12 | Excellent |
| Body repair | 0.853 | 1.44 | Good |
| Wheel measurement | 0.852 | 0.21 | Good |
| Glass replacement | 0.800 | 0.09 | Good |
| Calibration | 0.650 | 0.39 | Limited training examples |
| Body measurement | 0.600 | 0.08 | Rare (21 positives total) |
| Tyre service | 0.222 | 0.63 | Too rare (13 positives); effectively unlearnable at this volume |

**Total repair time MAE per order:** ~44 hrs on median true total of ~22 hrs — the aggregate error is dominated by the few extreme hail repair outliers.

---

## 7. Explainability

The pipeline is designed to be auditable at inference time. For each active predicted work step, the inference engine identifies:

1. **Which domain keyword flags triggered** (e.g. `glas`, `scheibe_ers` for glass replacement)
2. **Which specific position texts matched** those regex patterns

**Example — TESLA windscreen replacement order (27 positions, EUR 1,741):**

| Predicted Step | Triggered Keywords | Matching Position Texts |
|---|---|---|
| Cleaning | `reinigung` | `"FAHRZEUGREINIGUNG"` |
| Glass replacement | `glas`, `scheibe_ers` | `"FRONTSCHEIBE"`, `"PRIMER FRONTSCHEIBE"` |

For the Stage 1 Logistic Regression models, feature coefficients over word TF-IDF features provide additional global interpretability — the top positive TF-IDF tokens per target align intuitively with domain knowledge (e.g. `lackier`, `grundier` → painting; `hagel`, `dellen` → hail repair).

---

## 8. Inference Architecture

The serialised pipeline (`two_stage_pipeline.pkl`, ~13 MB) packages:

| Component | Description |
|---|---|
| `tfidf_word` | Fitted `TfidfVectorizer` (word, 2,000 features) |
| `tfidf_char` | Fitted `TfidfVectorizer` (char_wb, 1,000 features) |
| `clf_models` | Dict of fitted classifiers per target (`logreg` + `lgbm`) |
| `reg_models` | Dict of fitted regressors per target (`ridge` + `lgbm`) |
| `thresholds` | Per-target, per-model-type tuned thresholds |
| `best_clf_per_target` | Maps each target to the best classifier type |
| `numeric_features` | Ordered list of numeric feature names (for alignment at inference) |
| `output_targets` | Canonical target ordering |

**Inference flow for a new order:**
1. Parse and clean `calculatedPositions` (strip null/empty rows)
2. Concatenate position texts → transform via both TF-IDF vectorisers
3. Compute numeric aggregates + keyword flags → align to training feature order
4. Horizontally stack sparse matrices → single feature vector
5. For each target: Stage 1 predict proba → apply threshold → if active, Stage 2 predict duration → clip to 0
6. Return structured prediction dict with probabilities, durations, active flags, and elapsed time

**Inference latency: ~45 ms per order on CPU** (single-threaded, no batching optimisation).

---

## 9. Known Limitations

| Limitation | Root cause | Mitigation / roadmap |
|---|---|---|
| `allTiresService` unlearnable (F1=0.22) | Only 13 positive examples in full dataset | Rule-based fallback; collect more data |
| `hailrepair` MAE high | Extreme duration skew (100–4,000 min) | Log-transform target; Tweedie/log-link regressor |
| No temporal validation | No timestamps in dataset | Add timestamp field; re-evaluate with rolling-window split |
| No hyperparameter optimisation | Fixed defaults throughout | Cross-validated Bayesian search (Optuna) for hard targets |
| TF-IDF is bag-of-words | Misses long-range positional context | Replace with German domain-adapted BERT once >5k records |
| Spare parts data absent | Not in current dataset | Part codes/descriptions would strongly improve `dismounting`, `assembly`, `glas` |
| No probability calibration | Predicted probabilities may be biased | Apply Platt scaling / isotonic regression for calibrated confidence scores |

---

## 10. Production Readiness Checklist

- [x] Pipeline serialised with `pickle` — load + predict in <50ms
- [x] Feature engineering mirrored exactly in `predict.py` (no training/serving skew)
- [x] Graceful fallback for rare targets (mean predictor)
- [x] `make = unknown` handled as a valid frequency-encoded category
- [x] Keyword-based explanation layer for auditability
- [ ] Model monitoring (prediction vs. actual drift detection)
- [ ] Calibrated probability outputs
- [ ] Champion/challenger deployment framework
- [ ] Time-based evaluation on production data

---

## 11. Summary

A **two-stage per-target pipeline** — classify occurrence, then regress duration on positive-only subsets — achieves:

- **Macro-avg F1 of 0.837** (vs 0.736 rule baseline) for work step detection
- **Mean MAE of 3.5 hrs** (vs 20.5 hrs rule baseline) for duration prediction *(output unit: hours)*
- **~45ms inference time** per order on CPU

The architecture is explainable, auditable, and extensible — additional input fields (spare parts, VIN metadata, timestamps) can be added to the feature pipeline without changing the model structure. It is production-deployable today and designed to scale naturally as data volume and geographic scope grow.
