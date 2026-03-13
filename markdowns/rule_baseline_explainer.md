# Rule-Based Baseline — How It Works

The rule baseline is a **"no-machine-learning" reference model** built entirely from domain knowledge and regular expressions. It serves as a benchmark to measure how much the ML model actually adds.

---

## Step 1 — Detect Whether a Work Step Occurs (Classification)

For each repair order, all position descriptions (`calculatedPositions[].text`) are concatenated into one text block and checked against hand-crafted keyword patterns:

| Pattern | Predicted Work Step |
|---|---|
| `KALIBRIER`, `ADAS`, `FAS`, `RADAR` | `calibration` |
| `VERMESS`, `KINEMATIK`, `FAHRWERK` | `wheelmeasurement` / `bodymeasurement` |
| `GLAS`, `SCHEIBE`, `FRONTSCHEIBE` | `glas` |
| `REINIGUNG`, `WAESCH` | `cleaning` |
| `HAGEL`, `DELLEN`, `PDR` | `hailrepair` |
| `LACKIER`, `VORBEREITUNG` | `paintingPreparation` / `paintingSpraying` |
| `REIFEN`, `FELGE` | `allTiresService` |
| `A+E`, `MONTAGE`, `EINBAU` | `dismounting` / `assembly` |
| `PLASTIK`, `KUNSTSTOFF` | `plasticrepair` |

**If a pattern matches → work step predicted as present (1).**  
**If no match → work step predicted as absent (0).**

---

## Step 2 — Estimate Duration (Regression)

For each matched target, the rule sums the `totalTime` of all positions whose text triggered the keyword match, then divides by a **conversion factor**:

```
predicted_minutes = sum(matching position totalTime) / conversion_factor
```

Five conversion factors (1, 6, 10, 12, 60) were tested. **Factor = 60 produced the lowest MAE**, consistent with input times being in a unit roughly 60× the output unit.

---

## Results

| Metric | Value |
|---|---|
| Overall F1 (occurrence) | 0.736 |
| Overall MAE (end-to-end) | 20.5 min |
| Conversion factor used | 60 |

### Per-target F1 (best → worst)

| Target | F1 | Why |
|---|---|---|
| `hailrepair` | 0.983 | `HAGEL` / `DELLEN` are unmistakable |
| `paintingSpraying` | 0.983 | `LACKIER` is very direct |
| `cleaning` | 0.976 | `FAHRZEUGREINIGUNG` always present |
| `dismounting` | 0.945 | `A+E` is pervasive |
| `plasticrepair` | 0.769 | `KUNSTSTOFF` / `STOSSFAENGER` fairly reliable |
| `bodyrepair` | 0.766 | `KAROSSERIE` present but overlaps with other work |
| `paintingPreparation` | 0.727 | `VORBEREITUNG` is common but context-dependent |
| `wheelmeasurement` | 0.509 | `VERMESS` can refer to body or wheel measurement |
| `calibration` | 0.486 | `KALIBRIER` / `FAS` sometimes mixed with other scopes |
| `assembly` | 0.471 | `A+E` fires for both `dismounting` and `assembly` — can't distinguish |
| `paintingFinish` | 0.365 | No single keyword; implied by context |
| `bodymeasurement` | 0.308 | Specific measurement term rarely appears separately |
| `glas` | 0.297 | `SCHEIBE` appears in hail orders too, causing false positives |
| `allTiresService` | 0.247 | Tyre work rarely stated directly in position text |

---

## Why Some Targets Are Hard for the Rule Baseline

| Problem | Affected Targets | Explanation |
|---|---|---|
| **Keyword ambiguity** | `assembly` vs `dismounting` | Both triggered by `A+E` — rule cannot split them |
| **Context dependency** | `paintingFinish` | Finishing is implied, not stated explicitly |
| **Keyword overlap** | `glas` vs `hailrepair` | `SCHEIBE` appears in hail orders describing dented panels near windows |
| **No keyword at all** | `allTiresService` | Tyre work is described in specialist sub-fields not covered by simple patterns |
| **Compound meaning** | `wheelmeasurement` vs `bodymeasurement` | `VERMESSUNG` covers both |

---

## Why We Build It at All

1. **Anchor for ML evaluation** — without a baseline, a test F1 of 0.84 is meaningless. With the baseline at 0.74 it shows +10pp improvement.
2. **Quantifies text signal quality** — high baseline F1 means the text is unambiguous for that target; low F1 means ML must learn subtler patterns.
3. **Business transparency** — explainable to a non-technical stakeholder in one sentence.
4. **Zero training cost** — works immediately on any new order with no model fitting.

---

## What the ML Model Adds

The two-stage ML model replaces the keyword-only approach with:
- **TF-IDF features** — learns which *combination* of terms predicts each target
- **Numeric aggregates** — cost-centre time/price distributions, ratios
- **Per-target thresholds** — tuned on validation data, not fixed at 0.5
- **Conditional regression** — separate model per target trained only on positive examples

Result: F1 rises from **0.736 → 0.837** and MAE drops from **20.5 → 3.5 min**.
