# Champion-Challenger Promotion Report

**Date:** 2026-03-21 09:42:47  
**Decision:** **PROMOTED**  
**Mode:** LIVE RUN

---

## Model versions

| Slot | Version | Trained on |
|---|---|---|
| Champion | 2.0.0 | 2026-03-21 |
| Challenger | 2.0.0 | 2026-03-21 |

---

## Metric comparison

| Metric | Champion | Challenger | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.2442 | 0.2442 | = +0.0000 |
| Macro MAE (hrs) | 1.3900 | 1.3900 | = +0.0000 |
| Freq-weighted F1 | 0.3332 | 0.3332 | = +0.0000 |
| Freq-weighted MAE (hrs) | 1.5500 | 1.5500 | = +0.0000 |
| Freq-weighted Accuracy | 0.7127 | 0.7127 | = +0.0000 |

---

## Promotion rule evaluation

| Rule | Description | Result | Detail |
|---|---|---|---|
| `R1 FW F1` | See below | **PASS** | challenger fw_f1=0.3332  vs  champion fw_f1=0.3332  (min required: 0.3282) |
| `R2 FW MAE` | See below | **PASS** | challenger fw_mae=1.5500 hrs  vs  champion fw_mae=1.5500 hrs  (max allowed: 1.6275) |
| `R3 MACRO F1` | See below | **PASS** | challenger macro_f1=0.2442  vs  champion macro_f1=0.2442  (min required: 0.2342) |

### Rule definitions

**`R1 FW F1`** — Frequency-weighted F1 must not drop by more than 0.005  
Primary business metric. A ≤0.5pp dip is tolerated (noise from stochastic training); any larger drop means the challenger is materially worse on common work steps.

**`R2 FW MAE`** — Frequency-weighted MAE must not increase by more than 5%  
Duration quality guard. Body-shop planning depends on accurate hour estimates; a challenger that inflates MAE by more than 5% would degrade scheduling quality.

**`R3 MACRO F1`** — Macro F1 must not drop by more than 0.010  
Rare-step fairness guard. Frequency-weighting can hide collapse on low-frequency steps. This rule ensures no individual target degrades catastrophically.

---

## Decision: PROMOTED

All three promotion rules passed. The challenger has been copied into
the live champion slot. The previous champion has been archived under
`models/archive/`.
