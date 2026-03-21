# Prediction Quality Monitoring Snapshot

**Generated:** 2026-03-21  
**Source:** synthetic batch (n=30)  
**Orders analysed:** 30

---

## 1. Work-step prevalence

| Target | Label | Active % | Avg confidence | Avg predicted hrs |
|---|---|---:|---:|---:|
| `calibration` | Calibration (ADAS/cameras) |   3.3% | 0.020 | 4.20 |
| `wheelmeasurement` | Wheel alignment measurement |   3.3% | 0.019 | 1.33 |
| `bodymeasurement` | Body/chassis measurement |   0.0% | 0.011 | 0.00 |
| `dismounting` | Dis-/mounting |  96.7% | 0.496 | 5.00 |
| `bodyrepair` | Body repair |  10.0% | 0.015 | 6.15 |
| `assembly` | Assembly |  73.3% | 0.310 | 10.00 |
| `plasticrepair` | Plastic repair |  30.0% | 0.071 | 4.48 |
| `allTiresService` | Tyre service |  23.3% | 0.042 | 2.44 |
| `cleaning` | Cleaning |   0.0% | 0.001 | 0.00 |
| `paintingPreparation` | Painting — preparation |   6.7% | 0.012 | 21.70 |
| `paintingSpraying` | Painting — spraying |  20.0% | 0.043 | 5.72 |
| `paintingFinish` | Painting — finish |   0.0% | 0.002 | 0.00 |
| `hailrepair` | Hail repair |  30.0% | 0.319 | 6.28 |
| `glas` | Glass replacement |  43.3% | 0.110 | 3.11 |

---

## 2. Aggregate statistics

- **Total predicted hours per order:** min 5.55  avg 20.70  max 38.08
- **Active work steps per order:** min 1.00  avg 3.40  max 5.00
- **Inference latency (ms):** min 25.50  avg 27.80  max 33.50

---

## 3. Alerts

[OK] No anomalies detected in this batch.

---

## 4. What would be monitored in production

| Signal | Method | Threshold |
|---|---|---|
| Classifier confidence distribution | KS test vs. training reference | p < 0.05 |
| Work-step prevalence drift | Rolling 7-day vs. 28-day baseline | > 15 pp shift |
| Predicted duration anomalies | Z-score per target | |z| > 3 |
| Request validation error rate | Fraction of 422 responses | > 5% |
| Inference latency (p95) | API logs | > 500 ms |
| Active predictions with zero hours | Direct check | any occurrence |

> In a live system this script would consume `/predict` request logs, compare
> against a reference distribution captured at training time, and push alerts
> to a monitoring dashboard (e.g. Grafana, MLflow, or a simple email digest).
