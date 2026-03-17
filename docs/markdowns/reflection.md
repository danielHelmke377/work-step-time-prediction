# Reflection — Limitations, Improvements & Production Readiness

---

## 1. Biggest Weaknesses and Limitations

**Small dataset.** With 491 orders (294 training), the models are trained on a sample that may not represent the full distribution of repair types, regions, or seasons. Results are promising but variance is high — a single test split of 99 records is not a reliable estimate of real-world performance.

**Rare targets cannot be learned reliably.** `allTiresService` (13 positives) and `bodymeasurement` (21 positives) have too few examples for any ML model to generalise from. These targets need either more data or a rule-based fallback.

**`hailrepair` regression is poor.** The duration distribution is extremely skewed (100–4,000 min). Without a log-transform, the regressor fails on large cases. This is a known, fixable issue but was not resolved within scope.

**No hyperparameter tuning.** Models use fixed, sensible defaults. A proper cross-validated grid search or Bayesian optimisation (e.g. Optuna) would likely improve results for the harder targets.

**Text is purely bag-of-words.** TF-IDF captures individual terms and bigrams but misses long-range dependencies in position descriptions. With more data, a German domain-adapted language model (e.g. `german-bert`) could extract richer representations.

---

## 2. What I Would Do Differently with More Time or Data

| With more time | With more data |
|---|---|
| Log-transform `hailrepair` target | Replace TF-IDF with German BERT embeddings |
| Full cross-validated hyperparameter search | Train per-make sub-models where volume allows |
| Include spare parts as features | Time-based train/val split for realistic evaluation |
| Calibrate classifier probabilities (Platt scaling) | Expand keyword rules from corpus analysis |
| Add a model card and prediction confidence intervals | Learn `allTiresService` and `bodymeasurement` reliably |

---

## 3. Cold-Start Problem: New Workshop with Little Historical Data

When a new workshop joins with no history, the global model already handles most of the work — it was trained on orders from many different workshops and makes. The cold-start risk is real but manageable:

- **Start with the global model.** It generalises reasonably across workshops and makes. Predictions will not be workshop-specific but will be useful from day one.
- **Flag `make = unknown` explicitly.** The model handles it as a valid category, but predictions should carry a wider confidence interval.
- **Collect feedback actively.** Ask the workshop to confirm or correct the predicted work steps after each repair. Even 20–30 confirmed orders are enough to detect systematic bias.
- **Fine-tune on local data once volume allows.** After ~100 local orders, retrain or fine-tune the regression stage on the workshop's own data to capture local labour rate conventions, specialty areas, and price levels.
- **Rule baseline as safety net.** For the first weeks, blend the ML output with the rule baseline using a confidence-weighted average. The rules are always interpretable and auditable.

---

## 4. Keeping Models Accurate Over Time in Production

**Monitoring**
- Track prediction vs. actual for every completed order (once ground truth is available)
- Alert when MAE or F1 degrades by more than 10% relative to the baseline period
- Monitor input feature distributions weekly (price levels, time units, new makes, new position text vocabulary)

**Drift detection**
- **Concept drift:** Test whether the relationship between inputs and outputs is changing (e.g. new ADAS systems requiring new calibration steps not in the training vocabulary)
- **Data drift:** Monitor TF-IDF vocabulary coverage — if new technical terms appear frequently and are OOV, the text representation is degrading
- Use the rule baseline as a permanent reference: if the rule F1 stays stable but model F1 drops, it is likely a model drift issue, not a data quality issue

**Retraining strategy**
- **Scheduled retraining:** Monthly on a rolling window of the last 12 months of confirmed orders
- **Triggered retraining:** If monitoring alerts fire (MAE > threshold), trigger an immediate retraining cycle
- **Champion/challenger setup:** Always keep the current production model running; only promote a new model if it beats the champion on a held-out recent test set
- **Version all models and pipelines** with the training date and dataset hash, so any production issue can be rolled back instantly
