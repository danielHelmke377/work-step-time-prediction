# Model Card: Work Step Time Predictor

## Model Details
- **Architecture:** Two-stage cascaded pipeline (Multi-label Classification → Conditional Regression).
- **Core Models:** `LGBMClassifier`, `Ridge`, `LGBMRegressor` (Scikit-Learn & LightGBM).
- **Domain:** German Automotive Repair & Body Shops.
- **Input:** Unstructured JSON representation of repair orders (Make, text items, cost centers, times, prices).
- **Output:** 14 independent numeric floating-point values representing predicted hours for specific work steps.

## Intended Use
- **Primary Use Case:** Automatically estimating the duration of 14 key repair workflow steps based on initial repair order documentation.
- **Target Audience:** Body shop managers, insurance estimators, and automotive service planning software.
- **Out-of-Scope Uses:** 
  - Making fully autonomous operational or financial decisions without human oversight.
  - Predicting outcomes for non-automotive or non-German language repair text.

## Training Data & Limitations
- **Data Scale:** The model was trained on an extremely constrained dataset of ~500 historical repair orders. 
- **Language Bias:** The feature engineering relies heavily on German automotive domain vocabulary and hand-crafted regex flags (e.g., `lackier`, `scheibe`). It will completely fail on English or other languages.
- **Make/Brand Bias:** The model applies one-hot encoded features for the top 8 vehicle makes (e.g., `MERCEDES-BENZ`, `VW`). Rare makes (e.g., `Maserati`) are grouped into an 'Other' category and may suffer from less accurate duration predictions.

## Metrics & Evaluation
- **Optimization Strategy:** Models were selected and tuned based on **Frequency-Weighted metrics** to align with business reality (optimizing for the most common tasks rather than over-indexing on rare edge cases).
- **Threshold Tuning:** Classification decision thresholds were strictly tuned using F1-score maximization on the validation set to balance Precision and Recall for imbalanced targets.

## Known Failure Modes
1. **Extreme Sparsity (The "Hail" Problem):** 
   - Targets with less than 5 positive examples in the dataset (like `hailrepair` or `plasticrepair`) inherently struggle with regression variance. The pipeline includes safety fallback logic (e.g., mean imputation) if the regressor fails or cannot be trained.
2. **Out-of-Vocabulary Surprises:**
   - Because the TF-IDF vectorizer was trained on only ~500 records, the appearance of entirely novel part names or shorthand abbreviations in production could degrade text classification accuracy.
3. **Cascading Errors:**
   - The two-stage architecture inherently stacks errors. If Stage 1 fails to identify that a target is active (False Negative), Stage 2 will never trigger, forcing a 0.0 hour prediction regardless of the true duration. High recall in Stage 1 was prioritized to mitigate this.
