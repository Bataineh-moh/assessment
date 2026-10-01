# Freight Rate Prediction - Machine Learning Assessment

This repository contains the end-to-end Machine Learning pipeline to predict freight shipping rates (`predicted_rate`) on spot market loads, evaluate temporal generalization, and generate batch predictions for production scoring.

---

## 1. Project Overview & Architecture
* **Model**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor`).
* **Feature Engineering**: 
  - Cyclical temporal encodings (day of year, day of week sine/cosine).
  - Empirical Bayes smoothed target encoding for high-cardinality origin-destination lanes (4,000+ distinct pairs).
  - Ton-mile domain interactions and log distance transforms.
* **Data Quality Handling**: Rectified negative weight anomalies (sign-inversion errors) via absolute value correction and leveraged native missing-value routing for unobserved market indices and weights.
* **Validation Strategy**: Temporal hold-out validation — the model is trained on the first 8 months of `train-test.csv` (Jan-Aug) and evaluated on the last 2 (Sep-Oct), the most recent data available, mirroring the real task of predicting the unseen Nov/Dec validation period. This step prints MAE/RMSE/R²/MAPE on every run before the final model is refit on all 10 labeled months.
* **Seasonal Extrapolation Correction**: `train-test.csv` only covers Jan-Oct, so every date-derived feature is out of range for the Nov/Dec prediction window, and tree ensembles cannot extrapolate a trend past the feature range seen in training — they fall back to the nearest training-boundary leaf. To recover realistic day-to-day movement (most visible in the fixed December chart, which was flat before this fix), a small Ridge regression is fit on the holdout's out-of-sample residuals using only bounded cyclical features (sin/cos of day-of-week and day-of-year), then added on top of the tree's prediction for the validation and December sets. Because sine/cosine stay within [-1, 1] for any future date, this correction extrapolates smoothly instead of flatlining. The underlying signal is weak (holdout residual R² ≈ 0.001), so the correction is heavily regularized to stay a modest adjustment (a few percent) rather than fit holdout noise.

---

## 2. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd assessment
pip install -r requirements.txt
```

---

## 3. Data Placement

The raw data files are not committed to this repository (see `.gitignore`) and must be placed manually before running the pipeline:

```
data/train-test.csv      # labeled development data (train_test.csv also accepted)
data/validation.csv      # 12,000 loads requiring final predictions
```

`data/validation-predictions-template.csv` and `data/december_chart_inputs.csv` are already included.

---

## 4. How to Run

Run all commands from the repository root (`assessment/`).

**Step 1 — Train the model and generate predictions:**

```bash
python pyfiles/train_and_predict.py
```

This reads `data/train-test.csv` and `data/validation.csv`, trains the model, and writes:
* `validation_predictions.csv` — final `load_id,predicted_rate` predictions for all 12,000 validation loads.
* `december_chart_inputs.csv` / `data/december_chart_inputs.csv` — the fixed Lexington → Fort Wayne scenario (one row per December day) with `predicted_rate` filled in.

**Step 2 — Validate the outputs and generate the December chart:**

```bash
python pyfiles/score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv --output-dir scorer_results
```

This checks both output files against the required format and saves the chart to `scorer_results/candidate_december.png`.