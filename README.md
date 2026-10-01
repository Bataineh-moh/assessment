# Freight Rate Prediction - Machine Learning Assessment

This repository predicts freight shipping rates (`predicted_rate`) for the loads in
`data/validation.csv`, using `data/train-test.csv` as labeled training data.

---

## 1. Approach

* **Model**: `HistGradientBoostingRegressor` from scikit-learn (a gradient-boosted tree model).
  It handles missing values on its own and works well on tabular data like this without a lot
  of tuning, which made it a good first choice here.
* **Features used**: pickup/delivery city, equipment type, distance (and log of distance),
  weight, pickup/delivery coordinates, market index, quote signal, the average historical rate
  for that pickup -> delivery lane, and date-based features (month, day of week, day of year,
  weekend flag, and sine/cosine of day-of-year so December 31st and January 1st are treated as
  close together instead of far apart).
* **Data cleaning**: a small number of rows had a negative `weight`, which looks like a data
  entry mistake, so the sign is corrected. Missing `weight`/`market_index` values are left as-is
  and handled natively by the model (it can split around missing values on its own).
* **Validation**: trains on the first 8 months (Jan-Aug) and tests on the last 2 (Sep-Oct) of
  `train-test.csv`, since that best mirrors the real task of predicting the following months
  (Nov/Dec) from history. Prints MAE, RMSE, R², and MAPE on every run. After checking those
  numbers look reasonable, the final model is retrained on all 10 labeled months so it has as
  much data as possible for the actual predictions.
* **December chart fix**: `train-test.csv` only has data through October, so the model has
  never seen a December date and ends up predicting almost the same rate for every day in the
  fixed December chart. To fix this, the code checks the historical data for simple day-of-week
  patterns (e.g., are Tuesdays usually a bit higher than average?) and adds that small average
  adjustment on top of the model's prediction. It's a simple lookup, not a second model.

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

The raw data files are not committed to this repository (see `.gitignore`) and must be placed
manually before running the pipeline:

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

This reads `data/train-test.csv` and `data/validation.csv`, prints the validation metrics,
trains the final model, and writes:
* `validation_predictions.csv` — final `load_id,predicted_rate` predictions for all 12,000 validation loads.
* `data/december_chart_inputs.csv` — the fixed Lexington → Fort Wayne scenario (one row per December day) with `predicted_rate` filled in.

**Step 2 — Validate the outputs and generate the December chart:**

```bash
python pyfiles/score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv --output-dir scorer_results
```

This checks both output files against the required format and saves the chart to `scorer_results/candidate_december.png`.
