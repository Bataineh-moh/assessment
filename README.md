# Freight Rate Prediction - Machine Learning Assessment

This repository contains the end-to-end Machine Learning pipeline to predict freight shipping rates (`predicted_rate`) on spot market loads, evaluate temporal generalization, and generate batch predictions for production scoring.

---

## 1. Project Overview & Architecture
* **Model**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor`).
* **Feature Engineering**: 
  - Cyclical temporal encodings (day of year, day of week sine/cosine).
  - Empirical Bayes smoothed target encoding for high-cardinality origin-destination lanes (4,000+ distinct pairs).
  - Ton-mile domain interactions and log distance transforms.
* **Data Quality Handling**: Rectified negative weight anomalies (sign-inversion errors) via absolute value correction and leveraged native missing-value routing for unobserved market indices.
* **Validation Strategy**: Temporal hold-out validation (training on earlier months, testing on subsequent months) to prevent future data leakage.

---

## 2. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd assessment
pip install -r requirements.txt