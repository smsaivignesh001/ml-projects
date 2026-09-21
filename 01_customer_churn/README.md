# Project 1 — Customer Churn Prediction

Predict which customers are likely to leave a subscription service so the business can
target retention efforts. This project implements every step of the PDF's approach.

| PDF step | Where it lives |
|----------|----------------|
| 1. Collect demographics, tenure, usage, support interactions, billing history | `data/customers.csv` (backend dataset) |
| 2. Handle missing values, categoricals, outliers, class imbalance | `src/preprocessing.py` |
| 3. Create features: usage trend, support-contact frequency, recent payment changes | `src/features.py` |
| 4. Train baseline Logistic Regression + tree models (Random Forest, XGBoost) | `src/train.py` |
| 5. Compare ROC-AUC, precision, recall, F1, calibration | `src/evaluate.py`, plots in `outputs/` |
| 6. Deploy best model + ranked high-risk list with review reasons | `src/train.py` → `outputs/high_risk_customers.csv`, `app.py` |

## Data

`data/customers.csv` — ~6,000 customers, one flat table (backend dataset): age, gender,
region, tenure, contract type, last/prior 30-day usage (GB), average call minutes, support
tickets (last 90d), average ticket resolution time, monthly/total charges, late payments
(last 6mo), payment method, and the `churn` label.

There's no separate usage-history time series in this data source, so `usage_trend` is
computed directly from `usage_last_30d_gb` vs `usage_prev_30d_gb` rather than from a
multi-month join.

## Feature engineering (`src/features.py`)

* **Usage trend** — relative change between last-30-day and prior-30-day usage.
* **Support-contact frequency** — tickets per month of tenure, high-support-burden flag.
* **Recent payment changes** — late-payment flag (last 6 months), charge-to-tenure ratio.
* Plus tenure groups and a new-month-to-month interaction flag.

## Modelling (`src/train.py`)

* **Preprocessor** — median imputation, winsorising (1st/99th percentile) for outliers,
  one-hot encoding for categoricals, standard scaling. Fit on the training fold only.
* **Models** — Logistic Regression (baseline), Random Forest, XGBoost. If `xgboost` isn't
  installed, `train.py` automatically falls back to scikit-learn's
  `HistGradientBoostingClassifier` so the pipeline still runs; install `xgboost>=2.0` to
  use the original model.
* **Class imbalance** — `class_weight='balanced'` / `balanced_subsample` and XGBoost
  `scale_pos_weight`.
* **Evaluation** — ROC-AUC, PR-AUC, precision, recall, F1 at the best-F1 threshold, plus a
  reliability (calibration) diagram and Brier score. The best model by ROC-AUC is persisted
  together with its preprocessor.
* **Output** — the whole customer base is scored and ranked; the top 200 high-risk
  customers are written to `outputs/high_risk_customers.csv` with rule-based
  **suggested review reasons** (e.g. "Month-to-month contract · Usage down 35% · 2 late
  payments").

## Run it

```bash
pip install -r requirements.txt
python -m src.train             # train, evaluate, rank, save artefacts + plots
streamlit run app.py            # interactive dashboard
# or do it all at once:
./run_all.sh
```

### Outputs (`outputs/`)
* `metrics.json` — full comparison table + headline numbers.
* `high_risk_customers.csv` — ranked retention list with review reasons.
* `roc_curves.png`, `precision_recall.png`, `calibration_curve.png`,
  `confusion_matrix.png`, `feature_importance.png`.

### Models (`models/`)
* `best_churn_model.joblib` — dict with `preprocessor`, `model`, `feature_names`, `threshold`.
* `model_card.json` — training metadata + all metrics.

## Using your own data
Point `load_and_engineer()` (in `src/features.py`) at your `customers.csv` with the same
column names — everything downstream (preprocessing, training, ranking, UI) is unchanged.
