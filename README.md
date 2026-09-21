# Machine Learning with Python — Medium-Level Projects (merged)

Three complete, runnable machine-learning projects following the specification in
`Machine Learning with Python PROJECT DETAILS.pdf`. This is a merge of two separate builds
of the same three-project assignment:

* **Frontend / pipeline code** — from the more modular build (Streamlit UI, `src/`
  package per project: config, features, preprocessing, train, evaluate).
* **Backend data** — from the second build's flat CSV datasets (`customers.csv`,
  `sensor_data.csv`, `products.csv` + `interactions.csv`).

The two builds used **different data schemas**, so this wasn't a drop-in swap: for each
project, `src/features.py`, `src/preprocessing.py` and `src/train.py` were rewritten to
work with the backend data's actual columns, and every model was retrained on it end to
end. See each project's own README for exactly what changed. Each project's synthetic
`generate_data.py` was removed — it targeted the old schema and would have overwritten the
backend dataset if run.

| # | Project | Task | Model(s) | Key metrics | Deployment |
|---|---------|------|----------|-------------|------------|
| 1 | Customer Churn Prediction | Binary classification | Logistic Regression, Random Forest, XGBoost† | ROC-AUC, Precision, Recall, F1, Calibration | Streamlit + ranked high-risk list |
| 2 | Predictive Maintenance | Time-window failure classification | Logistic Regression, Random Forest, XGBoost† | Recall, Precision, PR-AUC, False alarms | Streamlit monitoring dashboard |
| 3 | Personalized Product Recommendation | Recommender | Popularity, Implicit ALS, Content-based (TF-IDF), Hybrid | Precision@K, Recall@K, NDCG@K | FastAPI + Streamlit |

† `xgboost` isn't installed in the sandbox this was built in, so `train.py` in projects 1
and 2 automatically falls back to scikit-learn's `HistGradientBoostingClassifier` when
`xgboost` is unavailable, and all shipped model artifacts were trained with that fallback.
Install `xgboost>=2.0` and rerun `python -m src.train` in each project to use the original
XGBoost model instead — no other code changes needed.

## Repository layout

```
.
├── README.md                     # this file
├── requirements.txt               # shared dependencies
├── 01_customer_churn/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/                       # config, features, preprocessing, train, evaluate
│   ├── data/customers.csv         # backend dataset
│   ├── models/                    # trained model artifacts (included, ready to use)
│   ├── outputs/                   # metrics.json, plots, ranked list (included)
│   └── app.py                     # Streamlit UI (frontend)
├── 02_predictive_maintenance/
│   ├── data/sensor_data.csv       # backend dataset
│   └── ... (same layout)
└── 03_product_recommendation/
    ├── data/products.csv, interactions.csv   # backend dataset
    ├── api.py                     # FastAPI service
    └── ... (same layout, plus api.py)
```

## Quick start (any project)

```bash
cd 01_customer_churn        # or 02_predictive_maintenance / 03_product_recommendation
pip install -r requirements.txt
python -m src.train          # retrain on the backend data, regenerate outputs/models
streamlit run app.py         # launch the dashboard
```

Trained models and generated outputs are already included in each project's `models/` and
`outputs/` folders — you don't have to retrain before launching `app.py`.

## What to read next
Each project folder has its own README with the exact data schema, feature engineering,
model comparison numbers (from an actual run on this backend data), and how to point the
pipeline at your own data.
