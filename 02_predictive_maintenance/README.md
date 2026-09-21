# Project 2 — Predictive Maintenance for Industrial Equipment

Predict whether a machine is likely to fail within a future time window using sensor
readings, so maintenance can be dispatched *before* the failure. Implements every step of
the PDF's approach.

| PDF step | Where it lives |
|----------|----------------|
| 1. Timestamped sensor data (temperature, vibration, pressure, operating hours) | `data/sensor_data.csv` (backend dataset) |
| 2. Rolling statistics & lag features, no future-data leakage | `src/features.py::add_features` |
| 3. Define a failure-prediction window and label each observation | `src/features.py::add_labels` |
| 4. Train classifiers; handle rare failures with sampling/weights | `src/train.py` |
| 5. Evaluate recall, precision, PR-AUC and number of false alarms | `src/evaluate.py`, `outputs/` |
| 6. Monitoring view of machine risk scores + alerts for high-risk equipment | `app.py`, `models/machine_risk_snapshot.csv` |

## Data

`data/sensor_data.csv` — 15 machines, **hourly** readings over 90 days (~32k rows, backend
dataset): `hour` (integer hour index, not a wall-clock date), `operating_hours`,
`hours_since_maintenance`, `temperature_c`, `vibration_mm_s`, `pressure_kpa`, and an inline
`failure_event` flag (1 on the hour a failure happens — 62 events total, ~4 per machine;
`hours_since_maintenance` resets after each one).

There's no separate failures table in this data source, so `add_labels` reconstructs
"hours until the next failure" directly from the per-row `failure_event` flag instead of
joining a telemetry table against a failures table.

## Labelling & leakage-safe features (`src/features.py`)

* **Label (step 3):** an observation is positive if the machine fails within the next
  `PREDICTION_WINDOW_HOURS` (default 24 h). Positive rate ≈ 4.8% → **~1:20 imbalance**.
* **Features (step 2):** for every sensor, backward-looking rolling mean/std/max over
  6 h / 12 h / 24 h windows, lags 1–3, a first difference, a short-window slope
  (rate-of-change), deviation from the recent mean, plus a couple of cross-sensor
  interactions (temp×vibration, vibration/pressure). **All windows look only at the current
  and past rows** — no future reading leaks into a feature.
* **Split:** chronological (train = first 75% of the timeline, test = last 25%). The
  leakage-only helper `time_to_failure_h` is excluded from the feature set.

## Modelling (`src/train.py`)

Logistic Regression, Random Forest and XGBoost, each with rare-event handling
(`class_weight='balanced'` / `balanced_subsample`, XGBoost `scale_pos_weight`). Best model
is chosen by **PR-AUC**. If `xgboost` isn't installed, `train.py` automatically falls back
to scikit-learn's `HistGradientBoostingClassifier` (feature importance falls back further
to permutation importance, since that model exposes neither `feature_importances_` nor
`coef_`) — install `xgboost>=2.0` to use the original model. Example run on the backend
dataset:

| Model | PR-AUC | Recall | Precision | False alarms |
|-------|--------|--------|-----------|--------------|
| RandomForest | 1.000 | 0.994 | 0.984 | 8 |
| GradientBoosting (xgboost fallback) | 1.000 | 0.996 | 0.982 | 9 |
| LogisticRegression | 0.999 | 0.998 | 0.975 | 13 |

(This dataset's failures have a strong, consistent sensor precursor, so recall is close to
1.0 for all three models — less "sudden failure" noise than a real plant, but a clean
signal to verify the pipeline against.)

## Monitoring view (`app.py`)

* **Fleet Status** — current risk score per machine with 🔴 ALERT / 🟠 WATCH / 🟢 OK
  colour-coding and headline metrics (recall, false alarms, alert counts).
* **Machine Drill-down** — per-machine risk timeline (x-axis = operating hour) shaded by
  the failure window, marked with actual failure hours, plus a 4-panel sensor view
  (temperature, vibration, pressure, hours since maintenance).
* **Model Performance** and **Plots** — PR/ROC curves, the precision/recall/false-alarm
  trade-off vs. threshold, confusion matrix, and feature importances.

## Run it

```bash
pip install -r requirements.txt
python -m src.train           # label, feature, train, evaluate, snapshot
streamlit run app.py          # monitoring dashboard
# or:
./run_all.sh
```

### Outputs
* `models/machine_risk_snapshot.csv` — latest risk score + alert per machine (step 6).
* `models/risk_timeline.csv` — full scored history powering the drill-down.
* `outputs/` — `metrics.json`, PR/ROC/confusion/importance plots, false-alarm trade-off.

## Using your own data
Replace `data/sensor_data.csv` with your historian export (same columns, one row per
machine per reading, with a `failure_event` flag). `add_labels` builds the window labels
directly from that flag; everything downstream is unchanged. Tune
`PREDICTION_WINDOW_HOURS` in `src/config.py` to your maintenance lead time.
