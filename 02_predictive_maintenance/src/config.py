"""Central configuration for the Predictive Maintenance project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Backend data source: one flat, hourly sensor reading per row, with an inline
# failure_event flag instead of a separate failures table.
SENSOR_CSV = DATA_DIR / "sensor_data.csv"

BEST_MODEL_PATH = MODEL_DIR / "best_maintenance_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"
RISK_SNAPSHOT_CSV = MODEL_DIR / "machine_risk_snapshot.csv"
RISK_TIMELINE_CSV = MODEL_DIR / "risk_timeline.csv"

METRICS_JSON = OUTPUT_DIR / "metrics.json"
PR_PNG = OUTPUT_DIR / "precision_recall.png"
ROC_PNG = OUTPUT_DIR / "roc_curves.png"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance.png"
FALSE_ALARM_PNG = OUTPUT_DIR / "false_alarm_tradeoff.png"

RANDOM_STATE = 42

# ---- Labelling -------------------------------------------------------------
PREDICTION_WINDOW_HOURS = 24     # label=1 if a failure occurs within the next 24h


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
