"""Central configuration for the Customer Churn project.

All file paths are resolved relative to the project root so the pipeline can be run
from anywhere (``python -m src.train`` from the project folder).
"""
from __future__ import annotations

from pathlib import Path

# Project root = .../01_customer_churn
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Raw dataset (sourced from the flat customers.csv backend — no separate
# monthly usage_history table in this data source)
CUSTOMERS_CSV = DATA_DIR / "customers.csv"

# Model artefacts
BEST_MODEL_PATH = MODEL_DIR / "best_churn_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"

# Outputs
METRICS_JSON = OUTPUT_DIR / "metrics.json"
RANKED_LIST_CSV = OUTPUT_DIR / "high_risk_customers.csv"
CALIBRATION_PNG = OUTPUT_DIR / "calibration_curve.png"
ROC_PNG = OUTPUT_DIR / "roc_curves.png"
PR_PNG = OUTPUT_DIR / "precision_recall.png"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance.png"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix.png"

RANDOM_STATE = 42

# Target class imbalance for the synthetic dataset (~26% churn)
CHURN_BASE_RATE = 0.26

N_CUSTOMERS = 12_000
USAGE_MONTHS = 6  # months of usage history per customer


def ensure_dirs() -> None:
    """Create the standard project directories if they do not exist."""
    for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
