"""Central configuration for the Product Recommendation project."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Backend data source: a flat product catalogue + event log (no separate users
# table — the user universe is derived from whoever appears in interactions.csv).
ITEMS_CSV = DATA_DIR / "products.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

MODEL_PATH = MODEL_DIR / "recommender.joblib"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"

METRICS_JSON = OUTPUT_DIR / "metrics.json"
SEGMENTS_CSV = OUTPUT_DIR / "segment_recommendations.csv"
CATALOG_DIVERSITY_PNG = OUTPUT_DIR / "catalog_coverage.png"
METRICS_BAR_PNG = OUTPUT_DIR / "metrics_comparison.png"

RANDOM_STATE = 42

# Implicit-feedback weights per interaction type (purchase is strongest).
# Matches the event_weight scale already present in the backend interactions data.
INTERACTION_WEIGHTS = {"view": 1.0, "click": 2.0, "cart": 3.0, "purchase": 5.0}

# ---- Model hyper-parameters ------------------------------------------------
ALS_FACTORS = 32
ALS_ITERATIONS = 15
ALS_REG = 0.05
ALS_ALPHA = 8.0              # confidence scaling for implicit feedback

TOP_K = 10                   # evaluation / recommendation list length
TIME_TRAIN_FRAC = 0.8        # chronological train fraction

# Hybrid blending weights. ALS dominates (it is the strongest ranker); popularity adds a
# prior that helps low-activity users and tail items; content adds category alignment and
# is what powers the cold-start fallback.
BLEND = {"als": 0.72, "content": 0.10, "popularity": 0.18}


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
