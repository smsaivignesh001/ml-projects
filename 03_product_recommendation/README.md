# Project 3 — Personalized Product Recommendation Model

Recommend products to users based on their past interactions and product similarity.
Implements every step of the PDF's approach, including a **hybrid** model and a
**content-based cold-start fallback**, evaluated with proper ranking metrics on a
**time-based split**, and served through both a **FastAPI** service and a **Streamlit** UI.

| PDF step | Where it lives |
|----------|----------------|
| 1. Prepare user-item interaction data (views, clicks, carts, purchases) | `data/products.csv`, `data/interactions.csv` (backend dataset) |
| 2. Popularity-based baseline | `src/popularity.py` |
| 3. Collaborative filtering / matrix factorization | `src/matrix_factorization.py` (implicit ALS) |
| 4. Item metadata → content-based fallback for new products | `src/content_based.py` |
| 5. Precision@K, Recall@K, NDCG@K on a time-based validation split | `src/evaluate.py`, `src/train.py` |
| 6. Recommendation API + analysis across user segments | `api.py`, `app.py`, `src/recommender.py` |

## Data

* **`data/products.csv`** — 200 products: `product_id`, `category` (8 categories),
  `popularity_weight`, `price`. No brand or tags in this data source, so the content-based
  model builds its TF-IDF document from category + price band only.
* **`data/interactions.csv`** — ~15,400 events: `user_id`, `product_id`, `event_type`
  (view/click/cart/purchase), `event_weight`, `day` (integer day index 0–149, used as the
  chronological axis — there's no separate users table, so the user universe is just every
  `user_id` seen here).

`src/train.py::load_data` renames these to the pipeline's internal column names
(`product_id`→`item_id`, `event_type`→`interaction_type`, `day`→`timestamp`) so the rest of
the code is unchanged. "Cold-start" items are no longer a static flag in the data — they're
computed at runtime as whichever items have zero training interactions
(`UnifiedRecommender.cold_start_item_ids()`).

## Models

1. **Popularity** (`popularity.py`) — non-personalized baseline; items ranked by
   log-damped engagement weight (purchase > cart > click > view).
2. **Collaborative filtering** (`matrix_factorization.py`) — **implicit-feedback ALS**
   (Hu–Koren–Volinsky 2008): confidence `c_ui = 1 + α·r_ui`, alternating closed-form
   solves for user/item factors. No gradient tuning, fast, the standard strong baseline.
3. **Content-based** (`content_based.py`) — TF-IDF over item metadata (category, price
   band). User profile = interaction-weighted average of item vectors; scores = cosine
   similarity. Provides **item-to-item similarity** that works for **cold-start** products
   with no history.
4. **Hybrid** (`recommender.py`) — per-user min-max-normalised blend (ALS 0.72, popularity
   0.18, content 0.10). Fallback chain: known user → hybrid; cold user → popularity;
   cold item → content similarity.

## Evaluation (`src/evaluate.py`) — step 5

Chronological split (earliest 80% train, latest 20% test, cut at day 119). For each
held-out user, every catalogue item not already seen in training is ranked; the top-K is
scored against the items the user actually engaged with in the test window:

```
Precision@K = |relevant ∩ topK| / K
Recall@K    = |relevant ∩ topK| / |relevant|
NDCG@K      = DCG@K / IDCG@K      (binary relevance)
HitRate@K   = fraction of users with ≥1 relevant item in top-K
```

Actual result on the backend dataset (K=10) — personalization clearly beats the popularity
baseline:

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 |
|-------|-------------:|----------:|--------:|-----------:|
| **Hybrid** | **0.069** | **0.132** | **0.112** | **0.508** |
| ALS (collab. filtering) | 0.056 | 0.107 | 0.090 | 0.438 |
| Content-based | 0.056 | 0.102 | 0.079 | 0.421 |
| Popularity | 0.045 | 0.095 | 0.080 | 0.371 |

Content-based is weaker as a *ranker* here (fewer metadata fields than a real catalogue
would have) but still beats the non-personalized popularity baseline, and remains the
fallback for cold-start items — see the Similar/Cold-start tab.

## API (`api.py`) — step 6

```bash
uvicorn api:app --reload --port 8000     # docs at http://localhost:8000/docs
```

| Endpoint | Purpose |
|----------|---------|
| `GET /recommend/{user_id}?k=10` | personalized top-K with reasons + strategy used |
| `GET /similar/{item_id}?k=10` | content-based item-to-item (cold-start safe) |
| `GET /popular?k=10` | popularity baseline |
| `GET /metrics` | offline Precision@K / Recall@K / NDCG@K per model |
| `GET /segments` | recommendation analysis across user segments |

## UI (`app.py`)

Tabs: **Recommend** (per-user top-K with reasons, cold-user simulation), **Similar /
Cold-start** (item-to-item, including ❄️ cold-start products), **Model Performance**
(metric comparison + strong-signal evaluation), **Segments** (Light/Medium/Heavy/Cold),
and **Plots** (metric comparison, catalogue coverage/diversity).

```bash
streamlit run app.py
```

## Run it

```bash
pip install -r requirements.txt
python -m src.train           # fit all models, evaluate on time split, persist
uvicorn api:app --reload      # recommendation API
streamlit run app.py          # interactive UI
# or:
./run_all.sh
```

### Artefacts
* `models/recommender.joblib` (~450 KB) — compact model; dense score matrices are rebuilt
  on load via `load_recommender()`.
* `outputs/metrics.json` — all ranking metrics + segment analysis.
* `outputs/segment_recommendations.csv`, `metrics_comparison.png`, `catalog_coverage.png`.

## Using your own data
Provide `products.csv` (with `product_id, category, price`) and `interactions.csv` (with
`user_id, product_id, event_type, day`). `src/train.py::load_data` handles the column
rename; the split, training and evaluation are unchanged. Tune `ALS_FACTORS`, `ALS_ALPHA`,
`TOP_K` and the hybrid `BLEND` weights in `src/config.py`.
