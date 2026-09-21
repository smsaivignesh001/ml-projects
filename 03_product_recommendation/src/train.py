"""Training + evaluation pipeline for the Product Recommendation project (steps 2-6).

  2. popularity baseline   3. collaborative filtering (ALS)   4. content-based fallback
  5. Precision@K / Recall@K / NDCG@K on a time-based split   6. segment analysis

Run:
    python -m src.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from . import config
from . import evaluate as ev
from .recommender import UnifiedRecommender

warnings.filterwarnings("ignore")
K = config.TOP_K


def load_data():
    """Load the backend catalogue + event log, normalised to the internal schema.

    products.csv uses product_id -> item_id; interactions.csv uses product_id ->
    item_id, event_type -> interaction_type, day -> timestamp (an integer day
    index, not a wall-clock date — sorting/ordering by it still gives chronology).
    There is no separate user profile table, so the user universe is simply
    every user_id seen in the interaction log.
    """
    items = pd.read_csv(config.ITEMS_CSV).rename(columns={"product_id": "item_id"})
    interactions = pd.read_csv(config.INTERACTIONS_CSV).rename(columns={
        "product_id": "item_id", "event_type": "interaction_type", "day": "timestamp",
    })
    users = pd.DataFrame({"user_id": sorted(interactions["user_id"].unique())})
    return users, items, interactions


def segment_analysis(rec: UnifiedRecommender, sample_per_segment: int = 60) -> pd.DataFrame:
    """Show how recommendations differ across user segments (step 6)."""
    seg_df = rec.segment_users()
    rows = []
    for seg in ["Cold", "Light", "Medium", "Heavy"]:
        users = seg_df[seg_df["segment"] == seg]["user_id"].tolist()
        if not users:
            continue
        rng = np.random.default_rng(config.RANDOM_STATE)
        sample = rng.choice(users, size=min(sample_per_segment, len(users)), replace=False)
        cats, n_recs = [], 0
        for uid in sample:
            r = rec.recommend(uid, K)["recommendations"]
            n_recs += len(r)
            cats += [x["category"] for x in r if "category" in x]
        top_cats = pd.Series(cats).value_counts().head(3)
        rows.append({
            "segment": seg,
            "n_users": len(users),
            "users_sampled": len(sample),
            "top_recommended_categories": ", ".join(f"{c} ({n})" for c, n in top_cats.items()),
            "distinct_categories_recommended": pd.Series(cats).nunique(),
            "avg_items_per_user": round(n_recs / max(len(sample), 1), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    config.ensure_dirs()
    users, items, interactions = load_data()
    print(f">> Loaded {len(users)} users, {len(items)} items, {len(interactions):,} interactions")

    train_df, test_df = ev.time_split(interactions, config.TIME_TRAIN_FRAC)
    print(f"   time-based split -> train={len(train_df):,} test={len(test_df):,} "
          f"(cutoff day {train_df['timestamp'].max()})")

    print(">> Fitting hybrid recommender (ALS + content + popularity) on TRAIN only ...")
    rec = UnifiedRecommender(items, users).fit(train_df)

    # ---- evaluation structures ----
    train_items, test_relevant = ev.build_eval_structures(
        train_df, test_df, rec.item_index, rec.user_index)
    print(f"   evaluating on {len(test_relevant)} held-out users with training history")

    # ---- score matrices per model ----
    score_map = {
        "Popularity": rec.component_scores["popularity"],
        "Content-based": rec.component_scores["content"],
        "ALS (collab. filtering)": rec.component_scores["als"],
        "Hybrid": rec.blended_scores,
    }

    metrics_list = []
    print(f">> Ranking metrics @K={K}:")
    for name, scores in score_map.items():
        m = ev.evaluate_scores(name, scores, train_items, test_relevant, k=K)
        metrics_list.append(m)
        print(f"   {name:24s} P@K={m.precision_at_k:.4f} R@K={m.recall_at_k:.4f} "
              f"NDCG@K={m.ndcg_at_k:.4f} HitRate={m.hit_rate:.3f}")

    # strong-signal evaluation (only cart/purchase count as relevant)
    _, test_relevant_strong = ev.build_eval_structures(
        train_df, test_df, rec.item_index, rec.user_index, strong_only=True)
    strong_metrics = []
    for name, scores in score_map.items():
        m = ev.evaluate_scores(name, scores, train_items, test_relevant_strong, k=K)
        m.name = name
        strong_metrics.append(m)

    metrics_df = ev.metrics_dataframe(metrics_list)
    best_name = metrics_df.iloc[0]["name"]
    print(f">> Best model by NDCG@{K}: {best_name}")

    # ---- plots ----
    ev.plot_metrics_comparison(metrics_df, config.METRICS_BAR_PNG, k=K)
    ev.plot_catalog_coverage(score_map, rec.item_ids, config.CATALOG_DIVERSITY_PNG, k=K)

    # ---- segment analysis ----
    seg = segment_analysis(rec)
    seg.to_csv(config.SEGMENTS_CSV, index=False)

    # ---- example recommendations (while scores are still in memory) ----
    print("\nSegment analysis:")
    print(seg.to_string(index=False))
    seg_users = rec.segment_users()
    heavy_users = seg_users[seg_users["segment"] == "Heavy"]["user_id"].tolist()
    if heavy_users:
        hu = heavy_users[0]
        ex = rec.recommend(hu, 5)
        print("\nExample recommendations (first Heavy user):")
        print(f"  user {hu} via {ex['strategy']}")
        for r in ex["recommendations"]:
            print(f"    - {r['item_id']} [{r.get('category')}] "
                  f"score={r['score']} :: {r['reason']}")

    # ---- persist model + card + metrics ----
    # Store only the compact parts (factors, sparse matrices); the dense score matrices
    # are rebuilt on load by ``load_recommender``. This keeps the artefact a few MB.
    rec.blended_scores = None
    rec.component_scores = {}
    joblib.dump(rec, config.MODEL_PATH)
    with open(config.METRICS_JSON, "w") as f:
        json.dump({
            "k": K,
            "best_model": best_name,
            "train_interactions": int(len(train_df)),
            "test_interactions": int(len(test_df)),
            "n_users_evaluated": int(metrics_df.iloc[0]["n_users_evaluated"]),
            "metrics_all": metrics_df.to_dict(orient="records"),
            "metrics_strong_signals": [m.to_dict() for m in strong_metrics],
            "segment_analysis": seg.to_dict(orient="records"),
            "n_items": len(items),
            "n_cold_start_items": len(rec.cold_start_item_ids()),
        }, f, indent=2, default=str)

    with open(config.MODEL_CARD_PATH, "w") as f:
        json.dump({
            "task": "Personalized product recommendation",
            "models": ["popularity", "ALS implicit matrix factorization",
                       "content-based (TF-IDF)", "hybrid blend"],
            "blend_weights": config.BLEND,
            "als": {"factors": config.ALS_FACTORS, "iterations": config.ALS_ITERATIONS,
                    "reg": config.ALS_REG, "alpha": config.ALS_ALPHA},
            "evaluation": {"split": "time-based", "k": K,
                           "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K"]},
            "best_model": best_name,
        }, f, indent=2)

    print("\n>> Saved model, metrics and plots.")


if __name__ == "__main__":
    main()
