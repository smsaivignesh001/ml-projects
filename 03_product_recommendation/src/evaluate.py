"""Offline evaluation for recommendations (step 5 of the PDF).

Uses a **time-based validation split**: the earliest interactions train the models, the
latest interactions are held out as ground truth. For each held-out user we rank every
catalogue item they have *not* already seen in training, take the top-K, and score against
the items they actually engaged with in the test window using:

    Precision@K = |relevant ∩ topK| / K
    Recall@K    = |relevant ∩ topK| / |relevant|
    NDCG@K      = DCG@K / IDCG@K   (binary relevance, position-discounted)

Ranking metrics on implicit feedback with a chronological split is the standard way to
avoid the optimistic bias of a random split (which lets the model see the future).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class RankingMetrics:
    name: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_rate: float          # fraction of users with >=1 relevant item in top-K
    k: int
    n_users_evaluated: int

    def to_dict(self):
        return asdict(self)


def time_split(interactions: pd.DataFrame, train_frac: float = 0.8):
    """Chronological train/test split of interactions."""
    d = interactions.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(d) * train_frac)
    return d.iloc[:cut].reset_index(drop=True), d.iloc[cut:].reset_index(drop=True)


def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def evaluate_scores(name: str,
                    scores: np.ndarray,
                    train_items: dict[int, set[int]],
                    test_relevant: dict[int, set[int]],
                    k: int = 10) -> RankingMetrics:
    """Evaluate a dense (n_users x n_items) score matrix against held-out relevance.

    Items the user interacted with during training are masked out (score -> -inf) so the
    model is scored only on novel recommendations.
    """
    precisions, recalls, ndcgs, hits = [], [], [], []
    n_items = scores.shape[1]
    for u, relevant in test_relevant.items():
        if not relevant:
            continue
        row = scores[u].astype(np.float64).copy()
        seen = train_items.get(u)
        if seen:
            mask = np.fromiter(seen, dtype=int, count=len(seen))
            row[mask] = -np.inf
        k_eff = min(k, n_items)
        top = np.argpartition(-row, k_eff - 1)[:k_eff]
        top = top[np.argsort(-row[top])]        # ordered best-first
        rel_vec = np.isin(top, list(relevant)).astype(float)

        n_hit = float(rel_vec.sum())
        precisions.append(n_hit / k_eff)
        recalls.append(n_hit / len(relevant))
        hits.append(1.0 if n_hit > 0 else 0.0)

        dcg = _dcg(rel_vec)
        ideal = np.ones(min(len(relevant), k_eff))
        idcg = _dcg(ideal)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return RankingMetrics(
        name=name,
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        k=k,
        n_users_evaluated=len(precisions),
    )


def build_eval_structures(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          item_index: dict[str, int], user_index: dict[str, int],
                          strong_only: bool = False):
    """Return (train_items, test_relevant) keyed by user index -> set of item indices."""
    train_items: dict[int, set[int]] = {}
    for uid, iid in zip(train_df["user_id"], train_df["item_id"]):
        u, i = user_index.get(uid), item_index.get(iid)
        if u is None or i is None:
            continue
        train_items.setdefault(u, set()).add(i)

    test = test_df
    if strong_only:
        test = test[test["interaction_type"].isin(["cart", "purchase"])]
    test_relevant: dict[int, set[int]] = {}
    for uid, iid in zip(test["user_id"], test["item_id"]):
        u, i = user_index.get(uid), item_index.get(iid)
        if u is None or i is None:
            continue
        test_relevant.setdefault(u, set()).add(i)
    # keep only users who have both training history and a test target
    test_relevant = {u: r for u, r in test_relevant.items() if u in train_items}
    return train_items, test_relevant


def metrics_dataframe(metrics_list: list[RankingMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    return df.sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)


def plot_metrics_comparison(df: pd.DataFrame, path, k: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate"]
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", f"HitRate@{k}"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(df), 1)
    plt.figure(figsize=(9, 5))
    for r, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        plt.bar(x + r * width, vals, width, label=row["name"])
        for xi, v in zip(x + r * width, vals):
            plt.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title(f"Recommender comparison @K={k}")
    plt.legend()
    plt.ylim(0, max(0.05, np.nanmax(df[metrics].values) * 1.25))
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_catalog_coverage(scores_by_model: dict[str, np.ndarray], item_ids: list[str],
                          path, k: int = 10, n_users: int = 300) -> None:
    """Catalogue coverage: how many distinct items appear in users' top-K (diversity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverage = {}
    rng = np.random.default_rng(0)
    sample_users = rng.choice(scores_by_model[next(iter(scores_by_model))].shape[0],
                              size=min(n_users, scores_by_model[next(iter(scores_by_model))].shape[0]),
                              replace=False)
    for name, scores in scores_by_model.items():
        seen = set()
        for u in sample_users:
            top = np.argpartition(-scores[u], k)[:k]
            seen.update(int(i) for i in top)
        coverage[name] = len(seen) / len(item_ids)

    plt.figure(figsize=(6, 4.5))
    plt.bar(range(len(coverage)), list(coverage.values()), color="teal")
    plt.xticks(range(len(coverage)), list(coverage.keys()))
    plt.ylabel(f"Catalogue coverage @K={k}")
    plt.ylim(0, 1)
    plt.title("Catalogue Coverage (diversity of top-K across users)")
    for i, v in enumerate(coverage.values()):
        plt.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
