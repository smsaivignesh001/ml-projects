"""Popularity baseline recommender (step 2 of the PDF).

The simplest useful recommender: rank items by how much engagement they attract, weighted
so stronger signals (purchases) count more than weaker ones (views). Non-personalized —
every user gets the same list — which makes it the reference a real model must beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class PopularityModel:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {"view": 1.0, "click": 2.0, "cart": 4.0, "purchase": 8.0}
        self.item_ids: list[str] = []
        self.scores: np.ndarray | None = None
        self.category_scores: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "PopularityModel":
        self.item_ids = items["item_id"].tolist()
        idx = {iid: i for i, iid in enumerate(self.item_ids)}

        inter = interactions.copy()
        inter["w"] = inter["interaction_type"].map(self.weights).fillna(1.0)
        pop = inter.groupby("item_id")["w"].sum()
        scores = np.zeros(len(self.item_ids))
        for iid, s in pop.items():
            if iid in idx:
                scores[idx[iid]] = s
        # log-damp so a few blockbusters don't dominate
        self.scores = np.log1p(scores)

        # per-category popularity (used for segment / diversity analysis)
        cat = (inter.assign(score=inter["w"])
               .merge(items[["item_id", "category"]], on="item_id", how="left")
               .groupby("category")["score"].sum()
               .sort_values(ascending=False))
        self.category_scores = cat
        return self

    def score_all(self, n_users: int) -> np.ndarray:
        """Dense (n_users x n_items) matrix — identical row for every user."""
        return np.tile(self.scores, (n_users, 1))

    def top_items(self, k: int = 10) -> list[tuple[str, float]]:
        top = np.argsort(-self.scores)[:k]
        return [(self.item_ids[i], float(self.scores[i])) for i in top]
