"""Unified recommender combining collaborative filtering, content, and popularity.

Provides the production-facing object used by both the API and the UI:

* ``recommend(user_id, k)`` – personalized top-K with human-readable reasons, and a
  graceful **fallback** chain:
      known user  -> blended (ALS + content + popularity)
      cold user   -> popularity
* ``similar_items(item_id, k)`` – content-based item-to-item, which also works for
  brand-new (**cold-start**) products that have no interaction history yet.

Scores from the three models are min-max normalised per user and blended with the weights
in ``config.BLEND``. Items the user already interacted with in training are masked out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .content_based import ContentModel
from .matrix_factorization import ImplicitALS
from .popularity import PopularityModel


def build_interaction_matrix(interactions: pd.DataFrame,
                             user_index: dict[str, int],
                             item_index: dict[str, int],
                             weights: dict[str, float]) -> sp.csr_matrix:
    """Weighted sparse (n_users x n_items) matrix; multiple interactions accumulate."""
    rows, cols, vals = [], [], []
    wmap = interactions["interaction_type"].map(weights).fillna(1.0).values
    u = interactions["user_id"].map(user_index).values
    i = interactions["item_id"].map(item_index).values
    ok = ~(pd.isna(u) | pd.isna(i))
    for uu, ii, vv in zip(u[ok].astype(int), i[ok].astype(int), wmap[ok]):
        rows.append(uu); cols.append(ii); vals.append(vv)
    R = sp.coo_matrix((vals, (rows, cols)),
                      shape=(len(user_index), len(item_index))).tocsr()
    return R


def _rowwise_minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    mn = a.min(axis=1, keepdims=True)
    mx = a.max(axis=1, keepdims=True)
    return (a - mn) / (mx - mn + 1e-9)


class UnifiedRecommender:
    def __init__(self, items: pd.DataFrame, users: pd.DataFrame,
                 weights: dict[str, float] | None = None):
        self.items = items.reset_index(drop=True)
        self.users = users.reset_index(drop=True)
        self.weights = weights or config.INTERACTION_WEIGHTS
        self.item_ids = self.items["item_id"].tolist()
        self.user_ids = self.users["user_id"].tolist()
        self.item_index = {iid: i for i, iid in enumerate(self.item_ids)}
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        # metadata lookups for explanations
        self.item_meta = self.items.set_index("item_id")[["category", "price"]].to_dict("index")

        self.als: ImplicitALS | None = None
        self.content: ContentModel | None = None
        self.popularity: PopularityModel | None = None

        self.R_train: sp.csr_matrix | None = None
        self.train_items: dict[int, set[int]] = {}
        self.blended_scores: np.ndarray | None = None
        self.component_scores: dict[str, np.ndarray] = {}

    # ---------------- fitting ----------------
    def fit(self, train_interactions: pd.DataFrame) -> "UnifiedRecommender":
        self.R_train = build_interaction_matrix(
            train_interactions, self.user_index, self.item_index, self.weights)

        # per-user seen items (masked out of recommendations)
        u = train_interactions["user_id"].map(self.user_index).values
        i = train_interactions["item_id"].map(self.item_index).values
        ok = ~(pd.isna(u) | pd.isna(i))
        self.train_items = {}
        for uu, ii in zip(u[ok].astype(int), i[ok].astype(int)):
            self.train_items.setdefault(uu, set()).add(ii)

        # 1) collaborative filtering (ALS on implicit feedback)
        self.als = ImplicitALS(factors=config.ALS_FACTORS, iterations=config.ALS_ITERATIONS,
                               reg=config.ALS_REG, alpha=config.ALS_ALPHA,
                               seed=config.RANDOM_STATE).fit(self.R_train)
        # 2) content-based
        self.content = ContentModel().fit(self.items)
        # 3) popularity baseline
        self.popularity = PopularityModel(self.weights).fit(train_interactions, self.items)

        # component + blended score matrices (recomputed on load to keep the artefact small)
        self.recompute_scores()
        return self

    def recompute_scores(self) -> "UnifiedRecommender":
        """(Re)build the dense component and blended score matrices from compact parts.

        The persisted artefact stores only the small factors/sparse matrices; these dense
        (n_users x n_items) matrices are cheap to rebuild on load.
        """
        s_als = self.als.score_all()
        s_content = self.content.score_all(self.R_train)
        s_pop = self.popularity.score_all(len(self.user_ids))
        self.component_scores = {"als": s_als, "content": s_content, "popularity": s_pop}

        w = config.BLEND
        self.blended_scores = (
            w["als"] * _rowwise_minmax(s_als)
            + w["content"] * _rowwise_minmax(s_content)
            + w["popularity"] * _rowwise_minmax(s_pop)
        )
        return self

    # ---------------- user profile helpers ----------------
    def _user_history(self, u: int) -> set[str]:
        cats = set()
        for i in self.train_items.get(u, set()):
            meta = self.item_meta.get(self.item_ids[i])
            if meta:
                cats.add(meta["category"])
        return cats

    def cold_start_item_ids(self) -> list[str]:
        """Items with zero training interactions — no collaborative signal yet,
        so only the content-based model can recommend them."""
        if self.R_train is None:
            return []
        totals = np.asarray(self.R_train.sum(axis=0)).ravel()
        return [self.item_ids[i] for i in range(len(self.item_ids)) if totals[i] == 0]

    # ---------------- recommendation ----------------
    def recommend(self, user_id: str, k: int = 10) -> dict:
        """Personalized top-K. Falls back to popularity for unknown/cold users."""
        u = self.user_index.get(user_id)
        cold = (u is None) or (len(self.train_items.get(u, set())) == 0)

        if cold:
            top = self.popularity.top_items(k)
            recs = [{
                "item_id": iid, "score": round(float(s), 4),
                "reason": "Popular right now (no personal history yet — cold start)",
                **self.item_meta.get(iid, {}),
            } for iid, s in top]
            return {"user_id": user_id, "strategy": "popularity (cold-start)",
                    "recommendations": recs}

        scores = self.blended_scores[u].astype(float).copy()
        seen = self.train_items.get(u, set())
        if seen:
            scores[np.fromiter(seen, int, len(seen))] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        cats = self._user_history(u)
        als_row = self.component_scores["als"][u]
        recs = []
        for i in top:
            iid = self.item_ids[i]
            meta = self.item_meta.get(iid, {})
            reasons = self._explain(iid, meta, cats, als_row[i])
            recs.append({"item_id": iid, "score": round(float(scores[i]), 4),
                         "reason": reasons, **meta})
        return {"user_id": user_id, "strategy": "hybrid (ALS + content + popularity)",
                "recommendations": recs}

    def _explain(self, item_id: str, meta: dict, user_cats: set,
                 als_score: float) -> str:
        bits = []
        if meta.get("category") in user_cats:
            bits.append(f"in your favourite category '{meta.get('category')}'")
        if not bits:
            bits.append("content-similar to items you liked")
        if als_score > np.median(self.component_scores["als"]):
            bits.append("users like you also engaged with it")
        reason = "; ".join(bits[:2])
        return reason[:1].upper() + reason[1:] if reason else reason

    def similar_items(self, item_id: str, k: int = 10) -> list[dict]:
        """Content-based item-to-item similarity — works for cold-start items too."""
        i = self.item_index.get(item_id)
        if i is None:
            return []
        sims = self.content.item_similarity(i, k)
        out = []
        for j, s in sims:
            iid = self.item_ids[j]
            out.append({"item_id": iid, "similarity": round(s, 4),
                        **self.item_meta.get(iid, {})})
        return out

    def segment_users(self) -> pd.DataFrame:
        """Assign each user to a segment for the analysis view (step 6)."""
        inter_w = self.R_train.copy()
        total = np.asarray(inter_w.sum(axis=1)).ravel()
        n_items = np.diff(inter_w.indptr)
        seg = np.where(total >= np.quantile(total[total > 0], 0.66), "Heavy",
                       np.where(total >= np.quantile(total[total > 0], 0.33), "Medium", "Light"))
        seg = np.where(n_items == 0, "Cold", seg)
        # dominant preferred category per user
        dom_cat = []
        for u in range(len(self.user_ids)):
            cats = self._user_history(u)
            dom_cat.append(sorted(cats)[0] if cats else "n/a")
        return pd.DataFrame({
            "user_id": self.user_ids, "segment": seg,
            "n_items_seen": n_items, "engagement_weight": np.round(total, 1),
            "dominant_category": dom_cat,
        })


def load_recommender(path=None) -> UnifiedRecommender:
    """Load a persisted recommender and rebuild its dense score matrices."""
    import joblib
    path = path or config.MODEL_PATH
    rec = joblib.load(path)
    if getattr(rec, "blended_scores", None) is None:
        rec.recompute_scores()
    return rec
