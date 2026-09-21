"""Content-based recommender using item metadata (step 4 — cold-start fallback).

Builds a TF-IDF representation of each item from its metadata (category, brand, tags,
price band). A user profile is the interaction-weighted average of the items they engaged
with; recommendations are the items whose content vector is closest to that profile.

Because every item — including brand-new products with **no interaction history** — has a
content vector, this model can recommend cold-start items and is the natural fallback when
collaborative filtering has no signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _price_band(price: float) -> str:
    if price < 20:
        return "price_low"
    if price < 60:
        return "price_mid"
    if price < 150:
        return "price_high"
    return "price_premium"


def item_documents(items: pd.DataFrame) -> pd.Series:
    """Turn each item's metadata into a single bag-of-terms string.

    The backend catalogue only carries category and price (no brand/tags), so the
    content vector is built from those — category is emphasised (repeated) since
    it is the strongest available signal.
    """
    docs = (
        items["category"].astype(str) + " "
        + items["category"].astype(str) + " "          # emphasise category
        + items["price"].apply(_price_band).astype(str)
    )
    return docs


class ContentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"[a-zA-Z0-9_]+",
            sublinear_tf=True, norm="l2",
        )
        self.item_matrix = None      # (n_items x vocab), l2-normalised rows
        self.item_ids = None

    def fit(self, items: pd.DataFrame) -> "ContentModel":
        self.item_ids = items["item_id"].tolist()
        docs = item_documents(items)
        self.item_matrix = self.vectorizer.fit_transform(docs)
        self.item_matrix = normalize(self.item_matrix, norm="l2")  # cosine = dot product
        return self

    def user_profiles(self, R: sp.spmatrix) -> np.ndarray:
        """Weighted-average content profile per user.

        R: sparse (n_users x n_items) of interaction weights. Profile = normalise(R @ T).
        """
        P = R @ self.item_matrix                 # (n_users x vocab)
        P = normalize(P, norm="l2")
        return np.asarray(P.todense())

    def score_all(self, R: sp.spmatrix) -> np.ndarray:
        """Dense (n_users x n_items) content-similarity scores."""
        profiles = self.user_profiles(R)          # (n_users x vocab)
        T = np.asarray(self.item_matrix.todense())  # (n_items x vocab)
        return profiles @ T.T

    def item_similarity(self, item_index: int, k: int = 10,
                        exclude_self: bool = True) -> list[tuple[int, float]]:
        """k most content-similar items to ``item_index`` (cosine similarity)."""
        sims = np.asarray((self.item_matrix @ self.item_matrix[item_index].T).todense()).ravel()
        if exclude_self:
            sims[item_index] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]
