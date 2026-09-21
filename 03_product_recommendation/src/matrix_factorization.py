"""Implicit-feedback matrix factorization via Alternating Least Squares (step 3).

Implements the weighted-ALS model of Hu, Koren & Volinsky (2008), "Collaborative
Filtering for Implicit Feedback Problems". This is the standard, strong collaborative
filtering baseline for implicit data (views/clicks/carts/purchases rather than ratings):

    minimise  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda (||X||^2 + ||Y||^2)

with preference p_ui = 1 for any observed interaction and confidence c_ui = 1 + alpha*r_ui,
where r_ui is the interaction weight (purchase > cart > click > view). Each ALS sweep has a
closed-form per-user / per-item solve, so it is fast and needs no gradient tuning.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _als_step(R: sp.csr_matrix, Y: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """Solve for the row factors X given fixed column factors Y (one ALS half-step).

    R: sparse (m x n) weighted interactions.  Y: (n x f) fixed factors.  Returns X: (m x f).
    For each row u:
        x_u = (Y^T Y + reg I + Y_u^T (c_u - I) Y_u)^-1  *  (Y_u^T c_u)
    where c_ui = 1 + alpha * r_ui and p_ui = 1 on observed entries.
    """
    m = R.shape[0]
    f = Y.shape[1]
    YtY = Y.T @ Y + reg * np.eye(f)
    X = np.zeros((m, f), dtype=np.float64)
    indptr, indices, data = R.indptr, R.indices, R.data
    for u in range(m):
        s, e = indptr[u], indptr[u + 1]
        if e == s:
            continue
        idx = indices[s:e]
        vals = data[s:e]
        Yu = Y[idx]                       # (k, f)
        cu = 1.0 + alpha * vals           # (k,)
        A = YtY + Yu.T @ (Yu * (cu - 1.0)[:, None])
        b = Yu.T @ cu                     # sum_i c_ui * y_i
        X[u] = np.linalg.solve(A, b)
    return X


class ImplicitALS:
    def __init__(self, factors: int = 32, iterations: int = 15,
                 reg: float = 0.05, alpha: float = 8.0, seed: int = 42):
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.alpha = alpha
        self.seed = seed

    def fit(self, R: sp.spmatrix) -> "ImplicitALS":
        R = sp.csr_matrix(R, dtype=np.float64)
        Rt = sp.csr_matrix(R.T)
        n_users, n_items = R.shape
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))
        for _ in range(self.iterations):
            self.user_factors = _als_step(R, self.item_factors, self.reg, self.alpha)
            self.item_factors = _als_step(Rt, self.user_factors, self.reg, self.alpha)
        return self

    def score_all(self) -> np.ndarray:
        """Dense (n_users x n_items) preference score matrix."""
        return self.user_factors @ self.item_factors.T

    def score_user(self, u: int) -> np.ndarray:
        return self.user_factors[u] @ self.item_factors.T
