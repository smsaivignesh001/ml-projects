"""Preprocessing for Predictive Maintenance.

Sensor features are numeric. We impute the few NaNs left by warm-up windows (median),
winsorise extremes, and standard-scale so the logistic-regression baseline is well
behaved. Class imbalance (rare failures) is handled in the models via class weights /
scale_pos_weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = {"machine_id", "hour", "failure_event", "label", "time_to_failure_h"}


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.limits_ = np.vstack([np.nanquantile(X, self.lower, axis=0),
                                  np.nanquantile(X, self.upper, axis=0)])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = self.limits_
        return np.clip(X, lo, hi)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.limits_.shape[1])], dtype=object)
        return np.asarray(input_features, dtype=object)


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN so imputation can handle them."""
    return df.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer()),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("num", num_pipe, feature_cols)], remainder="drop")


def imbalance_ratio(y: np.ndarray | pd.Series) -> float:
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    return int(len(y) - pos) / pos


def time_based_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Chronological split across the whole fleet (no future leakage into training)."""
    d = df.sort_values("hour").reset_index(drop=True)
    cut = int(len(d) * (1 - test_frac))
    return d.iloc[:cut].reset_index(drop=True), d.iloc[cut:].reset_index(drop=True)
