"""Preprocessing for the churn project (step 2 of the PDF).

Handles:
  * missing values  – median imputation for numerics, a dedicated ``Unknown`` category
                      for categoricals,
  * outliers        – winsorising (clipping) numeric columns to the 1st/99th percentiles
                      learned on the training fold only,
  * categorical vars – one-hot encoding,
  * class imbalance  – handled downstream via ``class_weight='balanced'`` and XGBoost
                       ``scale_pos_weight`` (see ``train.py``); this module exposes the
                       ratio helper.

A single ``ColumnTransformer`` is returned so the exact same transforms are applied to
training, validation, and live inference data (no leakage, no drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "customer_id"
TARGET = "churn"

CATEGORICAL_COLS = [
    "gender", "region", "contract_type", "payment_method", "tenure_group",
]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to their learned [q_low, q_high] percentiles.

    Fit on training data only, then applied to any data — removes the effect of extreme
    outliers (e.g. the corrupted ``monthly_charges`` rows) without dropping records.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.limits_ = np.vstack([
            np.nanquantile(X, self.lower, axis=0),
            np.nanquantile(X, self.upper, axis=0),
        ])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = self.limits_
        return np.clip(X, lo, hi)

    def get_feature_names_out(self, input_features=None):
        # Pass-through: winsorising does not change the feature set.
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.limits_.shape[1])], dtype=object)
        return np.asarray(input_features, dtype=object)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in CATEGORICAL_COLS + [ID_COL, TARGET]]
    # keep only numeric dtypes; tenure_group is categorical
    return [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Return a ColumnTransformer plus the numeric/categorical feature name lists."""
    num_cols = numeric_columns(df)
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")
    return pre, num_cols, cat_cols


def imbalance_ratio(y: np.ndarray | pd.Series) -> float:
    """scale_pos_weight = n_negative / n_positive, for tree models."""
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    neg = int(len(y) - pos)
    return neg / pos


def train_val_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
