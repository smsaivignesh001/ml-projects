"""Feature engineering for the churn project.

Implements step 3 of the PDF: *"Create features such as usage trend, support-contact
frequency, and recent payment changes."*

The backend data source is a single flat ``customers.csv`` table (no separate monthly
usage-history series), so all engineered features are derived directly from columns on
that table. Everything here is a pure function of the row's own data — no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Add business-derived signals to the raw customer table.

    * ``usage_trend``                – relative change in data usage over the last 30
      days vs. the prior 30 days (the usage-trend feature called out in the brief).
    * ``support_contact_frequency``  – support tickets per month of tenure.
    * ``recent_payment_change``      – flag for a late payment in the last 6 months.
    * ``charges_per_tenure_month``   – total charges normalised by tenure.
    * ``tenure_group``               – bucketed tenure, used as a categorical feature.
    """
    df = customers.copy()

    # --- usage trend ---------------------------------------------------------
    df["usage_trend"] = (
        (df["usage_last_30d_gb"] - df["usage_prev_30d_gb"]) / (df["usage_prev_30d_gb"] + 1)
    )

    # --- support-contact frequency --------------------------------------------
    df["support_contact_frequency"] = df["support_tickets_last_90d"] / df["tenure_months"].clip(lower=1)
    df["high_support_burden"] = (df["support_tickets_last_90d"] >= 2).astype(int)

    # --- recent payment changes -----------------------------------------------
    df["recent_payment_change"] = (df["late_payments_last_6m"] > 0).astype(int)
    df["charges_per_tenure_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)

    # --- tenure & engagement buckets -------------------------------------------
    df["tenure_group"] = pd.cut(
        df["tenure_months"], bins=[-1, 6, 12, 24, 48, 120],
        labels=["0-6", "7-12", "13-24", "25-48", "48+"],
    )

    # --- interaction features ---------------------------------------------------
    df["new_month_to_month"] = (
        (df["tenure_months"] <= 6) & (df["contract_type"] == "Month-to-month")
    ).astype(int)

    # --- outlier capping (step 2): winsorize numeric cols at the 1st/99th pct ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
        ["customer_id", "churn"], errors="ignore"
    )
    for col in numeric_cols:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    return df


def load_and_engineer() -> pd.DataFrame:
    """Convenience loader used by training and the app."""
    from . import config

    customers = pd.read_csv(config.CUSTOMERS_CSV)
    return engineer_features(customers)


if __name__ == "__main__":
    df = load_and_engineer()
    print("Engineered feature table:", df.shape)
    print(df.head().T)
