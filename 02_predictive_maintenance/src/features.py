"""Feature engineering + labelling for Predictive Maintenance (steps 2 & 3 of the PDF).

* Step 2 — rolling statistics and lag features computed with **backward-looking windows
  only**, so no future reading ever leaks into a feature.
* Step 3 — a failure **prediction window**: each observation is labelled 1 if the machine
  fails within the next ``PREDICTION_WINDOW_HOURS``, else 0.

Backend data source: a single flat, hourly ``sensor_data.csv`` with an inline
``failure_event`` flag (1 on the hour a failure happens) rather than a separate
failures table — the label logic below reconstructs the same "time to next failure"
signal zip1's design used, just from this row-level flag.

The label legitimately uses the future (that is the target); the features never do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SENSOR_COLS = ["temperature_c", "vibration_mm_s", "pressure_kpa"]
# rolling windows / lags expressed in hours (data cadence = 1 reading/hour)
ROLL_WINDOWS = [6, 12, 24]
LAGS = [1, 2, 3]


def load_sensor_data() -> pd.DataFrame:
    df = pd.read_csv(config.SENSOR_CSV)
    return df.sort_values(["machine_id", "hour"]).reset_index(drop=True)


def _label_machine(g: pd.DataFrame, window: int) -> pd.DataFrame:
    g = g.sort_values("hour")
    hours = g["hour"].to_numpy()
    fail_hours = hours[g["failure_event"].to_numpy() == 1]
    if fail_hours.size == 0:
        ttf = np.full(len(hours), np.inf)
    else:
        idx = np.searchsorted(fail_hours, hours, side="left")
        ttf = np.full(len(hours), np.inf)
        valid = idx < fail_hours.size
        ttf[valid] = fail_hours[idx[valid]] - hours[valid]
    g = g.copy()
    g["time_to_failure_h"] = ttf
    g["label"] = ((ttf >= 0) & (ttf <= window)).astype(int)
    return g


def add_labels(df: pd.DataFrame, window_hours: int | None = None) -> pd.DataFrame:
    """Label each reading: 1 if a failure_event occurs within the next window hours."""
    window = window_hours or config.PREDICTION_WINDOW_HOURS
    df = df.sort_values(["machine_id", "hour"]).reset_index(drop=True)
    parts = [_label_machine(g, window) for _, g in df.groupby("machine_id", sort=False)]
    return pd.concat(parts).sort_values(["machine_id", "hour"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking rolling stats, lags, deltas and trend features per machine."""
    df = df.sort_values(["machine_id", "hour"]).reset_index(drop=True)
    out = df.copy()
    grouped = {c: out.groupby("machine_id", sort=False)[c] for c in SENSOR_COLS}

    for col in SENSOR_COLS:
        g = grouped[col]
        for w in ROLL_WINDOWS:
            out[f"{col}_rmean{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            out[f"{col}_rstd{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).std()).fillna(0.0)
            out[f"{col}_rmax{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).max())
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = g.transform(lambda s, lag=lag: s.shift(lag))
        out[f"{col}_delta1"] = out[col] - out[f"{col}_lag1"]
        out[f"{col}_slope6"] = g.transform(
            lambda s: s.rolling(6, min_periods=2).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 2 else 0.0,
                raw=True,
            )
        )
        out[f"{col}_dev24"] = out[col] - out[f"{col}_rmean24"]

    # cross-sensor interactions that often precede failure
    out["temp_x_vib"] = out["temperature_c"] * out["vibration_mm_s"]
    out["vib_over_pressure"] = out["vibration_mm_s"] / out["pressure_kpa"].replace(0, np.nan)
    out["operating_hours"] = df["operating_hours"]
    out["hours_since_maintenance"] = df["hours_since_maintenance"]

    # drop lags that are NaN at the very start of each machine's history
    out = out.dropna(subset=[f"{SENSOR_COLS[0]}_lag{LAGS[-1]}"]).reset_index(drop=True)
    return out


def build_dataset(window_hours: int | None = None) -> pd.DataFrame:
    raw = load_sensor_data()
    labelled = add_labels(raw, window_hours)
    featured = add_features(labelled)
    return featured


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns (exclude ids, target, and leakage-only helpers)."""
    exclude = {"machine_id", "hour", "failure_event", "label", "time_to_failure_h"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = build_dataset()
    pos = df["label"].mean()
    print("Dataset:", df.shape)
    print(f"Positive (failure-within-window) rate: {pos:.4f}  "
          f"-> imbalance 1:{(1 - pos) / max(pos, 1e-9):.0f}")
    print("Feature columns:", len(feature_columns(df)))
    print(df[["machine_id", "hour", "temperature_c", "vibration_mm_s",
              "time_to_failure_h", "label"]].head(8).to_string(index=False))
