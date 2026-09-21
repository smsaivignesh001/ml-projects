"""Training pipeline for Predictive Maintenance (steps 4-6 of the PDF).

  4. Train classifiers; handle rare failures with class weights / scale_pos_weight.
  5. Evaluate recall, precision, PR-AUC and the number of false alarms.
  6. Produce a machine-level risk snapshot used by the monitoring dashboard.

Run:
    python -m src.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover - environment without xgboost installed
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_XGB = False

from . import config
from . import evaluate as ev
from .features import build_dataset, feature_columns
from .preprocessing import (
    build_preprocessor, imbalance_ratio, replace_inf, time_based_split,
)

warnings.filterwarnings("ignore")


def build_models(spw: float) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=3000, C=0.5,
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0,
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=config.RANDOM_STATE, tree_method="hist",
        )
    else:
        # xgboost isn't installed in this environment — sklearn's histogram
        # gradient booster is the closest drop-in; install xgboost>=2.0 and
        # rerun to use the original model.
        models["GradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.06,
            random_state=config.RANDOM_STATE,
        )
    return models


def risk_snapshot(df: pd.DataFrame, pre, model, threshold: float) -> pd.DataFrame:
    """Latest observation per machine -> current risk score + alert status."""
    df = df.sort_values("hour")
    latest = df.groupby("machine_id", as_index=False).tail(1).copy()
    X = pre.transform(replace_inf(latest[feature_columns(latest)]))
    latest["risk_score"] = np.round(model.predict_proba(X)[:, 1], 4)
    latest["time_to_failure_h"] = latest["time_to_failure_h"].replace([np.inf], np.nan).round(1)
    latest["alert"] = np.where(
        latest["risk_score"] >= threshold, "🔴 ALERT",
        np.where(latest["risk_score"] >= threshold * 0.6, "🟠 WATCH", "🟢 OK"),
    )
    cols = ["machine_id", "hour", "risk_score", "alert", "temperature_c",
            "vibration_mm_s", "pressure_kpa", "hours_since_maintenance"]
    cols = [c for c in cols if c in latest.columns]
    return latest[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    print(">> Building labelled feature dataset ...")
    df = replace_inf(build_dataset())
    feats = feature_columns(df)
    pos_rate = df["label"].mean()
    print(f"   rows={len(df):,}  features={len(feats)}  positive_rate={pos_rate:.4f} "
          f"(1:{(1 - pos_rate) / max(pos_rate, 1e-9):.0f})")

    train_df, test_df = time_based_split(df, test_frac=0.25)
    print(f"   chronological split -> train={len(train_df):,} ({train_df['label'].mean():.4f}) "
          f"test={len(test_df):,} ({test_df['label'].mean():.4f})")

    pre = build_preprocessor(feats)
    X_train = pre.fit_transform(train_df[feats])
    X_test = pre.transform(test_df[feats])
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    spw = imbalance_ratio(y_train)
    print(f"   scale_pos_weight={spw:.1f}")

    models = build_models(spw)
    probs, metrics_list, fitted = {}, [], {}
    print(">> Training ...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        probs[name] = prob
        fitted[name] = model
        m, _ = ev.evaluate_model(name, y_test, prob)
        metrics_list.append(m)
        print(f"   {name:18s} PR-AUC={m.pr_auc:.3f} Recall={m.recall:.3f} "
              f"Precision={m.precision:.3f} F1={m.f1:.3f} "
              f"FalseAlarms={m.false_positives} Missed={m.false_negatives}")

    metrics_df = ev.metrics_dataframe(metrics_list)
    best_name = metrics_df.iloc[0]["name"]
    best_model = fitted[best_name]
    best_metrics = next(m for m in metrics_list if m.name == best_name)
    print(f">> Best model: {best_name} (by PR-AUC)")

    # feature importances
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feats)
    elif hasattr(best_model, "coef_"):
        imp = pd.Series(np.abs(best_model.coef_[0]), index=feats)
    else:
        # Models like HistGradientBoostingClassifier expose neither — fall
        # back to permutation importance on the held-out split.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            best_model, X_test, y_test, n_repeats=5,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        imp = pd.Series(perm.importances_mean, index=feats)
    ev.plot_feature_importance(imp, config.FEATURE_IMPORTANCE_PNG)

    # plots
    ev.plot_pr_curves(probs, y_test, config.PR_PNG)
    ev.plot_roc_curves(probs, y_test, config.ROC_PNG)
    ev.plot_false_alarm_tradeoff(y_test, probs[best_name], config.FALSE_ALARM_PNG, best_name)
    y_pred = (probs[best_name] >= best_metrics.threshold).astype(int)
    ev.plot_confusion(y_test, y_pred, f"Confusion — {best_name}", config.CONFUSION_PNG)

    # persist
    joblib.dump({"preprocessor": pre, "model": best_model, "feature_names": feats,
                 "threshold": best_metrics.threshold}, config.BEST_MODEL_PATH)
    joblib.dump(pre, config.PREPROCESSOR_PATH)
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feats, f, indent=2)

    # risk snapshot over the most recent data (for the monitoring dashboard)
    snap = risk_snapshot(df, pre, best_model, best_metrics.threshold)
    snap.to_csv(config.RISK_SNAPSHOT_CSV, index=False)

    # full risk timeline (precomputed so the dashboard drill-down is instant)
    print(">> Scoring full history for the monitoring timeline ...")
    X_all = pre.transform(replace_inf(df[feats]))
    timeline = df[["machine_id", "hour", "label", "time_to_failure_h", "failure_event",
                   "temperature_c", "vibration_mm_s", "pressure_kpa",
                   "hours_since_maintenance"]].copy()
    timeline["risk_score"] = np.round(best_model.predict_proba(X_all)[:, 1], 4)
    timeline["time_to_failure_h"] = timeline["time_to_failure_h"].replace([np.inf], np.nan)
    timeline.to_csv(config.RISK_TIMELINE_CSV, index=False)

    with open(config.METRICS_JSON, "w") as f:
        json.dump({
            "best_model": best_name,
            "prediction_window_hours": config.PREDICTION_WINDOW_HOURS,
            "positive_rate": float(pos_rate),
            "metrics_table": metrics_df.to_dict(orient="records"),
            "n_machines": int(df["machine_id"].nunique()),
            "n_alerts": int((snap["alert"] == "🔴 ALERT").sum()),
            "n_watch": int((snap["alert"] == "🟠 WATCH").sum()),
        }, f, indent=2, default=str)

    with open(config.MODEL_CARD_PATH, "w") as f:
        json.dump({
            "task": "Predict machine failure within a time window",
            "prediction_window_hours": config.PREDICTION_WINDOW_HOURS,
            "best_model": best_name,
            "threshold": best_metrics.threshold,
            "n_features": len(feats),
            "class_imbalance_scale_pos_weight": spw,
            "metrics": metrics_df.to_dict(orient="records"),
        }, f, indent=2, default=str)

    print(">> Saved artefacts to models/ and outputs/")
    print("\nMachine risk snapshot (top 6):")
    print(snap.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
