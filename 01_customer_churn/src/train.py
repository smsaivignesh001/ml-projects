"""End-to-end training pipeline for the Customer Churn project (steps 4-6 of the PDF).

  4. Train a baseline Logistic Regression and tree-based models (Random Forest, XGBoost).
  5. Compare ROC-AUC, precision, recall, F1 and calibration.
  6. Persist the best model and write a ranked list of high-risk customers with
     suggested review reasons.

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
from .features import load_and_engineer
from .preprocessing import (
    ID_COL, TARGET, build_preprocessor, imbalance_ratio, train_val_split,
)

warnings.filterwarnings("ignore")


def build_models(scale_pos_weight: float) -> dict[str, object]:
    """Models with class-imbalance handling baked in."""
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
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


def _reasons_for_row(row: pd.Series, benchmarks: dict) -> list[str]:
    """Human-readable drivers that flag this customer for retention review."""
    reasons: list[str] = []
    if row.get("contract_type") == "Month-to-month":
        reasons.append("Month-to-month contract (no lock-in)")
    if row.get("tenure_months", 99) <= 6:
        reasons.append("New customer (tenure <= 6 months)")
    trend = row.get("usage_trend", 0) or 0
    if trend <= -0.3:
        reasons.append(f"Usage down {abs(trend):.0%} recently")
    elif trend <= -0.1:
        reasons.append(f"Usage declining ({trend:.0%})")
    st = row.get("support_tickets_last_90d", 0) or 0
    if st >= 3:
        reasons.append(f"High support load ({int(st)} tickets / 90d)")
    elif st >= 1:
        reasons.append(f"{int(st)} support ticket(s) / 90d")
    lp = row.get("late_payments_last_6m", 0) or 0
    if lp >= 1:
        reasons.append(f"{int(lp)} late payment(s) in last 6mo")
    if not reasons:
        reasons.append("No single dominant driver — general review")
    return reasons[:4]


def build_ranked_list(df: pd.DataFrame, y_prob: np.ndarray, top_n: int = 200) -> pd.DataFrame:
    """Rank all customers by churn risk and attach suggested review reasons."""
    out = df.copy()
    out["churn_probability"] = np.round(y_prob, 4)
    out["risk_band"] = pd.cut(
        out["churn_probability"], bins=[-0.01, 0.3, 0.5, 0.7, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    )
    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    out["risk_rank"] = np.arange(1, len(out) + 1)
    out = out.head(top_n).copy()

    benchmarks = {}  # reserved for future percentile-based explanations
    out["suggested_review_reasons"] = [
        " | ".join(_reasons_for_row(r, benchmarks)) for _, r in out.iterrows()
    ]
    cols = [
        "risk_rank", ID_COL, "churn_probability", "risk_band", "contract_type",
        "tenure_months", "monthly_charges", "usage_trend", "support_tickets_last_90d",
        "late_payments_last_6m", "suggested_review_reasons",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


def main() -> None:
    config.ensure_dirs()
    print(">> Loading & engineering features ...")
    df = load_and_engineer()
    train_df, val_df = train_val_split(df, test_size=0.2, seed=config.RANDOM_STATE)
    print(f"   train={len(train_df)}  val={len(val_df)}  churn_rate={df[TARGET].mean():.3f}")

    print(">> Fitting preprocessor on training fold ...")
    pre, num_cols, cat_cols = build_preprocessor(train_df)
    X_train = pre.fit_transform(train_df)
    X_val = pre.transform(val_df)
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    feature_names = list(pre.get_feature_names_out())
    spw = imbalance_ratio(y_train)
    print(f"   features={X_train.shape[1]}  scale_pos_weight={spw:.2f}")

    models = build_models(spw)
    probs: dict[str, np.ndarray] = {}
    metrics_list: list[ev.ModelMetrics] = []
    fitted: dict[str, object] = {}

    print(">> Training models ...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        probs[name] = prob
        fitted[name] = model
        m, _ = ev.evaluate_model(name, y_val, prob)
        metrics_list.append(m)
        print(f"   {name:18s} ROC-AUC={m.roc_auc:.3f} PR-AUC={m.pr_auc:.3f} "
              f"F1={m.f1:.3f} Recall={m.recall:.3f} Precision={m.precision:.3f} "
              f"Brier={m.brier:.3f}")

    # ---- compare + pick best (ROC-AUC primary, tie-break by F1) --------------
    metrics_df = ev.metrics_dataframe(metrics_list)
    best_name = metrics_df.iloc[0]["name"]
    best_model = fitted[best_name]
    print(f">> Best model: {best_name}")

    # ---- feature importances (from best tree model, else logistic coefs) -----
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
    else:
        # Models like HistGradientBoostingClassifier expose neither — fall
        # back to permutation importance on the held-out split.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            best_model, X_val, y_val, n_repeats=5,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        importances = pd.Series(perm.importances_mean, index=feature_names)
    ev.plot_feature_importance(importances, config.FEATURE_IMPORTANCE_PNG)

    # ---- plots ---------------------------------------------------------------
    ev.plot_roc_curves(probs, y_val, config.ROC_PNG)
    ev.plot_pr_curves(probs, y_val, config.PR_PNG)
    ev.plot_calibration(y_val, probs, config.CALIBRATION_PNG)
    best_metrics = next(m for m in metrics_list if m.name == best_name)
    y_pred = (probs[best_name] >= best_metrics.threshold).astype(int)
    ev.plot_confusion(y_val, y_pred, f"Confusion Matrix — {best_name}", config.CONFUSION_PNG)

    # ---- persist the best model + preprocessor ------------------------------
    pipeline = {"preprocessor": pre, "model": best_model,
                "feature_names": feature_names, "threshold": best_metrics.threshold}
    joblib.dump(pipeline, config.BEST_MODEL_PATH)
    joblib.dump(pre, config.PREPROCESSOR_PATH)
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    model_card = {
        "task": "Customer churn prediction (binary classification)",
        "best_model": best_name,
        "threshold": best_metrics.threshold,
        "n_features": len(feature_names),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "class_imbalance_scale_pos_weight": spw,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(config.MODEL_CARD_PATH, "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    # ---- ranked high-risk list over the FULL dataset ------------------------
    X_all = pre.transform(df)
    prob_all = best_model.predict_proba(X_all)[:, 1]
    ranked = build_ranked_list(df, prob_all, top_n=200)
    ranked.to_csv(config.RANKED_LIST_CSV, index=False)

    # ---- metrics.json --------------------------------------------------------
    with open(config.METRICS_JSON, "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics_table": metrics_df.to_dict(orient="records"),
            "val_churn_rate": float(y_val.mean()),
            "n_customers_scored": int(len(df)),
            "n_high_risk_flagged": int((prob_all >= 0.5).sum()),
        }, f, indent=2, default=str)

    print(">> Saved artefacts:")
    for p in [config.BEST_MODEL_PATH, config.METRICS_JSON, config.RANKED_LIST_CSV,
              config.ROC_PNG, config.PR_PNG, config.CALIBRATION_PNG,
              config.FEATURE_IMPORTANCE_PNG, config.CONFUSION_PNG]:
        print("   ", p.name)
    print("\nTop 5 high-risk customers:")
    print(ranked.head(5)[["risk_rank", ID_COL, "churn_probability", "risk_band",
                          "suggested_review_reasons"]].to_string(index=False))


if __name__ == "__main__":
    main()
