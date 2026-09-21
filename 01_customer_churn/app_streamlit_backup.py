"""Streamlit UI for the Customer Churn project (deployment layer, step 6 of the PDF).

Shows model performance, the ranked list of high-risk customers with suggested review
reasons, a single-customer risk explainer, and the evaluation plots.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.features import load_and_engineer
from src.train import _reasons_for_row

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not config.BEST_MODEL_PATH.exists():
        return None
    return joblib.load(config.BEST_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_engineered():
    if not config.CUSTOMERS_CSV.exists():
        return None
    return load_and_engineer()


@st.cache_data(show_spinner=False)
def load_metrics():
    if not config.METRICS_JSON.exists():
        return None
    with open(config.METRICS_JSON) as f:
        return json.load(f)


pipeline = load_pipeline()
metrics = load_metrics()

st.title("📉 Customer Churn Prediction")
st.caption("Predict which customers are likely to leave so retention teams can act first.")

if pipeline is None or metrics is None:
    st.warning(
        "No trained model found. Run the pipeline first:\n\n"
        "```\npython -m src.generate_data\npython -m src.train\n```"
    )
    st.stop()

pre = pipeline["preprocessor"]
model = pipeline["model"]
threshold = pipeline["threshold"]
df = load_engineered()

tab_overview, tab_risk, tab_single, tab_plots = st.tabs(
    ["📊 Overview", "🚨 High-Risk Customers", "🔎 Score a Customer", "🖼️ Plots"]
)

# ---------------- Overview ----------------
with tab_overview:
    best = metrics["best_model"]
    table = pd.DataFrame(metrics["metrics_table"])
    c1, c2, c3, c4 = st.columns(4)
    brow = table[table["name"] == best].iloc[0]
    c1.metric("Best model", best)
    c2.metric("ROC-AUC", f"{brow['roc_auc']:.3f}")
    c3.metric("Recall @ threshold", f"{brow['recall']:.3f}")
    c4.metric("Decision threshold", f"{threshold:.2f}")

    st.subheader("Model comparison")
    st.dataframe(
        table.style.format({
            "roc_auc": "{:.3f}", "pr_auc": "{:.3f}", "precision": "{:.3f}",
            "recall": "{:.3f}", "f1": "{:.3f}", "accuracy": "{:.3f}",
            "threshold": "{:.2f}", "brier": "{:.3f}",
        }).background_gradient(subset=["roc_auc", "pr_auc", "f1"], cmap="RdYlGn"),
        width="stretch",
    )
    st.write(
        f"Scored **{metrics['n_customers_scored']:,}** customers; "
        f"**{metrics['n_high_risk_flagged']:,}** flagged as high-risk "
        f"(probability ≥ 0.5)."
    )

# ---------------- High-risk list ----------------
with tab_risk:
    st.subheader("Ranked high-risk customers")
    st.caption("Highest churn probability first, with suggested review reasons for the retention team.")
    if config.RANKED_LIST_CSV.exists():
        ranked = pd.read_csv(config.RANKED_LIST_CSV)
        band = st.multiselect("Filter by risk band",
                              sorted(ranked["risk_band"].astype(str).unique()),
                              default=sorted(ranked["risk_band"].astype(str).unique()))
        view = ranked[ranked["risk_band"].astype(str).isin(band)]
        st.dataframe(view, width="stretch", height=520)
        st.download_button("⬇️ Download ranked list (CSV)",
                           ranked.to_csv(index=False).encode(),
                           file_name="high_risk_customers.csv", mime="text/csv")
    else:
        st.info("Run `python -m src.train` to generate the ranked list.")

# ---------------- Single-customer explainer ----------------
with tab_single:
    st.subheader("Score an individual customer")
    if df is None:
        st.info("No data found. Run `python -m src.generate_data`.")
    else:
        ids = df["customer_id"].tolist()
        cid = st.selectbox("Customer ID", ids, index=0)
        row = df[df["customer_id"] == cid].iloc[0]
        X = pre.transform(df[df["customer_id"] == cid])
        prob = float(model.predict_proba(X)[0, 1])
        band = "Critical" if prob >= 0.7 else "High" if prob >= 0.5 else "Medium" if prob >= 0.3 else "Low"
        colr = {"Critical": "error", "High": "error", "Medium": "warning", "Low": "success"}[band]

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{prob:.1%}")
        c2.metric("Risk band", band, delta_color="off")
        c3.metric("Above threshold?", "Yes" if prob >= threshold else "No")
        st.markdown(f"**Status:** :{colr}[{'INTERVENE — retention review recommended' if prob >= threshold else 'Monitor'}]")

        st.markdown("**Suggested review reasons**")
        for r in _reasons_for_row(row, {}):
            st.markdown(f"- {r}")

        with st.expander("Raw & engineered features for this customer"):
            show = row.dropna().to_frame("value").T
            st.dataframe(show, width="stretch")

# ---------------- Plots ----------------
with tab_plots:
    plots = [
        ("ROC curves", config.ROC_PNG),
        ("Precision-Recall curves", config.PR_PNG),
        ("Calibration", config.CALIBRATION_PNG),
        ("Confusion matrix (best model)", config.CONFUSION_PNG),
        ("Feature importance", config.FEATURE_IMPORTANCE_PNG),
    ]
    for title, path in plots:
        if Path(path).exists():
            st.subheader(title)
            st.image(str(path), width="stretch")
