"""Streamlit UI for the Product Recommendation project (step 6 of the PDF).

Explore personalized recommendations, the content-based cold-start fallback, the offline
ranking metrics, and how recommendations differ across user segments.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.recommender import load_recommender

st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Loading recommender...")
def get_rec():
    if not config.MODEL_PATH.exists():
        return None
    return load_recommender(config.MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics():
    if not config.METRICS_JSON.exists():
        return None
    with open(config.METRICS_JSON) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_items():
    if not config.ITEMS_CSV.exists():
        return None
    return pd.read_csv(config.ITEMS_CSV).rename(columns={"product_id": "item_id"})


rec, metrics, items = get_rec(), get_metrics(), get_items()

st.title("🛒 Personalized Product Recommendations")
st.caption("Hybrid recommender — ALS collaborative filtering + content-based + popularity, "
           "with a cold-start fallback for new products and users.")

if rec is None or metrics is None:
    st.warning("No trained model found. Run:\n\n```\npython -m src.train\n```")
    st.stop()

tab_rec, tab_similar, tab_perf, tab_seg, tab_plots = st.tabs(
    ["🎯 Recommend", "🔗 Similar / Cold-start", "📊 Model Performance", "👥 Segments", "🖼️ Plots"]
)

# ---------------- Recommend ----------------
with tab_rec:
    st.subheader("Personalized recommendations")
    users = rec.user_ids
    col1, col2 = st.columns([3, 1])
    with col1:
        uid = st.selectbox("User ID", users, index=0)
    with col2:
        k = st.slider("K", 3, 20, 10)
    cold_toggle = st.checkbox("Simulate a brand-new (cold) user", value=False)
    query_uid = "U_new_cold_user" if cold_toggle else uid

    result = rec.recommend(query_uid, k)
    st.info(f"**Strategy used:** {result['strategy']}")
    df = pd.DataFrame(result["recommendations"])
    if not df.empty:
        cols = [c for c in ["item_id", "category", "price", "score", "reason"] if c in df.columns]
        st.dataframe(df[cols], width="stretch", hide_index=True)

    with st.expander("This user's engagement history (categories)"):
        if not cold_toggle:
            u_idx = rec.user_index.get(uid)
            cats = rec._user_history(u_idx) if u_idx is not None else set()
            st.write("**Preferred categories:**", ", ".join(sorted(cats)) or "none")

# ---------------- Similar / cold-start ----------------
with tab_similar:
    st.subheader("Content-based similar items (works for cold-start products)")
    st.caption("Item-to-item similarity from product metadata. New products with no "
               "interaction history can still be matched — this is the cold-start fallback.")
    if items is not None:
        cold_ids = rec.cold_start_item_ids()
        c1, c2 = st.columns([2, 1])
        with c1:
            iid = st.selectbox("Item ID", rec.item_ids, index=0)
        with c2:
            only_cold = st.checkbox("Show a cold-start item", value=False)
            if only_cold and cold_ids:
                iid = st.selectbox("Cold-start item", cold_ids, index=0, key="cold_sel")
        is_cold = iid in set(cold_ids)
        if is_cold:
            st.warning(f"❄️ **{iid} is a cold-start item** (no interaction history) — "
                       "recommendations below come purely from content similarity.")
        sims = rec.similar_items(iid, k=10)
        if sims:
            sdf = pd.DataFrame(sims)
            cols = [c for c in ["item_id", "category", "price", "similarity"] if c in sdf.columns]
            st.dataframe(sdf[cols], width="stretch", hide_index=True)
        else:
            st.info("Unknown item.")

# ---------------- Performance ----------------
with tab_perf:
    st.subheader(f"Offline ranking metrics @K={metrics['k']} (time-based split)")
    mdf = pd.DataFrame(metrics["metrics_all"])
    show = mdf[["name", "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate"]]
    st.dataframe(
        show.style.format({"precision_at_k": "{:.4f}", "recall_at_k": "{:.4f}",
                           "ndcg_at_k": "{:.4f}", "hit_rate": "{:.3f}"})
        .background_gradient(subset=["precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate"],
                             cmap="RdYlGn"),
        width="stretch", hide_index=True,
    )
    st.success(f"**Best model:** {metrics['best_model']} — evaluated on "
               f"{metrics['n_users_evaluated']:,} held-out users.")
    st.caption(
        "Personalized models (ALS, Hybrid) beat the non-personalized **Popularity** baseline; "
        "**Content-based** is weaker for ranking but is essential for cold-start items. The "
        "**Hybrid** blends all three and is the deployed model."
    )
    with st.expander("Strong-signal evaluation (only cart/purchase count as relevant)"):
        sdf = pd.DataFrame(metrics["metrics_strong_signals"])
        st.dataframe(sdf[["name", "precision_at_k", "recall_at_k", "ndcg_at_k"]].style.format(
            {"precision_at_k": "{:.4f}", "recall_at_k": "{:.4f}", "ndcg_at_k": "{:.4f}"}),
            width="stretch", hide_index=True)

# ---------------- Segments ----------------
with tab_seg:
    st.subheader("Recommendation analysis across user segments")
    st.caption("Users are segmented by engagement level. This shows how the catalogue "
               "surfaced to each segment differs (step 6 of the PDF).")
    seg = pd.DataFrame(metrics["segment_analysis"])
    st.dataframe(seg, width="stretch", hide_index=True)
    if config.SEGMENTS_CSV.exists():
        st.download_button("⬇️ Download segment analysis (CSV)",
                           pd.read_csv(config.SEGMENTS_CSV).to_csv(index=False).encode(),
                           file_name="segment_recommendations.csv", mime="text/csv")

# ---------------- Plots ----------------
with tab_plots:
    for title, path in [("Recommender comparison @K", config.METRICS_BAR_PNG),
                        ("Catalogue coverage (diversity)", config.CATALOG_DIVERSITY_PNG)]:
        if Path(path).exists():
            st.subheader(title)
            st.image(str(path), width="stretch")
