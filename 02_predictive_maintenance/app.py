"""Streamlit monitoring dashboard for Predictive Maintenance (step 6 of the PDF).

Shows fleet-wide machine risk scores, triggers alerts for high-risk equipment, and lets an
engineer drill into a single machine's sensor + risk timeline around failures.

Launch:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Predictive Maintenance Monitor", page_icon="🏭", layout="wide")


@st.cache_data(show_spinner=False)
def load_snapshot():
    if not config.RISK_SNAPSHOT_CSV.exists():
        return None
    return pd.read_csv(config.RISK_SNAPSHOT_CSV)


@st.cache_data(show_spinner=False)
def load_timeline():
    if not config.RISK_TIMELINE_CSV.exists():
        return None
    return pd.read_csv(config.RISK_TIMELINE_CSV)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not config.METRICS_JSON.exists():
        return None
    with open(config.METRICS_JSON) as f:
        return json.load(f)


snap, timeline, metrics = load_snapshot(), load_timeline(), load_metrics()

st.title("🏭 Predictive Maintenance Monitor")
st.caption(
    f"Predicts whether a machine will fail within the next "
    f"{metrics['prediction_window_hours'] if metrics else '?'} hours from sensor telemetry."
)

if snap is None or metrics is None:
    st.warning("No trained model found. Run:\n\n```\npython -m src.train\n```")
    st.stop()

tab_fleet, tab_machine, tab_perf, tab_plots = st.tabs(
    ["🚦 Fleet Status", "🔬 Machine Drill-down", "📈 Model Performance", "🖼️ Plots"]
)

# ---------------- Fleet status ----------------
with tab_fleet:
    best = metrics["best_model"]
    brow = pd.DataFrame(metrics["metrics_table"]).query("name == @best").iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Machines monitored", metrics["n_machines"])
    c2.metric("🔴 Alerts", metrics["n_alerts"])
    c3.metric("🟠 Watch", metrics["n_watch"])
    c4.metric("Recall (caught failures)", f"{brow['recall']:.2f}")
    c5.metric("False alarms (test)", int(brow["false_positives"]))

    st.subheader("Live fleet risk")
    st.caption("Most recent reading per machine, scored by the deployed model. "
               "🔴 ALERT means failure is predicted inside the window — dispatch maintenance.")

    def _color(risk):
        if risk >= 0.5:
            return "background-color: #ffd6d6"
        if risk >= 0.3:
            return "background-color: #fff0cc"
        return "background-color: #e2f5e2"

    styled = snap.style.map(_color, subset=["risk_score"]).format({"risk_score": "{:.3f}"})
    st.dataframe(styled, width="stretch", height=460)
    st.download_button("⬇️ Download fleet risk (CSV)", snap.to_csv(index=False).encode(),
                       file_name="machine_risk_snapshot.csv", mime="text/csv")

# ---------------- Machine drill-down ----------------
with tab_machine:
    if timeline is None:
        st.info("Run `python -m src.train` to build the risk timeline.")
    else:
        machines = sorted(timeline["machine_id"].unique())
        alerted = snap[snap["alert"] == "🔴 ALERT"]["machine_id"].tolist()
        default_idx = machines.index(alerted[0]) if alerted and alerted[0] in machines else 0
        mid = st.selectbox("Machine", machines, index=default_idx)
        mdata = timeline[timeline["machine_id"] == mid].sort_values("hour")
        mfail = mdata[mdata["failure_event"] == 1]

        latest = mdata.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current risk", f"{latest['risk_score']:.2f}")
        c2.metric("Temperature (°C)", f"{latest['temperature_c']:.1f}")
        c3.metric("Vibration (mm/s)", f"{latest['vibration_mm_s']:.2f}")
        c4.metric("Hours since maintenance", f"{latest['hours_since_maintenance']:.0f}")

        # risk + label timeline
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(mdata["hour"], mdata["risk_score"], color="crimson", lw=1.4, label="Predicted risk")
        ax.fill_between(mdata["hour"], 0, 1, where=(mdata["label"] == 1),
                        color="orange", alpha=0.15, label="Within failure window")
        ax.axhline(0.5, ls="--", color="grey", lw=1, label="Alert threshold")
        for _, f in mfail.iterrows():
            ax.axvline(f["hour"], color="black", lw=1.5, alpha=0.7)
        if len(mfail):
            ax.axvline(mfail.iloc[0]["hour"], color="black", lw=1.5, alpha=0.7, label="Actual failure")
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel("Risk")
        ax.set_xlabel("Operating hour")
        ax.set_title(f"Machine {mid} — predicted failure risk over time")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # sensor panel
        fig2, axs = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
        for axx, col, lab, colr in zip(
            axs.ravel(),
            ["temperature_c", "vibration_mm_s", "pressure_kpa", "hours_since_maintenance"],
            ["Temperature (°C)", "Vibration (mm/s)", "Pressure (kPa)", "Hours since maintenance"],
            ["tab:red", "tab:purple", "tab:blue", "tab:green"],
        ):
            axx.plot(mdata["hour"], mdata[col], color=colr, lw=0.9)
            for _, f in mfail.iterrows():
                axx.axvline(f["hour"], color="black", lw=1.2, alpha=0.6)
            axx.set_ylabel(lab, fontsize=9)
            axx.set_xlabel("Operating hour", fontsize=9)
        fig2.suptitle(f"Machine {mid} — sensor telemetry (black lines = failures)", y=1.0)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ---------------- Model performance ----------------
with tab_perf:
    st.subheader("Model comparison (chronological test split)")
    tbl = pd.DataFrame(metrics["metrics_table"])
    show = tbl[["name", "pr_auc", "roc_auc", "recall", "precision", "f1",
                "threshold", "true_positives", "false_positives", "false_negatives"]]
    st.dataframe(
        show.style.format({
            "pr_auc": "{:.3f}", "roc_auc": "{:.3f}", "recall": "{:.3f}",
            "precision": "{:.3f}", "f1": "{:.3f}", "threshold": "{:.2f}",
        }).background_gradient(subset=["pr_auc", "recall", "precision"], cmap="RdYlGn"),
        width="stretch",
    )
    st.info(
        "**How to read this:** failures are rare (~3 % of readings). PR-AUC and recall tell "
        "you how many real failures are caught; `false_positives` are **false alarms** "
        "(needless maintenance dispatches). Note some failures are *sudden* with no sensor "
        "precursor, so recall is naturally capped below 1.0."
    )

# ---------------- Plots ----------------
with tab_plots:
    for title, path in [
        ("Precision-Recall curves", config.PR_PNG),
        ("ROC curves", config.ROC_PNG),
        ("Precision / Recall / False-alarm trade-off", config.FALSE_ALARM_PNG),
        ("Confusion matrix (best model)", config.CONFUSION_PNG),
        ("Feature importance", config.FEATURE_IMPORTANCE_PNG),
    ]:
        if Path(path).exists():
            st.subheader(title)
            st.image(str(path), width="stretch")
