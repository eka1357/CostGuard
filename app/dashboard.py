"""
CostGuard - Interactive Cost-Aware Fraud Detection Dashboard
Streamlit application demonstrating real-time threshold optimization, financial savings,
and per-transaction SHAP explainability.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import load_artifact
from src.cost_optimizer import (
    sweep_thresholds,
    optimize_thresholds,
    calculate_savings_and_metrics,
    DEFAULT_TRANSACTION_VALUE,
    DEFAULT_INVESTIGATION_COST,
)
from src.explain import compute_shap_explanations, extract_case_drivers

# ==========================================
# Streamlit Page Configuration & Theming
# ==========================================
st.set_page_config(
    page_title="CostGuard | Cost-Aware Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for restrained institutional financial styling
st.markdown(
    """
    <style>
    /* Clean typography and padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Minimalist metric card styling */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
    }
    div[data-testid="metric-container"] label {
        color: #64748b;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 600;
    }
    /* Restrained section dividers */
    hr {
        margin: 1.25rem 0;
        border: 0;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_cached_data():
    """Load cached test predictions and pre-generated SHAP metadata."""
    test_cache = load_artifact("test_eval_cache.joblib")
    y_test = test_cache["y_test"].to_numpy()
    y_prob = test_cache["xgb_cal_probs"]
    X_test = test_cache["X_test"]
    raw_amounts = test_cache["raw_amounts"]

    shap_cache = load_artifact("shap_explanations.joblib")
    return X_test, y_test, y_prob, raw_amounts, shap_cache


# 1. Load Data
try:
    X_test, y_test, y_prob, raw_amounts, shap_cache = load_cached_data()
except Exception as e:
    st.error(
        f"Model artifacts not found. Please run the training pipeline first: `python src/train.py` ({e})"
    )
    st.stop()


# 2. Sidebar Controls: Cost Matrix Parameters
st.sidebar.markdown("### Operational Cost Parameters")
st.sidebar.caption(
    "Adjust investigation cost to observe how optimal threshold and projected dollars saved shift live. "
    "Fraud loss values are per-transaction from the dataset's actual Amount column."
)

investigation_cost = st.sidebar.slider(
    "Alert Investigation Cost ($)",
    min_value=1.0,
    max_value=50.0,
    value=DEFAULT_INVESTIGATION_COST,
    step=1.0,
    help="Labor and verification expense incurred for each alert investigated.",
)

st.sidebar.markdown("**Transaction Values:** Per-transaction Amount from dataset")

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision Threshold Policy")

# Dynamic threshold optimization
opt_results = optimize_thresholds(
    y_true=y_test,
    y_prob=y_prob,
    amounts=raw_amounts,
    investigation_cost=investigation_cost,
    num_steps=500,
)

cost_opt_thresh = float(opt_results["cost_optimal"]["threshold"])
f1_opt_thresh = float(opt_results["f1_optimal"]["threshold"])

threshold_mode = st.sidebar.radio(
    "Threshold Selection",
    ["Cost-Optimal (Recommended)", "F1-Optimal", "Manual Override"],
    index=0,
)

if threshold_mode == "Cost-Optimal (Recommended)":
    active_threshold = cost_opt_thresh
elif threshold_mode == "F1-Optimal":
    active_threshold = f1_opt_thresh
else:
    active_threshold = st.sidebar.slider(
        "Manual Operating Threshold",
        min_value=0.001,
        max_value=0.999,
        value=cost_opt_thresh,
        step=0.005,
    )

st.sidebar.markdown(f"**Current Threshold:** `{active_threshold:.4f}`")

# 3. Header & Project Overview
st.title("CostGuard: Cost-Aware Fraud Detection")
st.caption(
    "Financial threshold optimization and local SHAP explainability on Kaggle/ULB Credit Card Fraud Dataset (56,962 test transactions)."
)

# 4. KPI Metrics Row
current_metrics = calculate_savings_and_metrics(
    y_true=y_test,
    y_prob=y_prob,
    threshold=active_threshold,
    amounts=raw_amounts,
    investigation_cost=investigation_cost,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Net Dollars Saved",
        value=f"${current_metrics['dollars_saved']:,.2f}",
        delta=f"{opt_results['pct_gain_over_f1']:+.1f}% vs F1-Optimal"
        if threshold_mode == "Cost-Optimal (Recommended)"
        else None,
    )
with col2:
    st.metric(
        label="Operating Threshold",
        value=f"{active_threshold:.4f}",
        delta="Cost-Optimal" if active_threshold == cost_opt_thresh else ("F1-Optimal" if active_threshold == f1_opt_thresh else "Manual"),
    )
with col3:
    st.metric(
        label="Frauds Caught (Recall)",
        value=f"{current_metrics['true_positives']} / {int(current_metrics['true_positives'] + current_metrics['false_negatives'])}",
        delta=f"{current_metrics['recall']:.1%}",
    )
with col4:
    st.metric(
        label="False Alarms (FP Alerts)",
        value=f"{current_metrics['false_positives']}",
        delta=f"Prec: {current_metrics['precision']:.1%}",
    )

st.markdown("---")

# 5. Interactive Chart: Expected Savings vs Decision Threshold
st.subheader("Threshold vs. Expected Dollars Saved Curve")

sweep_df = opt_results["sweep_curve"]

fig = go.Figure()

# Main savings curve
fig.add_trace(
    go.Scatter(
        x=sweep_df["threshold"],
        y=sweep_df["dollars_saved"],
        mode="lines",
        name="Projected Net Savings ($)",
        line=dict(color="#0284c7", width=2.5),
        hovertemplate="Threshold: %{x:.4f}<br>Net Savings: $%{y:,.2f}<extra></extra>",
    )
)

# Vertical line for Cost-Optimal Threshold
fig.add_vline(
    x=cost_opt_thresh,
    line_dash="dash",
    line_color="#059669",
    line_width=1.8,
    annotation_text=f"Cost-Optimal: {cost_opt_thresh:.3f} (${opt_results['cost_optimal']['dollars_saved']:,.0f})",
    annotation_position="top right",
    annotation_font=dict(size=11, color="#059669"),
)

# Vertical line for F1-Optimal Threshold
fig.add_vline(
    x=f1_opt_thresh,
    line_dash="dot",
    line_color="#d97706",
    line_width=1.5,
    annotation_text=f"F1-Optimal: {f1_opt_thresh:.3f} (${opt_results['f1_optimal']['dollars_saved']:,.0f})",
    annotation_position="bottom right",
    annotation_font=dict(size=11, color="#d97706"),
)

# Current threshold marker
fig.add_trace(
    go.Scatter(
        x=[active_threshold],
        y=[current_metrics["dollars_saved"]],
        mode="markers",
        name="Active Threshold",
        marker=dict(color="#e11d48", size=9, symbol="circle"),
        hovertemplate=f"Active Threshold: {active_threshold:.4f}<br>Savings: ${current_metrics['dollars_saved']:,.2f}<extra></extra>",
    )
)

fig.update_layout(
    height=380,
    margin=dict(l=40, r=40, t=25, b=40),
    xaxis=dict(
        title="Decision Threshold (Classification Cutoff)",
        gridcolor="#f1f5f9",
        range=[0.0, 1.0],
    ),
    yaxis=dict(
        title="Net Dollars Saved ($)",
        gridcolor="#f1f5f9",
        tickprefix="$",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 6. Flagged Cases & SHAP Explanations
st.subheader("Flagged Transactions & Local SHAP Attribution")
st.caption(
    "Inspection table for transactions exceeding the operating decision threshold, auditing specific feature contributions."
)

flagged_mask = y_prob >= active_threshold
flagged_indices = np.where(flagged_mask)[0]
total_flagged = len(flagged_indices)

if total_flagged == 0:
    st.info(f"No transactions meet or exceed the active threshold ({active_threshold:.4f}).")
else:
    # Sort flagged by risk probability descending
    sorted_order = np.argsort(y_prob[flagged_indices])[::-1]
    top_flagged_idx = flagged_indices[sorted_order[:15]]

    table_rows = []
    for rank, idx in enumerate(top_flagged_idx, 1):
        actual = int(y_test[idx])
        prob = float(y_prob[idx])
        orig_id = int(X_test.index[idx])
        amount = float(X_test.iloc[idx].get("Amount", 0.0))
        table_rows.append({
            "Rank": rank,
            "Tx ID": orig_id,
            "Fraud Probability": f"{prob:.2%}",
            "Ground Truth": "Fraud" if actual == 1 else "Legitimate",
            "Classification": "True Positive (Fraud Stopped)" if actual == 1 else "False Positive (False Alarm)",
            "_idx": idx,
        })

    flagged_table_df = pd.DataFrame(table_rows)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(f"**Top Flagged Alerts (Showing {len(table_rows)} of {total_flagged:,} Total Flagged)**")
        display_df = flagged_table_df.drop(columns=["_idx"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with c2:
        selected_rank = st.selectbox(
            "Select Flagged Transaction to Inspect:",
            options=flagged_table_df["Rank"].tolist(),
            format_func=lambda r: f"Rank #{r} — Tx ID {flagged_table_df.loc[flagged_table_df['Rank']==r, 'Tx ID'].values[0]} ({flagged_table_df.loc[flagged_table_df['Rank']==r, 'Fraud Probability'].values[0]} risk)",
        )

        selected_row = flagged_table_df[flagged_table_df["Rank"] == selected_rank].iloc[0]
        selected_idx = int(selected_row["_idx"])

        # Check if selected transaction has precomputed SHAP values in cache
        shap_top_indices = shap_cache["summary_df"]["test_index"].tolist()

        if selected_idx in shap_top_indices:
            cache_pos = shap_top_indices.index(selected_idx)
            drivers = shap_cache["drivers_list"][cache_pos]
            top_pos = drivers["top_positive_drivers"]
            top_neg = drivers["top_negative_drivers"]
        else:
            # Fallback: compute local attribution on demand for selected sample
            sample_df = X_test.iloc[[selected_idx]]
            model_raw = load_artifact("xgb_model.joblib")
            shap_local = compute_shap_explanations(model_raw, sample_df)
            drivers = extract_case_drivers(shap_local, list(X_test.columns), case_idx=0, top_k=4)
            top_pos = drivers["top_positive_drivers"]
            top_neg = drivers["top_negative_drivers"]

        # Build feature attribution bar chart
        all_drivers = top_pos + top_neg
        driver_df = pd.DataFrame(all_drivers).sort_values("shap_attribution", ascending=True)

        bar_colors = ["#e11d48" if val > 0 else "#2563eb" for val in driver_df["shap_attribution"]]

        fig_shap = go.Figure()
        fig_shap.add_trace(
            go.Bar(
                x=driver_df["shap_attribution"],
                y=driver_df["feature"],
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="Feature: %{y}<br>SHAP Value: %{x:+.3f}<extra></extra>",
            )
        )

        fig_shap.update_layout(
            title=dict(
                text=f"Local Feature Attribution (Tx #{selected_row['Tx ID']})",
                font=dict(size=13),
            ),
            height=280,
            margin=dict(l=30, r=20, t=35, b=30),
            xaxis=dict(
                title="SHAP Value (Impact on Log-Odds)",
                zeroline=True,
                zerolinecolor="#94a3b8",
                gridcolor="#f1f5f9",
            ),
            yaxis=dict(title=""),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "Red bars escalate fraud probability (positive SHAP contribution); Blue bars mitigate fraud probability."
        )

# 7. Operational Methodology Footer
st.markdown("---")
st.markdown("#### Methodology & Cost Assumptions")
st.markdown(
    f"""
    - **Expected Savings Formulation (Per-Transaction)**:
      $$\\text{{Net Savings}} = \\sum_{{\\text{{caught frauds}}}} \\text{{Amount}}_i - (\\text{{Total Alerts}} \\times C_{{\\text{{invest}}}})$$
      where each fraud's loss is its actual transaction Amount, and $C_{{\\text{{invest}}}} = \\${investigation_cost:,.2f}$.
    - **Baseline Fraud Loss** (no model): $\\${current_metrics['baseline_loss']:,.2f}$ across {int(current_metrics['true_positives'] + current_metrics['false_negatives'])} fraud transactions.
    - **Disclaimer**: *CostGuard is an experimental research prototype for GIBC V2 Track 02. Not a medical device, not a diagnostic tool, and not financial advice.*
    """
)
