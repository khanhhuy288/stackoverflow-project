"""Page 1 - Salary Predictor with inline What-If explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import (
    FORM_DEFAULTS,
    TARGET,
    get_column_options,
    load_model,
    load_survey_data,
)
from utils.predict import (
    MODEL_MAE_USD,
    build_input_row,
    compute_percentile,
    compute_whatif_deltas,
    predict_salary,
)

st.header("🔮 Salary Predictor")
st.markdown(
    "Fill in your developer profile below to get an estimated annual "
    "compensation range based on the 2025 Stack Overflow Survey."
)

df = load_survey_data()
predictor = load_model()

salaries = df[TARGET]

# ── Input form ──────────────────────────────────────────────────────────────

countries = get_column_options(df, "Country")
dev_types = get_column_options(df, "DevType")
employment_opts = get_column_options(df, "Employment")
org_sizes = get_column_options(df, "OrgSize")
remote_opts = get_column_options(df, "RemoteWork")
ed_levels = get_column_options(df, "EdLevel")


def _default_index(options: list[str], key: str) -> int:
    """Return the index of the default value for *key*, or 0 if not found."""
    default = FORM_DEFAULTS.get(key)
    if default and default in options:
        return options.index(default)
    return 0


with st.form("predictor_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        country = st.selectbox("Country", countries, index=_default_index(countries, "Country"))
        dev_type = st.selectbox("Developer Role", dev_types, index=_default_index(dev_types, "DevType"))
        employment = st.selectbox("Employment", employment_opts, index=_default_index(employment_opts, "Employment"))
        org_size = st.selectbox("Company Size", org_sizes, index=_default_index(org_sizes, "OrgSize"))

    with col_right:
        work_exp = st.slider("Years of Professional Experience", 0, 50, 5)
        years_code = st.slider("Total Years Coding", 0, 50, 8)
        remote_work = st.radio(
            "Work Arrangement",
            remote_opts,
            index=remote_opts.index(FORM_DEFAULTS["RemoteWork"]) if FORM_DEFAULTS["RemoteWork"] in remote_opts else 0,
            horizontal=True,
        )
        ed_level = st.selectbox("Education Level", ed_levels, index=_default_index(ed_levels, "EdLevel"))

    submitted = st.form_submit_button("Predict My Salary", use_container_width=True, type="primary")

# ── Persist prediction in session state ─────────────────────────────────────

if submitted:
    input_row = build_input_row(
        country=country,
        work_exp=float(work_exp),
        org_size=org_size,
        years_code=float(years_code),
        employment=employment,
        dev_type=dev_type,
        remote_work=remote_work,
        ed_level=ed_level,
    )
    result = predict_salary(predictor, input_row)
    pct = compute_percentile(result["salary"], salaries)

    st.session_state["prediction"] = {
        "result": result,
        "pct": pct,
        "input_row": input_row,
    }

# ── Results (rendered from session state) ───────────────────────────────────

if "prediction" not in st.session_state:
    st.stop()

pred = st.session_state["prediction"]
result = pred["result"]
pct = pred["pct"]
input_row = pred["input_row"]

st.divider()

# ── Headline metrics ────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Estimated Salary", f"${result['salary']:,}")
m2.markdown(
    f"<p style='font-size:0.875rem;color:#808495;margin-bottom:0'>Likely Range</p>"
    f"<p style='font-size:2.25rem;font-weight:700;margin:0'>"
    f"${result['low']:,} - ${result['high']:,}</p>",
    unsafe_allow_html=True,
)
m3.metric("Percentile", f"Top {100 - pct:.0f}%", help="Among professional developers in the survey")

# ── Distribution chart with marker ──────────────────────────────────────────
fig = px.histogram(
    salaries,
    nbins=80,
    labels={"value": "Annual Salary (USD)", "count": "Developers"},
    opacity=0.7,
)
fig.add_vline(
    x=result["salary"],
    line_dash="dash",
    line_color="#FF4B4B",
    line_width=2,
    annotation_text=f"Your estimate: ${result['salary']:,}",
    annotation_position="top right",
    annotation_font_color="#FF4B4B",
)
fig.update_layout(
    showlegend=False,
    xaxis_title="Annual Salary (USD)",
    yaxis_title="Number of Developers",
    margin=dict(t=30, b=40),
    height=300,
)
st.plotly_chart(fig, width="stretch")

# ── Inline What-If ──────────────────────────────────────────────────────────
st.subheader("🔄 What If...?")
st.markdown(
    "See how your predicted salary changes if you adjust a single factor, "
    "keeping everything else the same."
)

whatif_feature = st.selectbox(
    "Explore a factor",
    ["Country", "DevType", "RemoteWork", "OrgSize", "Employment", "EdLevel"],
    key="whatif_select",
)

options = get_column_options(df, whatif_feature)
with st.spinner("Computing scenarios..."):
    deltas = compute_whatif_deltas(predictor, input_row, result["salary"], whatif_feature, options)

delta_df = pd.DataFrame(deltas)

top_n = min(15, len(delta_df))
show_df = delta_df.head(top_n)

fig_wi = go.Figure()
colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in show_df["delta"]]
fig_wi.add_trace(
    go.Bar(
        y=show_df["option"],
        x=show_df["delta"],
        orientation="h",
        marker_color=colors,
        text=[f"+${d:,}" if d >= 0 else f"-${abs(d):,}" for d in show_df["delta"]],
        textposition="outside",
    )
)
fig_wi.update_layout(
    xaxis_title="Salary Change (USD)",
    yaxis_title="",
    height=max(350, top_n * 28),
    margin=dict(l=10, r=10, t=10, b=40),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_wi, width="stretch")

# ── Model disclaimer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Based on 19,255 professional developer responses from the "
    "2025 Stack Overflow Developer Survey. "
    f"Model R² = 0.55, typical error ~ ${MODEL_MAE_USD:,}. "
    "Predictions are rough estimates, not guarantees."
)
