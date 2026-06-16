"""Page 2 - Market Insights: choropleth, feature importance, breakdowns."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import (
    FEATURE_IMPORTANCE,
    TARGET,
    load_survey_data,
    wrap_label,
)

st.header("🌍 Market Insights")
st.markdown(
    "Explore developer compensation trends from the "
    "2025 Stack Overflow Developer Survey (19,255 professional developers)."
)

df = load_survey_data()

# ── 1. Choropleth map ──────────────────────────────────────────────────────

st.subheader("🗺️ Median Salary by Country")

country_stats = (
    df.groupby("Country")[TARGET]
    .agg(["median", "count"])
    .rename(columns={"median": "Median Salary", "count": "Respondents"})
    .reset_index()
)
country_stats = country_stats[country_stats["Respondents"] >= 10]

fig_map = px.choropleth(
    country_stats,
    locations="Country",
    locationmode="country names",
    color="Median Salary",
    hover_name="Country",
    hover_data={"Median Salary": ":$,.0f", "Respondents": ":,", "Country": False},
    color_continuous_scale="Viridis",
    labels={"Median Salary": "Median Salary (USD)"},
)
fig_map.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=450,
    coloraxis_colorbar=dict(title="USD", tickformat="$,.0f"),
    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
)
st.plotly_chart(fig_map, width="stretch")

# ── 2. Feature importance ──────────────────────────────────────────────────

st.subheader("📊 What Drives Salary?")
st.markdown(
    "Permutation feature importance from the prediction model. Higher values "
    "mean the feature has a larger effect on salary predictions. Features "
    "near zero are essentially noise."
)

fi_df = (
    pd.DataFrame(
        {"Feature": list(FEATURE_IMPORTANCE.keys()), "Importance": list(FEATURE_IMPORTANCE.values())}
    )
    .sort_values("Importance", ascending=True)
)

fig_fi = go.Figure()
fig_fi.add_trace(
    go.Bar(
        y=fi_df["Feature"],
        x=fi_df["Importance"],
        orientation="h",
        marker_color=["#FF4B4B" if v > 0.002 else "#cccccc" for v in fi_df["Importance"]],
    )
)
fig_fi.update_layout(
    xaxis_title="Permutation Importance",
    yaxis_title="",
    height=700,
    margin=dict(l=10, r=10, t=10, b=40),
)
st.plotly_chart(fig_fi, width="stretch")

# ── 3. Interactive breakdowns ──────────────────────────────────────────────

st.subheader("📈 Salary Breakdowns")

tab_role, tab_edu, tab_exp, tab_remote = st.tabs(
    ["By Role", "By Education", "By Experience", "By Work Arrangement"]
)

_FIXED_LEFT_MARGIN = 250

with tab_role:
    role_counts = df["DevType"].value_counts()
    top_roles = role_counts[role_counts >= 30].index.tolist()
    df_role = df[df["DevType"].isin(top_roles)].copy()

    median_order = (
        df_role.groupby("DevType")[TARGET].median().sort_values(ascending=False).index.tolist()
    )
    label_map = {r: wrap_label(r) for r in median_order}
    df_role["DevType"] = df_role["DevType"].map(label_map)
    wrapped_order = [label_map[r] for r in median_order]

    fig_role = px.box(
        df_role,
        x=TARGET,
        y="DevType",
        category_orders={"DevType": wrapped_order},
        labels={TARGET: "Annual Salary (USD)", "DevType": ""},
    )
    fig_role.update_layout(
        height=max(400, len(median_order) * 35),
        margin=dict(l=_FIXED_LEFT_MARGIN, r=10, t=10, b=40),
    )
    st.plotly_chart(fig_role, width="stretch")

with tab_edu:
    edu_order = (
        df.groupby("EdLevel")[TARGET].median().sort_values(ascending=False).index.tolist()
    )
    df_edu = df.copy()
    label_map = {e: wrap_label(e) for e in edu_order}
    df_edu["EdLevel"] = df_edu["EdLevel"].map(label_map)
    wrapped_order = [label_map[e] for e in edu_order]

    fig_edu = px.box(
        df_edu,
        x=TARGET,
        y="EdLevel",
        category_orders={"EdLevel": wrapped_order},
        labels={TARGET: "Annual Salary (USD)", "EdLevel": ""},
    )
    fig_edu.update_layout(
        height=450,
        margin=dict(l=_FIXED_LEFT_MARGIN, r=10, t=10, b=40),
    )
    st.plotly_chart(fig_edu, width="stretch")

with tab_exp:
    exp_df = df.dropna(subset=["WorkExp", TARGET])
    exp_df = exp_df[exp_df["WorkExp"] <= 40]

    agg = (
        exp_df.groupby("WorkExp")[TARGET]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    )
    agg.columns = ["Median", "Q1", "Q3"]
    agg = agg.reset_index()

    fig_exp = go.Figure()
    fig_exp.add_trace(
        go.Scatter(
            x=agg["WorkExp"],
            y=agg["Q3"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig_exp.add_trace(
        go.Scatter(
            x=agg["WorkExp"],
            y=agg["Q1"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,75,75,0.15)",
            name="IQR (25th-75th)",
        )
    )
    fig_exp.add_trace(
        go.Scatter(
            x=agg["WorkExp"],
            y=agg["Median"],
            mode="lines+markers",
            line=dict(color="#FF4B4B", width=2),
            name="Median",
        )
    )
    fig_exp.update_layout(
        xaxis_title="Years of Professional Experience",
        yaxis_title="Annual Salary (USD)",
        height=400,
        margin=dict(l=10, r=10, t=10, b=40),
        yaxis_tickformat="$,.0f",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_exp, width="stretch")

with tab_remote:
    df_remote = df.dropna(subset=["RemoteWork"]).copy()
    remote_order = (
        df_remote.groupby("RemoteWork")[TARGET]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    label_map = {r: wrap_label(r) for r in remote_order}
    df_remote["RemoteWork"] = df_remote["RemoteWork"].map(label_map)
    wrapped_order = [label_map[r] for r in remote_order]

    fig_remote = px.box(
        df_remote,
        x=TARGET,
        y="RemoteWork",
        category_orders={"RemoteWork": wrapped_order},
        labels={TARGET: "Annual Salary (USD)", "RemoteWork": ""},
    )
    fig_remote.update_layout(
        height=300,
        margin=dict(l=_FIXED_LEFT_MARGIN, r=10, t=10, b=40),
    )
    st.plotly_chart(fig_remote, width="stretch")
