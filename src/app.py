"""Developer Salary Dashboard — Streamlit entry point.

Run with:  streamlit run src/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Developer Salary Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Developer Salary Dashboard")
st.markdown(
    """
    Welcome! This dashboard uses a machine-learning model trained on
    **19,255 professional developer responses** from the
    [2025 Stack Overflow Developer Survey](https://survey.stackoverflow.co/)
    to explore and predict developer compensation worldwide.

    **Use the sidebar** to navigate between pages:

    - **Salary Predictor** — Enter your developer profile and get an
      estimated salary range, percentile ranking, and What-If scenarios.
    - **Market Insights** — Explore salary distributions by country,
      role, education, experience, and see which factors drive pay the most.
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("About the Model")
    st.markdown(
        """
        | Detail | Value |
        |--------|-------|
        | Algorithm | AutoGluon ensemble (CatBoost + others) |
        | Target | Annual compensation (USD) |
        | Training samples | 15,404 |
        | Test R² | 0.55 |
        | Test MAE | ~$28,700 |
        | Key predictor | Country (by far) |
        """
    )

with col2:
    st.subheader("Data Pipeline")
    st.markdown(
        """
        1. **Raw survey** — 53,921 responses, 172 columns
        2. **Scalar columns** — 43 columns selected
        3. **Cleaning** — professional devs only, valid experience,
           salary $1k–$500k
        4. **Log-target model** — predicts log₁₀(salary) for better
           relative accuracy
        5. **This dashboard** — 8 high-impact features exposed
        """
    )
