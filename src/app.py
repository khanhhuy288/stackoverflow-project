"""Developer Salary Dashboard - Streamlit entry point.

Run with:  streamlit run src/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Developer Salary Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("views/0_Home.py", title="Home", icon="🏠", default=True)
predictor = st.Page("views/1_Salary_Predictor.py", title="Salary Predictor", icon="🔮")
insights = st.Page("views/2_Market_Insights.py", title="Market Insights", icon="🌍")

nav = st.navigation([home, predictor, insights])
nav.run()
