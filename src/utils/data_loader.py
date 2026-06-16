"""Cached data loading and cleaning pipeline for the Streamlit dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = _PROJECT_ROOT / "exports" / "survey_2025_scalar_columns.csv"
MODEL_PATH = (
    _PROJECT_ROOT / "src" / "notebooks" / "AutogluonModels" / "ag-20260521_144442"
)

TARGET = "ConvertedCompYearly"
LOG_TARGET = "LogCompYearly"
LEAK_COLS = ["ResponseId", "CompTotal", "Currency"]

COMP_MIN = 1_000
COMP_MAX = 500_000

AGE_UPPER_BOUND = {
    "18-24 years old": 24,
    "25-34 years old": 34,
    "35-44 years old": 44,
    "45-54 years old": 54,
    "55-64 years old": 64,
    "65 years or older": 80,
    "Prefer not to say": np.nan,
}

# The 8 features exposed in the predictor form, ordered by importance.
INPUT_FEATURES = [
    "Country",
    "WorkExp",
    "OrgSize",
    "YearsCode",
    "Employment",
    "DevType",
    "RemoteWork",
    "EdLevel",
]

# Pre-computed from predictor_v2.feature_importance(test_clean) in the notebook.
FEATURE_IMPORTANCE = {
    "Country": 0.148590,
    "WorkExp": 0.023896,
    "OrgSize": 0.009236,
    "YearsCode": 0.005064,
    "Employment": 0.004102,
    "DevType": 0.003648,
    "RemoteWork": 0.003097,
    "Industry": 0.002845,
    "Age": 0.002309,
    "EdLevel": 0.002169,
    "JobSat": 0.001427,
    "PlatformChoice": 0.001031,
    "SODuration": 0.000969,
    "PurchaseInfluence": 0.000480,
    "AIComplex": 0.000471,
    "SOFriction": 0.000461,
    "LearnCodeChoose": 0.000436,
    "AISelect": 0.000427,
    "TechEndorseIntro": 0.000396,
    "ToolCountWork": 0.000362,
    "SOVisitFreq": 0.000321,
    "WebframeChoice": 0.000311,
    "AIAgents": 0.000299,
    "AIAcc": 0.000289,
    "NewRole": 0.000279,
    "AIModelsChoice": 0.000238,
    "LearnCodeAI": 0.000214,
    "ICorPM": 0.000196,
    "SOComm": 0.000175,
    "DatabaseChoice": 0.000165,
    "ToolCountPersonal": 0.000152,
    "SOAccount": 0.000117,
    "LanguageChoice": 0.000063,
    "AIThreat": 0.000059,
    "SOPartFreq": 0.000005,
    "AISent": 0.000002,
    "DevEnvsChoice": -0.000017,
    "AIAgentChange": -0.000037,
}


def _clean_survey(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning pipeline used in the comp_prediction notebook (v2)."""
    df = df[df["MainBranch"] == "I am a developer by profession"].copy()
    df = df.dropna(subset=[TARGET])

    df["_age_upper"] = df["Age"].map(AGE_UPPER_BOUND)

    mask = df["_age_upper"].notna() & df["WorkExp"].notna()
    df = df[~(mask & (df["WorkExp"] > df["_age_upper"]))]

    mask = df["_age_upper"].notna() & df["YearsCode"].notna()
    df = df[~(mask & (df["YearsCode"] > df["_age_upper"]))]

    mask = df["WorkExp"].notna() & df["YearsCode"].notna()
    df = df[~(mask & (df["WorkExp"] > df["YearsCode"] + 5))]

    df = df.drop(columns=["_age_upper"])

    df = df[df[TARGET] >= COMP_MIN]
    df = df[df[TARGET] <= COMP_MAX]

    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading survey data…")
def load_survey_data() -> pd.DataFrame:
    """Load and clean the survey CSV.  Returns the full cleaned frame *with*
    ``ConvertedCompYearly`` kept (needed for the insights page distributions).
    """
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = _clean_survey(df)
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])
    return df


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    """Load the trained AutoGluon TabularPredictor (v2, log-target)."""
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(str(MODEL_PATH))


def get_column_options(df: pd.DataFrame, col: str) -> list[str]:
    """Return sorted unique non-null values for a categorical column."""
    vals = df[col].dropna().unique().tolist()
    return sorted(vals)
