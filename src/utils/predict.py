"""Prediction helpers: build input row, run model, compute range & percentile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_loader import COUNTRY_RENAMES, INPUT_FEATURES, LOG_TARGET, TARGET

_REVERSE_COUNTRY = {v: k for k, v in COUNTRY_RENAMES.items()}

# Approximate MAE (in USD) of the v2 model on the test set.
MODEL_MAE_USD = 28_671

# All 39 feature columns the AutoGluon model was trained on (excluding the
# LogCompYearly target).  The model's feature generator requires every column
# to be present, even if the value is NaN.
_ALL_MODEL_FEATURES = [
    "MainBranch", "Age", "EdLevel", "Employment", "WorkExp",
    "LearnCodeChoose", "LearnCodeAI", "YearsCode", "DevType", "OrgSize",
    "ICorPM", "RemoteWork", "PurchaseInfluence", "TechEndorseIntro",
    "Industry", "AIThreat", "NewRole", "ToolCountWork", "ToolCountPersonal",
    "Country", "LanguageChoice", "DatabaseChoice", "PlatformChoice",
    "WebframeChoice", "DevEnvsChoice", "AIModelsChoice", "SOAccount",
    "SOVisitFreq", "SODuration", "SOPartFreq", "SOComm", "SOFriction",
    "AISelect", "AISent", "AIAcc", "AIComplex", "AIAgents",
    "AIAgentChange", "JobSat",
]


def build_input_row(
    *,
    country: str | None = None,
    work_exp: float | None = None,
    org_size: str | None = None,
    years_code: float | None = None,
    employment: str | None = None,
    dev_type: str | None = None,
    remote_work: str | None = None,
    ed_level: str | None = None,
) -> pd.DataFrame:
    """Create a single-row DataFrame matching the model's expected schema.

    All 39 feature columns are included.  The 8 user-provided features are
    set from the arguments; the remaining 31 are left as NaN (AutoGluon
    handles missing values natively).  ``MainBranch`` is hard-coded because
    the v2 model was trained on professional developers only.
    """
    row: dict = {feat: np.nan for feat in _ALL_MODEL_FEATURES}
    row["MainBranch"] = "I am a developer by profession"
    row["Country"] = _REVERSE_COUNTRY.get(country, country)
    row["WorkExp"] = work_exp
    row["OrgSize"] = org_size
    row["YearsCode"] = years_code
    row["Employment"] = employment
    row["DevType"] = dev_type
    row["RemoteWork"] = remote_work
    row["EdLevel"] = ed_level
    return pd.DataFrame([row])


def predict_salary(predictor, input_row: pd.DataFrame) -> dict:
    """Run the model and return point estimate + range in USD.

    Returns a dict with keys:
      - salary: point estimate (USD)
      - low / high: range based on +/- MAE
      - log_pred: raw log10 prediction
    """
    log_pred = float(predictor.predict(input_row).iloc[0])
    salary = 10**log_pred

    low = max(salary - MODEL_MAE_USD, 0)
    high = salary + MODEL_MAE_USD

    return {
        "salary": round(salary),
        "low": round(low),
        "high": round(high),
        "log_pred": log_pred,
    }


def compute_percentile(salary: float, salaries: pd.Series) -> float:
    """Return the percentile rank (0–100) of *salary* within *salaries*."""
    return float((salaries < salary).mean() * 100)


def compute_whatif_deltas(
    predictor,
    base_row: pd.DataFrame,
    base_salary: float,
    feature: str,
    options: list[str],
) -> list[dict]:
    """For a categorical *feature*, predict salary for every *option* and
    return the delta compared to *base_salary*, sorted by delta descending.
    """
    rows = pd.concat([base_row] * len(options), ignore_index=True)
    model_values = options
    if feature == "Country":
        model_values = [_REVERSE_COUNTRY.get(o, o) for o in options]
    rows[feature] = model_values

    log_preds = predictor.predict(rows)
    results = []
    for opt, lp in zip(options, log_preds):
        sal = round(10**lp)
        results.append({"option": opt, "salary": sal, "delta": sal - round(base_salary)})

    results.sort(key=lambda r: r["delta"], reverse=True)
    return results
