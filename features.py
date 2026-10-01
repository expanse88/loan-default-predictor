"""Cleaning + feature engineering, shared by training and the Streamlit app
so there is no train/serve skew."""
import numpy as np
import pandas as pd

TARGET = "default"

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}

# In this dataset the values 96 and 98 are placeholder codes, not real counts.
PLACEHOLDER_COLS = ["past_due_30_59", "past_due_60_89", "late_90"]


def load_raw(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df.rename(columns=RENAME)


def clean(df):
    df = df.copy()
    df = df[df["age"] >= 18]
    for c in PLACEHOLDER_COLS:
        df.loc[df[c] >= 96, c] = np.nan
    return df


def engineer(df):
    """Raw (cleaned) columns -> model features. Safe to call on one row."""
    X = df.drop(columns=[TARGET], errors="ignore").copy()
    X["income_missing"] = X["monthly_income"].isna().astype(int)
    X["utilization"] = X["utilization"].clip(upper=2)
    X["debt_ratio"] = X["debt_ratio"].clip(upper=5)
    X["log_income"] = np.log1p(X["monthly_income"])
    X["income_per_person"] = X["monthly_income"] / (X["dependents"].fillna(0) + 1)
    X["total_past_due"] = X[PLACEHOLDER_COLS].fillna(0).sum(axis=1)
    return X.drop(columns=["monthly_income"])
