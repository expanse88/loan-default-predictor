import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import engineer  # noqa: E402

st.set_page_config(page_title="Loan Default Risk", page_icon="💳", layout="wide")


@st.cache_resource
def load():
    art = joblib.load(ROOT / "models" / "model.joblib")
    return art, shap.TreeExplainer(art["model"])


art, explainer = load()
st.title("💳 Loan Default Risk Predictor")
st.caption("LightGBM + isotonic calibration, trained on the 'Give Me Some Credit' "
           "dataset. Demo only: not for real lending decisions.")

with st.sidebar:
    st.header("Applicant profile")
    age = st.slider("Age", 18, 90, 40)
    income_known = st.checkbox("Monthly income known", True)
    income = st.number_input("Monthly income", 0, 100000, 5000, 250,
                             disabled=not income_known)
    dependents = st.slider("Dependents", 0, 10, 1)
    util = st.slider("Credit utilization (balance / limit)", 0.0, 2.0, 0.3, 0.01)
    debt_ratio = st.slider("Debt ratio (debt payments / income)", 0.0, 5.0, 0.35, 0.01)
    lines = st.slider("Open credit lines & loans", 0, 40, 8)
    re_loans = st.slider("Real estate loans", 0, 10, 1)
    st.subheader("Payment history (last 2 years)")
    pd_30 = st.slider("Times 30-59 days past due", 0, 10, 0)
    pd_60 = st.slider("Times 60-89 days past due", 0, 10, 0)
    late_90 = st.slider("Times 90+ days late", 0, 10, 0)

row = pd.DataFrame([{
    "utilization": util, "age": age, "past_due_30_59": pd_30, "debt_ratio": debt_ratio,
    "monthly_income": income if income_known else float("nan"),
    "open_credit_lines": lines, "late_90": late_90, "real_estate_loans": re_loans,
    "past_due_60_89": pd_60, "dependents": dependents}])
X = engineer(row)[art["features"]]

raw = art["model"].predict_proba(X)[:, 1]
prob = float(min(max(art["calibrator"].predict(raw)[0], 0.001), 0.99))
threshold = art["threshold"]
approve = prob < threshold

c1, c2, c3 = st.columns(3)
c1.metric("Estimated default probability", f"{prob:.1%}")
c2.metric("Decision threshold", f"{threshold:.1%}")
c3.metric("Recommendation", "✅ Approve" if approve else "⚠️ Review / decline")

st.subheader("Why this score?")
exp = explainer(X)
if exp.values.ndim == 3:
    exp = exp[:, :, 1]
plt.figure()
shap.plots.waterfall(exp[0], max_display=10, show=False)
st.pyplot(plt.gcf(), clear_figure=True)
st.caption("SHAP values are in model log-odds. Red bars push risk up, blue bars push it down.")

with st.expander("Model performance (held-out test set)"):
    m = art["metrics"]
    st.json({"test": m["test"], "decision": m["decision"]})
