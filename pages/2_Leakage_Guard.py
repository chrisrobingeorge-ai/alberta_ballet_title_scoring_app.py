import streamlit as st
from config.registry import load_leakage_audit

st.set_page_config(page_title="Leakage Guard", layout="wide")
st.title("Leakage Guard – Allowed Features at Forecast Time")

st.markdown("""
### What is this page?
The **Leakage Guard** ensures the ML model doesn't "cheat" by using information 
that wouldn't be available when making a real forecast.

**Data Leakage** occurs when the model uses future information to predict outcomes, 
causing artificially high accuracy that won't hold in production.

### Why it matters:
- ✅ **Allowed features**: Things you know BEFORE the show runs (title, category, historical patterns)
- ❌ **Forbidden features**: Actual ticket sales, attendance, revenue (you're trying to predict these!)

This audit ensures the model only uses legitimate forecast-time predictors.
""")

audit = load_leakage_audit()
st.dataframe(audit, width='stretch')

st.subheader("✅ Allowed Features Only")
allowed = audit[audit["Allowed at Forecast Time (Y/N)"].str.upper() == "Y"]["Feature Name"].tolist()
st.success(f"**{len(allowed)}** features are safe to use for forecasting")
st.code("\n".join(allowed))
