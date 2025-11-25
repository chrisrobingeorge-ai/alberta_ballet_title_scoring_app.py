import streamlit as st
import pandas as pd
from data.loader import load_history_sales
from data.features import derive_basic_features, apply_registry_renames
from ml.scoring import load_model, score_dataframe
from pathlib import Path

st.set_page_config(page_title="Title Scoring", layout="wide")
st.title("Title Scoring – Forecast (Demo)")

df = apply_registry_renames(load_history_sales())
df = derive_basic_features(df)
st.dataframe(df.head(20), use_container_width=True)

# Check if model exists
model_path = Path(__file__).parent.parent / "models" / "title_demand_rf.pkl"
if not model_path.exists():
    st.warning("No trained model found. Please train a model first using the Model Training page.")
else:
    if st.button("Score displayed rows"):
        try:
            model = load_model()
            # For demo, score using numeric/categorical columns present
            # In production, filter to the exact training feature set.
            preds = score_dataframe(df.select_dtypes(include=["number", "object"]))
            st.write(preds.to_frame())
        except Exception as e:
            st.error(f"Error scoring: {e}")
