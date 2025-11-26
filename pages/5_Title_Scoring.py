import streamlit as st
import pandas as pd
from data.loader import load_history_sales
from data.features import derive_basic_features, apply_registry_renames
from ml.scoring import load_model, score_dataframe
from pathlib import Path
import joblib

st.set_page_config(page_title="Title Scoring", layout="wide")
st.title("Title Scoring – Forecast")

st.caption(
    "This page displays all historical titles and allows you to score them using "
    "the trained ML model. Train a model first using the **Model Training** page."
)

df = apply_registry_renames(load_history_sales())
df = derive_basic_features(df)

# Show count of all titles
st.info(f"📊 Showing **{len(df)}** titles from historical data")

# Display ALL rows, not just head(20)
st.dataframe(df, use_container_width=True, height=600)

# Check if model exists
model_path = Path(__file__).parent.parent / "models" / "title_demand_rf.pkl"
if not model_path.exists():
    st.warning("No trained model found. Please train a model first using the Model Training page.")
else:
    if st.button("Score ALL Titles"):
        try:
            model = load_model()
            # Get the feature names from the trained model's preprocessor
            # The model is a Pipeline with 'pre' (ColumnTransformer) step
            preprocessor = model.named_steps.get('pre')
            if preprocessor is not None:
                # Extract feature names from the ColumnTransformer
                num_features = preprocessor.transformers_[0][2] if len(preprocessor.transformers_) > 0 else []
                cat_features = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
                all_features = list(num_features) + list(cat_features)
                # Filter to only columns that exist in the current dataframe
                available_features = [f for f in all_features if f in df.columns]
                if available_features:
                    df_to_score = df[available_features]
                else:
                    # Fallback to numeric/categorical columns
                    df_to_score = df.select_dtypes(include=["number", "object"])
            else:
                df_to_score = df.select_dtypes(include=["number", "object"])
            
            preds = score_dataframe(df_to_score, model)
            st.write(preds.to_frame())
        except Exception as e:
            st.error(f"Error scoring: {e}")
