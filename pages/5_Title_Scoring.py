import streamlit as st
import pandas as pd
from data.loader import load_history_sales
from data.features import derive_basic_features, apply_registry_renames
from ml.scoring import load_model, score_dataframe
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Title Scoring", layout="wide")
st.title("Title Scoring – Forecast")

st.caption(
    "This page displays all historical titles and allows you to score them using "
    "the trained ML model. Train a model first using the **Model Training** page."
)

# --- Column Definitions (for reference) ---
# The data includes several ticket-related columns:
#
# FROM HISTORICAL DATA (history_city_sales.csv):
#   - single_tickets_calgary/edmonton: Actual single tickets sold per city
#   - subscription_tickets_calgary/edmonton: Actual subscription tickets per city
#   - total_single_tickets: Actual total single tickets sold (Calgary + Edmonton)
#   - yourmodel_single_tickets_calgary/edmonton: Historical model predictions per city
#   - yourmodel_total_single_tickets: Historical model's predicted total single tickets
#
# DERIVED FEATURES (computed by derive_basic_features):
#   - total_subscription_tickets: Sum of subscription tickets across both cities
#   - total_tickets_all: Grand total (total_single_tickets + total_subscription_tickets)
#
# ML MODEL OUTPUT (from score_dataframe):
#   - forecast_single_tickets: ML model's prediction for single ticket sales
#
# ACCURACY INTERPRETATION:
#   Compare yourmodel_total_single_tickets vs total_single_tickets to assess
#   historical model accuracy. Compare forecast_single_tickets vs total_single_tickets
#   for the ML model's predictions.

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
    if st.button("Score displayed rows"):
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
            
            # Build results DataFrame with show titles and key context columns
            result_df = pd.DataFrame({
                "show_title": df["show_title"] if "show_title" in df.columns else df.index,
                "forecast_single_tickets": preds.values,
            })
            
            # Add actual tickets for comparison if available
            if "total_single_tickets" in df.columns:
                result_df["actual_total_single_tickets"] = df["total_single_tickets"].values
            if "yourmodel_total_single_tickets" in df.columns:
                result_df["yourmodel_total_single_tickets"] = df["yourmodel_total_single_tickets"].values
            if "total_tickets_all" in df.columns:
                result_df["total_tickets_all"] = df["total_tickets_all"].values
            
            st.subheader("Scoring Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Show accuracy metrics if actual data is available
            if "total_single_tickets" in df.columns:
                actual = df["total_single_tickets"].dropna()
                predicted = preds.loc[actual.index]
                if len(actual) > 0:
                    mae = mean_absolute_error(actual, predicted)
                    r2 = r2_score(actual, predicted)
                    st.caption(
                        f"**ML Model (forecast_single_tickets):** MAE = {mae:,.0f} tickets, R² = {r2:.3f}"
                    )
                    
                    # Compare with yourmodel if available
                    if "yourmodel_total_single_tickets" in df.columns:
                        ym = df.loc[actual.index, "yourmodel_total_single_tickets"].dropna()
                        if len(ym) > 0:
                            ym_actual = actual.loc[ym.index]
                            ym_mae = mean_absolute_error(ym_actual, ym)
                            ym_r2 = r2_score(ym_actual, ym)
                            st.caption(
                                f"**Historical Model (yourmodel_total_single_tickets):** MAE = {ym_mae:,.0f} tickets, R² = {ym_r2:.3f}"
                            )
        except Exception as e:
            st.error(f"Error scoring: {e}")
