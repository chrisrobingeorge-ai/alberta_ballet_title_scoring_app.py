# train_pycaret_model.py
#
# One-time script to train a PyCaret regression model from history_city_sales.csv
# and save it as "title_demand_model.pkl" for the Streamlit app.

import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# 1. Load your historical data
df = pd.read_csv("data/history_city_sales.csv")

# 2. Make sure we have a target column called Total_Tickets
#    Adjust this logic if your column names are different.
if "Total_Tickets" not in df.columns:
    has_single = "Single_Tickets" in df.columns
    has_subs = "Subscription_Tickets" in df.columns

    if has_single and has_subs:
        df["Total_Tickets"] = df["Single_Tickets"] + df["Subscription_Tickets"]
        print("Created Total_Tickets = Single_Tickets + Subscription_Tickets")
    else:
        raise ValueError(
            "Cannot find a Total_Tickets column, or Single_Tickets + "
            "Subscription_Tickets to build it from. "
            "Please update this script to use your actual target column name."
        )

# 3. Set up PyCaret
s = setup(
    data=df,
    target="Total_Tickets",
    session_id=42,
    normalize=True,
    feature_selection=True,
    remove_multicollinearity=True,
    silent=True,
    verbose=False,
)

# 4. Let PyCaret try different models and pick the best by MAE
best_model = compare_models(n_select=1, sort="MAE")

# 5. Save the best model
save_model(best_model, "title_demand_model")
print("\n✓ Model saved as 'title_demand_model.pkl' in this folder.")
print("   Add/commit this file next to streamlit_app.py so the app can load it.")
