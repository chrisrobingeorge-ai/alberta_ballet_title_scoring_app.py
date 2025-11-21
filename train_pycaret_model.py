# train_pycaret_model.py
#
# One-time script to train a PyCaret regression model from history_city_sales.csv
# and save it as "title_demand_model.pkl" for the Streamlit app.

import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# 1. Load your historical data
df = pd.read_csv("data/history_city_sales.csv", thousands=",")

# 2. Rename columns to remove spaces (PyCaret / LightGBM prefer this)
df.columns = [c.replace(" ", "_") for c in df.columns]

# Now your columns are:
# - Show_Title
# - Single_Tickets_-_Calgary
# - Single_Tickets_-_Edmonton
# - Subscription_Tickets_-_Calgary
# - Subscription_Tickets_-_Edmonton

ticket_cols = [
    "Single_Tickets_-_Calgary",
    "Single_Tickets_-_Edmonton",
    "Subscription_Tickets_-_Calgary",
    "Subscription_Tickets_-_Edmonton",
]

# 3. Make sure all ticket columns are numeric and treat blanks as 0
for col in ticket_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 4. Build a Total_Tickets column as the target
df["Total_Tickets"] = df[ticket_cols].sum(axis=1)

print("Preview of training data:")
print(df[["Show_Title"] + ticket_cols + ["Total_Tickets"]].head())

# 5. Set up PyCaret
s = setup(
    data=df,
    target="Total_Tickets",
    session_id=42,
    normalize=True,
    feature_selection=True,
    remove_multicollinearity=True,
    ignore_features=["Show_Title"],  # don't use raw title text as a feature
)

# 6. Let PyCaret try different models and pick the best by MAE
best_model = compare_models(n_select=1, sort="MAE")

# 7. Save the best model
save_model(best_model, "title_demand_model")
print("\n✓ Model saved as 'title_demand_model.pkl' in this folder.")
print("   Add/commit this file next to streamlit_app.py so the app can load it.")
