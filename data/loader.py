from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_history_sales(csv_name: str = "history_city_sales.csv") -> pd.DataFrame:
    """Load historical show-level sales (Calgary/Edmonton)."""
    path = DATA_DIR / csv_name
    df = pd.read_csv(path, thousands=",")
    # Normalize column names: replace " - " with "_", then remaining spaces/dashes with "_"
    # This ensures "Single Tickets - Calgary" becomes "single_tickets_calgary" not "single_tickets___calgary"
    df.columns = [
        c.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    # Expected columns: show_title, single_tickets_calgary, single_tickets_edmonton,
    # subscription_tickets_calgary, subscription_tickets_edmonton, total_single_tickets
    return df


# =============================================================================
# EXTERNAL FACTORS INTEGRATION
# =============================================================================
# To integrate external factors (economic indicators, weather, etc.) into the
# model, you will need to:
#
# 1. CREATE CSV FILES in the data/ directory with external factor data.
#    Example files to create:
#      - data/external_economic.csv (alberta_unemployment_rate, alberta_cpi_index, etc.)
#      - data/external_weather.csv (weather_severity_index by city/date)
#
# 2. ADD LOADER FUNCTIONS below to read each external factor file.
#
# 3. MERGE the data in build_modelling_dataset() or a similar function.
#    The merge key will typically be:
#      - show_title + production_season (for per-show factors)
#      - opening_date or month_of_opening (for time-based factors)
#      - city (for city-specific factors)
#
# EXAMPLE: To add economic factors, create data/external_economic.csv with:
#   production_season,alberta_unemployment_rate,alberta_cpi_index,wti_oil_price_avg
#   2023-24,5.8,157.2,78.50
#   2024-25,6.1,162.3,82.00
#
# Then add a loader function:
#
# def load_external_economic(csv_name: str = "external_economic.csv") -> pd.DataFrame:
#     """Load external economic indicators by production season."""
#     path = DATA_DIR / csv_name
#     if not path.exists():
#         return pd.DataFrame()  # Return empty if file doesn't exist
#     df = pd.read_csv(path)
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
#     return df
#
# And merge in your dataset builder:
#
# def build_dataset_with_external():
#     sales = load_history_sales()
#     economic = load_external_economic()
#     if not economic.empty:
#         sales = sales.merge(economic, on="production_season", how="left")
#     return sales
#
# PROMPT FOR FUTURE INTEGRATION:
# "I've created CSV files for external factors. Please update data/loader.py
# to add loader functions for each file, then update ml/dataset.py to merge
# these factors into the modelling dataset using the appropriate join keys.
# The files are: [list your file names and their join keys]"
# =============================================================================
