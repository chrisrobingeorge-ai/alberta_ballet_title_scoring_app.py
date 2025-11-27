# data_helper_app.py
"""
External Data Helper for Title Scorer

This Streamlit page allows users to upload external data CSVs and merge them
with base titles data to produce model-ready feature tables for the ML pipeline.
"""

import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="External Data Helper for Title Scorer",
    layout="wide"
)

st.title("External Data Helper for Alberta Ballet Title Scoring Model")
st.write(
    "Upload your **base titles file** and any external data CSVs "
    "to automatically build a merged, model-ready feature table."
)

# ------------------------
# 1. Upload base titles CSV
# ------------------------

st.header("Step 1 – Upload base titles file")

base_file = st.file_uploader(
    "Upload base titles CSV (one row per show per city)",
    type=["csv"],
    key="base_titles"
)

if base_file is not None:
    base_df = pd.read_csv(base_file)
    st.subheader("Preview: Base titles")
    st.dataframe(base_df.head())

    # Try to infer or create a season_year column if missing
    if "season_year" not in base_df.columns:
        if "opening_date" in base_df.columns:
            base_df["opening_date"] = pd.to_datetime(base_df["opening_date"])
            base_df["season_year"] = base_df["opening_date"].dt.year
            st.info("`season_year` column created from `opening_date`.")
        else:
            st.warning(
                "No `season_year` column found and no `opening_date` column "
                "to derive it from. Please add one in your base CSV."
            )
else:
    base_df = None

st.markdown("---")

# ------------------------
# Helper function to upload & merge external data
# ------------------------


def upload_and_merge_external(
    label: str,
    expected_cols: list,
    join_keys_base: list,
    join_keys_ext: list,
    df_base: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generic helper: upload one external CSV, check expected columns,
    and merge onto the base DF using provided join keys.
    """
    ext_file = st.file_uploader(
        f"Upload {label} CSV",
        type=["csv"],
        key=label.replace(" ", "_").lower()
    )
    if ext_file is None:
        st.info(f"No file uploaded for {label}. Skipping.")
        return df_base

    ext_df = pd.read_csv(ext_file)
    st.subheader(f"Preview: {label}")
    st.dataframe(ext_df.head())

    # Basic sanity check on columns
    missing = [c for c in expected_cols if c not in ext_df.columns]
    if missing:
        st.error(
            f"{label}: Missing expected columns in external CSV: {missing}.\n"
            f"Found columns: {list(ext_df.columns)}"
        )
        return df_base

    # Show join mapping for transparency
    st.caption(
        f"{label} – Merging base[{join_keys_base}] "
        f"with external[{join_keys_ext}]"
    )

    merged = df_base.merge(
        ext_df,
        left_on=join_keys_base,
        right_on=join_keys_ext,
        how="left"
    )
    st.success(f"{label}: Merge complete. New columns added: "
               f"{[c for c in merged.columns if c not in df_base.columns]}")
    return merged


# ------------------------
# 2. External data sections
# ------------------------

if base_df is not None:
    st.header("Step 2 – Upload external datasets")

    # 2.1 Economic indicators (province-level, by year)
    with st.expander("External – Economic indicators (unemployment, CPI, GDP, oil, FX)", expanded=True):
        st.write(
            "CSV at **year granularity** (e.g., one row per year) with columns like:\n"
            "`year, alberta_unemployment_rate, alberta_cpi_index, "
            "alberta_real_gdp_growth_rate, wti_oil_price_avg, exchange_rate_cad_usd`."
        )
        econ_expected_cols = [
            "year",
            "alberta_unemployment_rate",
            "alberta_cpi_index",
            "alberta_real_gdp_growth_rate",
            "wti_oil_price_avg",
            "exchange_rate_cad_usd",
        ]
        base_df = upload_and_merge_external(
            label="Economic indicators",
            expected_cols=econ_expected_cols,
            join_keys_base=["season_year"],
            join_keys_ext=["year"],
            df_base=base_df,
        )

    # 2.2 Demographics (by city and year)
    with st.expander("External – Demographics (population, median income)", expanded=True):
        st.write(
            "CSV with **city-year** rows, e.g.: "
            "`year, city, population_city, median_household_income_city`.\n"
            "City values should match your base file (e.g. 'Calgary', 'Edmonton')."
        )
        demo_expected_cols = [
            "year",
            "city",
            "population_city",
            "median_household_income_city",
        ]
        base_df = upload_and_merge_external(
            label="Demographics",
            expected_cols=demo_expected_cols,
            join_keys_base=["season_year", "city"],
            join_keys_ext=["year", "city"],
            df_base=base_df,
        )

    # 2.3 Tourism (by city and year)
    with st.expander("External – Tourism visitation index", expanded=True):
        st.write(
            "CSV with **city-year** tourism indicators, e.g.: "
            "`year, city, tourism_visitation_index`."
        )
        tourism_expected_cols = [
            "year",
            "city",
            "tourism_visitation_index",
        ]
        base_df = upload_and_merge_external(
            label="Tourism visitation",
            expected_cols=tourism_expected_cols,
            join_keys_base=["season_year", "city"],
            join_keys_ext=["year", "city"],
            df_base=base_df,
        )

    # 2.4 Google Trends for titles (by title and year)
    with st.expander("External – Google Trends (title interest)", expanded=True):
        st.write(
            "CSV with **title-year** search interest, e.g.: "
            "`year, show_title, google_trends_title_interest`.\n"
            "Make sure `show_title` matches the base file exactly or add a key column."
        )
        trends_expected_cols = [
            "year",
            "show_title",
            "google_trends_title_interest",
        ]
        base_df = upload_and_merge_external(
            label="Google Trends (title interest)",
            expected_cols=trends_expected_cols,
            join_keys_base=["season_year", "show_title"],
            join_keys_ext=["year", "show_title"],
            df_base=base_df,
        )

    # 2.5 Arts sector confidence index (if you build one)
    with st.expander("External – Arts sector confidence index (optional)", expanded=False):
        st.write(
            "Optional CSV, e.g.: `year, arts_sector_confidence_index`.\n"
            "You can derive this from CADAC, Canada Council reports, etc."
        )
        arts_expected_cols = [
            "year",
            "arts_sector_confidence_index",
        ]
        base_df = upload_and_merge_external(
            label="Arts sector confidence",
            expected_cols=arts_expected_cols,
            join_keys_base=["season_year"],
            join_keys_ext=["year"],
            df_base=base_df,
        )

    st.markdown("---")

    # ------------------------
    # 3. Preview & download merged features
    # ------------------------

    st.header("Step 3 – Review and download merged feature table")

    st.subheader("Preview merged data")
    st.dataframe(base_df.head(50))

    csv_buffer = StringIO()
    base_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download merged feature CSV",
        data=csv_buffer.getvalue(),
        file_name="title_features_with_external_data.csv",
        mime="text/csv",
    )
else:
    st.info("Upload your base titles file in Step 1 to enable external merges.")
