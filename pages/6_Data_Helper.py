# data_helper_app.py
"""
External Data Helper for Title Scorer

This Streamlit page allows users to upload external data CSVs and merge them
with base titles data to produce model-ready feature tables for the ML pipeline.

Features:
- Smart auto-detection of file types based on column names
- Drag-and-drop any file and the system will identify what it is
- Comprehensive instructions and templates for each data category
"""

import streamlit as st
import pandas as pd
from io import StringIO
from typing import Optional

st.set_page_config(
    page_title="External Data Helper for Title Scorer",
    layout="wide"
)

# =============================================================================
# DATA CATEGORY DEFINITIONS
# =============================================================================
# Each category defines:
# - signature_columns: columns that identify this data type
# - required_columns: minimum columns needed for merging
# - optional_columns: additional columns that can be included
# - join_keys: how to merge with base data
# - description: what this data is for
# - sample_data: example rows to help users understand the format

DATA_CATEGORIES = {
    "base_titles": {
        "name": "Base Titles / Historical Sales",
        "signature_columns": ["show_title", "single_tickets"],
        "required_columns": ["show_title"],
        "optional_columns": [
            "season_year", "opening_date", "city", "venue_name",
            "single_tickets_calgary", "single_tickets_edmonton",
            "total_single_tickets", "performance_count", "run_type"
        ],
        "join_keys": None,  # This is the base, no join needed
        "description": """
**Base Titles Data** is your primary dataset with one row per show (or per show per city).
This is the foundation that all other data will be merged onto.

**Required columns:**
- `show_title` – The name of the production (e.g., "The Nutcracker", "Swan Lake")

**Recommended columns:**
- `season_year` – The year of the season (e.g., 2023, 2024)
- `opening_date` – When the show opened (used to derive season_year if missing)
- `city` – "Calgary" or "Edmonton" (if data is split by city)
- `single_tickets_calgary` / `single_tickets_edmonton` – Ticket sales by city
- `total_single_tickets` – Total single ticket sales
        """,
        "sample_data": pd.DataFrame({
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "season_year": [2023, 2023, 2024],
            "city": ["Calgary", "Calgary", "Calgary"],
            "total_single_tickets": [8500, 6200, 5800],
            "opening_date": ["2023-12-15", "2023-02-10", "2024-02-14"]
        })
    },
    "economic_indicators": {
        "name": "Economic Indicators",
        "signature_columns": ["unemployment", "cpi", "gdp", "oil_price", "exchange_rate"],
        "required_columns": ["year"],
        "optional_columns": [
            "alberta_unemployment_rate", "alberta_cpi_index",
            "alberta_real_gdp_growth_rate", "wti_oil_price_avg",
            "exchange_rate_cad_usd", "consumer_confidence_index"
        ],
        "join_keys": {"base": ["season_year"], "external": ["year"]},
        "description": """
**Economic Indicators** capture province-level economic conditions that may affect 
arts spending. One row per year.

**Sources:**
- Statistics Canada Labour Force Survey (unemployment)
- Statistics Canada Consumer Price Index (CPI)
- Alberta Treasury Board & Finance (GDP)
- U.S. EIA or Bank of Canada (oil prices)
- Bank of Canada (exchange rates)

**Required columns:**
- `year` – The calendar year (e.g., 2023, 2024)

**Recommended columns:**
- `alberta_unemployment_rate` – Unemployment rate (e.g., 5.8, 6.1)
- `alberta_cpi_index` – Consumer Price Index (e.g., 157.2)
- `alberta_real_gdp_growth_rate` – GDP growth percentage (e.g., 2.5)
- `wti_oil_price_avg` – Average WTI oil price in USD (e.g., 78.50)
- `exchange_rate_cad_usd` – CAD to USD exchange rate (e.g., 0.74)
        """,
        "sample_data": pd.DataFrame({
            "year": [2022, 2023, 2024],
            "alberta_unemployment_rate": [5.9, 5.7, 6.1],
            "alberta_cpi_index": [152.3, 157.2, 162.3],
            "alberta_real_gdp_growth_rate": [4.8, 2.5, 2.1],
            "wti_oil_price_avg": [94.53, 78.50, 82.00],
            "exchange_rate_cad_usd": [0.77, 0.74, 0.73]
        })
    },
    "demographics": {
        "name": "Demographics (City-Level)",
        "signature_columns": ["population", "income", "household"],
        "required_columns": ["year", "city"],
        "optional_columns": [
            "population_city", "median_household_income_city",
            "population_growth_rate", "median_age"
        ],
        "join_keys": {"base": ["season_year", "city"], "external": ["year", "city"]},
        "description": """
**Demographics** data provides city-level population and income statistics.
One row per city per year.

**Sources:**
- Statistics Canada Census and population estimates
- StatsCan income data by geography

**Required columns:**
- `year` – The calendar year
- `city` – City name matching your base data (e.g., "Calgary", "Edmonton")

**Recommended columns:**
- `population_city` – City population (e.g., 1,336,000)
- `median_household_income_city` – Median household income (e.g., 98500)
        """,
        "sample_data": pd.DataFrame({
            "year": [2023, 2023, 2024, 2024],
            "city": ["Calgary", "Edmonton", "Calgary", "Edmonton"],
            "population_city": [1336000, 1010000, 1356000, 1025000],
            "median_household_income_city": [98500, 92000, 101000, 94500]
        })
    },
    "tourism": {
        "name": "Tourism Visitation",
        "signature_columns": ["tourism", "visitation", "hotel", "visitor"],
        "required_columns": ["year", "city"],
        "optional_columns": [
            "tourism_visitation_index", "hotel_occupancy_rate",
            "visitor_count", "tourism_spending"
        ],
        "join_keys": {"base": ["season_year", "city"], "external": ["year", "city"]},
        "description": """
**Tourism** data captures visitor activity that may correlate with arts attendance.
One row per city per year.

**Sources:**
- Travel Alberta research & insights
- Municipal tourism statistics
- Hotel occupancy reports

**Required columns:**
- `year` – The calendar year
- `city` – City name matching your base data

**Recommended columns:**
- `tourism_visitation_index` – Relative visitation (100 = baseline year)
- `hotel_occupancy_rate` – Average hotel occupancy percentage
        """,
        "sample_data": pd.DataFrame({
            "year": [2023, 2023, 2024, 2024],
            "city": ["Calgary", "Edmonton", "Calgary", "Edmonton"],
            "tourism_visitation_index": [105, 98, 108, 101],
            "hotel_occupancy_rate": [68.5, 62.3, 71.2, 65.8]
        })
    },
    "google_trends": {
        "name": "Google Trends (Title Interest)",
        "signature_columns": ["trends", "search", "interest", "google"],
        "required_columns": ["year", "show_title"],
        "optional_columns": [
            "google_trends_title_interest", "search_volume",
            "trend_direction"
        ],
        "join_keys": {
            "base": ["season_year", "show_title"],
            "external": ["year", "show_title"]
        },
        "description": """
**Google Trends** data captures public interest in specific show titles.
One row per title per year.

**Sources:**
- Google Trends (https://trends.google.com)

**Required columns:**
- `year` – The calendar year
- `show_title` – Must match titles in your base data exactly

**Recommended columns:**
- `google_trends_title_interest` – Search interest score (0-100)
        """,
        "sample_data": pd.DataFrame({
            "year": [2023, 2023, 2023],
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "google_trends_title_interest": [85, 72, 58]
        })
    },
    "arts_sector": {
        "name": "Arts Sector Confidence",
        "signature_columns": ["arts", "confidence", "sector", "cultural"],
        "required_columns": ["year"],
        "optional_columns": [
            "arts_sector_confidence_index", "cultural_spending_index",
            "arts_funding_level"
        ],
        "join_keys": {"base": ["season_year"], "external": ["year"]},
        "description": """
**Arts Sector Confidence** captures the health of the broader arts ecosystem.
One row per year.

**Sources:**
- CADAC (Canadian Arts Data)
- Canada Council for the Arts reports
- Provincial arts funding data

**Required columns:**
- `year` – The calendar year

**Recommended columns:**
- `arts_sector_confidence_index` – Composite index (100 = baseline)
        """,
        "sample_data": pd.DataFrame({
            "year": [2022, 2023, 2024],
            "arts_sector_confidence_index": [92, 98, 101]
        })
    },
    "marketing": {
        "name": "Marketing & Promotions",
        "signature_columns": ["marketing", "campaign", "email", "social", "ad_spend"],
        "required_columns": ["show_title"],
        "optional_columns": [
            "marketing_budget_city", "digital_ad_spend", "email_campaign_count",
            "email_open_rate", "email_click_rate", "social_posts_count",
            "social_engagement_rate", "campaign_start_lead_time_days"
        ],
        "join_keys": {
            "base": ["show_title"],
            "external": ["show_title"]
        },
        "description": """
**Marketing & Promotions** data tracks campaign activity for each production.
One row per show (or per show per city if granular).

**Sources:**
- Internal marketing budget spreadsheets
- Email platforms (Mailchimp, WordFly, etc.)
- Social media analytics (Meta, Twitter/X)
- Ad platforms (Google Ads, Meta Ads)

**Required columns:**
- `show_title` – Must match titles in your base data

**Optional granularity columns:**
- `year` or `season_year` – If tracking across seasons
- `city` – If marketing differs by city

**Recommended columns:**
- `marketing_budget_city` – Total marketing spend
- `digital_ad_spend` – Paid digital advertising spend
- `email_campaign_count` – Number of email campaigns sent
- `social_engagement_rate` – Social media engagement percentage
        """,
        "sample_data": pd.DataFrame({
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "marketing_budget_city": [25000, 18000, 15000],
            "digital_ad_spend": [8000, 5000, 4000],
            "email_campaign_count": [12, 8, 6],
            "social_engagement_rate": [4.2, 3.8, 3.5]
        })
    },
    "production_attributes": {
        "name": "Production Attributes",
        "signature_columns": [
            "venue", "capacity", "run_type", "musical",
            "familiarity", "family"
        ],
        "required_columns": ["show_title"],
        "optional_columns": [
            "venue_name", "venue_capacity", "performance_count_city",
            "run_type", "musical_forces", "score_familiarity_index",
            "story_familiarity_index", "family_friendliness_flag",
            "average_base_ticket_price"
        ],
        "join_keys": {
            "base": ["show_title"],
            "external": ["show_title"]
        },
        "description": """
**Production Attributes** describes the characteristics of each show.
One row per show (or per show per city/venue).

**Sources:**
- Internal production planning documents
- Archtics/Ticketmaster venue configuration
- Artistic/marketing taxonomy

**Required columns:**
- `show_title` – The production name

**Recommended columns:**
- `venue_capacity` – Total seats available
- `performance_count_city` – Number of performances
- `run_type` – Category (e.g., "Classic", "Contemporary", "Family")
- `family_friendliness_flag` – 1 for family shows, 0 otherwise
- `average_base_ticket_price` – Average ticket price
        """,
        "sample_data": pd.DataFrame({
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "venue_capacity": [2500, 2500, 1800],
            "performance_count_city": [12, 6, 4],
            "run_type": ["Family Classic", "Romantic Tragedy", "Romantic Tragedy"],
            "family_friendliness_flag": [1, 0, 0],
            "average_base_ticket_price": [75.00, 85.00, 80.00]
        })
    },
    "timing_schedule": {
        "name": "Timing & Schedule Factors",
        "signature_columns": [
            "opening", "closing", "holiday", "school_break",
            "competing", "on_sale"
        ],
        "required_columns": ["show_title"],
        "optional_columns": [
            "opening_date", "closing_date", "days_on_sale_before_opening",
            "holiday_period_flag", "school_break_overlap_flag",
            "competing_major_event_flag", "weekday_vs_weekend_mix"
        ],
        "join_keys": {
            "base": ["show_title"],
            "external": ["show_title"]
        },
        "description": """
**Timing & Schedule Factors** captures when shows run and competing events.
One row per show (or per show per city).

**Sources:**
- Internal performance schedules
- Public holiday calendars
- School board calendars
- Sports/event schedules

**Required columns:**
- `show_title` – The production name

**Recommended columns:**
- `opening_date` – First performance date
- `closing_date` – Last performance date
- `holiday_period_flag` – 1 if runs during major holiday
- `school_break_overlap_flag` – 1 if overlaps with school break
- `competing_major_event_flag` – 1 if competing with major event
        """,
        "sample_data": pd.DataFrame({
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "opening_date": ["2023-12-15", "2023-02-10", "2024-02-14"],
            "closing_date": ["2023-12-24", "2023-02-18", "2024-02-18"],
            "holiday_period_flag": [1, 0, 1],
            "school_break_overlap_flag": [1, 0, 0],
            "competing_major_event_flag": [0, 0, 0]
        })
    },
    "audience_donor": {
        "name": "Audience & Donor Features",
        "signature_columns": [
            "audience", "donor", "patron", "subscriber",
            "engagement", "retention"
        ],
        "required_columns": ["show_title"],
        "optional_columns": [
            "subscriber_attendance_share", "donor_flag", "engagement_score",
            "new_vs_returning_attendee", "audience_age_bracket"
        ],
        "join_keys": {
            "base": ["show_title"],
            "external": ["show_title"]
        },
        "description": """
**Audience & Donor Features** captures patron demographics and behavior.
One row per show (aggregated from patron-level data).

**Sources:**
- Archtics/Ticketmaster CRM
- Donor/fundraising CRM
- Email and web analytics

**Required columns:**
- `show_title` – The production name

**Recommended columns:**
- `subscriber_attendance_share` – % of tickets from subscribers
- `new_vs_returning_attendee` – Ratio of new to returning patrons
- `donor_flag` – Indicator if show attracted donors
        """,
        "sample_data": pd.DataFrame({
            "show_title": ["The Nutcracker", "Swan Lake", "Romeo and Juliet"],
            "subscriber_attendance_share": [0.35, 0.45, 0.40],
            "new_vs_returning_attendee": [0.30, 0.25, 0.28],
            "donor_flag": [1, 1, 0]
        })
    }
}


# =============================================================================
# AUTO-DETECTION FUNCTIONS
# =============================================================================

def detect_file_category(df: pd.DataFrame) -> tuple[Optional[str], float, list[str]]:
    """
    Analyze a DataFrame's columns to determine which data category it belongs to.
    
    The detection uses a weighted scoring system:
    - Signature columns get highest weight (strong identifiers)
    - Required columns get medium weight
    - Optional columns get lower weight
    - Special "discriminator" keywords provide bonus points
    
    Returns:
        tuple of (category_key, confidence_score, matched_columns)
    """
    columns_lower = [c.lower().replace("_", " ").replace("-", " ") for c in df.columns]
    columns_str = " ".join(columns_lower)  # For quick keyword search
    
    # Discriminator keywords that strongly indicate a category
    # These help distinguish between categories with similar columns
    category_discriminators = {
        "base_titles": ["ticket", "sales", "single ticket", "subscription"],
        "economic_indicators": ["unemployment", "cpi", "gdp", "oil", "exchange"],
        "demographics": ["population", "income", "household"],
        "tourism": ["tourism", "visitation", "hotel", "visitor"],
        "google_trends": ["trends", "google", "search interest"],
        "arts_sector": ["arts sector", "confidence index", "cultural"],
        "marketing": ["marketing", "campaign", "email", "ad spend", "social"],
        "production_attributes": ["venue", "capacity", "familiarity", "pricing"],
        "timing_schedule": ["opening", "closing", "holiday", "school break"],
        "audience_donor": ["donor", "subscriber", "patron", "engagement"]
    }
    
    best_match = None
    best_score = 0.0
    best_matched = []
    
    for cat_key, cat_info in DATA_CATEGORIES.items():
        matched_cols = []
        score = 0.0
        
        # Check signature columns (highest weight - 0.3 each)
        sig_matches = 0
        for sig_col in cat_info.get("signature_columns", []):
            for actual_col in columns_lower:
                if sig_col.lower() in actual_col:
                    if actual_col not in matched_cols:
                        matched_cols.append(actual_col)
                    sig_matches += 1
                    break
        if cat_info.get("signature_columns"):
            score += (sig_matches / len(cat_info["signature_columns"])) * 0.4
        
        # Check required columns (medium weight - 0.25 each)
        req_matches = 0
        for req_col in cat_info.get("required_columns", []):
            req_lower = req_col.lower().replace("_", " ")
            for actual_col in columns_lower:
                if req_lower in actual_col or actual_col in req_lower:
                    if actual_col not in matched_cols:
                        matched_cols.append(actual_col)
                    req_matches += 1
                    break
        if cat_info.get("required_columns"):
            score += (req_matches / len(cat_info["required_columns"])) * 0.3
        
        # Check optional columns (lower weight)
        opt_matches = 0
        for opt_col in cat_info.get("optional_columns", []):
            opt_lower = opt_col.lower().replace("_", " ")
            for actual_col in columns_lower:
                if opt_lower == actual_col or opt_lower in actual_col:
                    if actual_col not in matched_cols:
                        matched_cols.append(actual_col)
                    opt_matches += 1
                    break
        if cat_info.get("optional_columns"):
            score += (opt_matches / len(cat_info["optional_columns"])) * 0.2
        
        # Discriminator bonus - strong keywords that uniquely identify this category
        discriminators = category_discriminators.get(cat_key, [])
        discriminator_matches = sum(1 for d in discriminators if d in columns_str)
        if discriminators and discriminator_matches > 0:
            score += (discriminator_matches / len(discriminators)) * 0.3
        
        # Penalty for categories that require certain columns but don't have them
        if cat_key != "base_titles":
            # If this looks like base data (has tickets/sales), penalize other categories
            has_ticket_cols = any("ticket" in c or "sales" in c for c in columns_lower)
            if has_ticket_cols:
                score *= 0.5
        
        if score > best_score:
            best_score = score
            best_match = cat_key
            best_matched = matched_cols
    
    return best_match, min(best_score, 1.0), best_matched


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lowercase."""
    df = df.copy()
    df.columns = [
        c.strip().lower()
        .replace(" - ", "_")
        .replace(" ", "_")
        .replace("-", "_")
        for c in df.columns
    ]
    return df


def try_infer_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Try to create a year column from date columns if missing."""
    df = df.copy()
    if "year" not in df.columns and "season_year" not in df.columns:
        # Look for date columns
        date_cols = [c for c in df.columns 
                     if "date" in c.lower() or "opening" in c.lower()]
        for date_col in date_cols:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df["year"] = df[date_col].dt.year
                return df
            except (ValueError, TypeError):
                continue
    return df


# =============================================================================
# STREAMLIT APP
# =============================================================================

st.title("🎭 External Data Helper for Alberta Ballet Title Scoring")

# Instructions section
with st.expander("📖 **How to Use This Tool** (Click to expand)", expanded=True):
    st.markdown("""
### Welcome to the Smart Data Helper!

This tool helps you build a complete dataset for the Title Scoring ML model by:
1. **Automatically detecting** what type of data you upload
2. **Merging** multiple data sources together
3. **Validating** that your data has the right columns

### Quick Start
1. **Drop any CSV file** in the upload area below
2. The system will **automatically identify** what type of data it is
3. Review the detected category and confirm or override
4. Repeat for all your data files
5. Download the merged dataset

### What Data Do You Need?

| Priority | Data Type | Purpose |
|----------|-----------|---------|
| **Required** | Base Titles | Your show list with sales data |
| Recommended | Economic Indicators | Unemployment, CPI, oil prices |
| Recommended | Demographics | City population and income |
| Optional | Marketing | Campaign data, ad spend |
| Optional | Production Attributes | Venue, pricing, show type |
| Optional | Timing/Schedule | Dates, holidays, competing events |
| Optional | Tourism | Visitor activity |
| Optional | Google Trends | Search interest by title |

### Tips
- **Column names are flexible** – the system looks for keywords, not exact names
- **You can upload files in any order** – the system will sort them out
- **Partial data is OK** – upload what you have; missing data uses defaults
    """)

st.markdown("---")

# =============================================================================
# SMART FILE UPLOAD SECTION
# =============================================================================

st.header("📁 Step 1 – Upload Your Data Files")

st.info("""
**Drop any CSV files here!** The system will automatically detect what type of data 
each file contains and help you merge them together.
""")

uploaded_files = st.file_uploader(
    "Upload one or more CSV files (data will be auto-detected)",
    type=["csv"],
    accept_multiple_files=True,
    key="smart_uploader"
)

# Store uploaded data in session state
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
if "base_df" not in st.session_state:
    st.session_state.base_df = None

if uploaded_files:
    st.subheader("📊 Detected Data Categories")
    
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        # Skip if already processed (to avoid re-processing on rerun)
        if file_key in st.session_state.uploaded_data:
            continue
        
        try:
            df = pd.read_csv(uploaded_file)
            df = normalize_column_names(df)
            df = try_infer_year_column(df)
            
            # Detect category
            detected_cat, confidence, matched_cols = detect_file_category(df)
            
            st.session_state.uploaded_data[file_key] = {
                "df": df,
                "detected_category": detected_cat,
                "confidence": confidence,
                "matched_columns": matched_cols,
                "confirmed_category": None
            }
        except pd.errors.EmptyDataError:
            st.error(f"❌ **{file_key}**: File appears to be empty")
        except pd.errors.ParserError as e:
            st.error(f"❌ **{file_key}**: Could not parse CSV - {str(e)[:100]}")
        except UnicodeDecodeError:
            st.error(f"❌ **{file_key}**: Encoding error. Try saving as UTF-8 CSV.")

# Display detected files
if st.session_state.uploaded_data:
    for file_key, file_info in st.session_state.uploaded_data.items():
        df = file_info["df"]
        detected_cat = file_info["detected_category"]
        confidence = file_info["confidence"]
        matched_cols = file_info["matched_columns"]
        
        with st.expander(f"📄 **{file_key}**", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if detected_cat:
                    cat_name = DATA_CATEGORIES[detected_cat]["name"]
                    if confidence > 0.3:
                        st.success(f"✅ Detected as: **{cat_name}** "
                                   f"(confidence: {confidence:.0%})")
                    else:
                        st.warning(f"🤔 Best guess: **{cat_name}** "
                                   f"(low confidence: {confidence:.0%})")
                else:
                    st.warning("⚠️ Could not auto-detect category")
                
                st.caption(f"Matched columns: {', '.join(matched_cols) if matched_cols else 'None'}")
            
            with col2:
                # Allow manual override
                category_options = ["(auto-detect)"] + [
                    DATA_CATEGORIES[k]["name"] for k in DATA_CATEGORIES.keys()
                ]
                selected = st.selectbox(
                    "Override category:",
                    category_options,
                    key=f"cat_select_{file_key}",
                    index=0
                )
                
                if selected != "(auto-detect)":
                    # Find the key for the selected name
                    for k, v in DATA_CATEGORIES.items():
                        if v["name"] == selected:
                            file_info["confirmed_category"] = k
                            break
                else:
                    file_info["confirmed_category"] = detected_cat
            
            # Preview data
            st.caption("Preview (first 5 rows):")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Columns: {', '.join(df.columns.tolist())}")

st.markdown("---")

# =============================================================================
# DATA CATEGORY REFERENCE
# =============================================================================

st.header("📚 Step 2 – Data Category Reference")

st.write("""
Not sure what columns you need? Click on any category below to see 
detailed requirements and sample data templates.
""")

# Create tabs for each category
tab_names = [DATA_CATEGORIES[k]["name"] for k in DATA_CATEGORIES.keys()]
tabs = st.tabs(tab_names)

for tab, (cat_key, cat_info) in zip(tabs, DATA_CATEGORIES.items()):
    with tab:
        st.markdown(cat_info["description"])
        
        st.subheader("📋 Sample Data Template")
        sample_df = cat_info.get("sample_data")
        if sample_df is not None:
            st.dataframe(sample_df, use_container_width=True)
            
            # Download template button
            csv_buffer = StringIO()
            sample_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=f"⬇️ Download {cat_info['name']} Template",
                data=csv_buffer.getvalue(),
                file_name=f"{cat_key}_template.csv",
                mime="text/csv",
                key=f"download_{cat_key}"
            )

st.markdown("---")

# =============================================================================
# MERGE AND DOWNLOAD
# =============================================================================

st.header("🔗 Step 3 – Merge Data and Download")

if not st.session_state.uploaded_data:
    st.info("👆 Upload some files in Step 1 to begin merging.")
else:
    # Find base data
    base_file = None
    base_df = None
    external_files = []
    
    for file_key, file_info in st.session_state.uploaded_data.items():
        category = file_info.get("confirmed_category") or file_info.get("detected_category")
        if category == "base_titles":
            base_file = file_key
            base_df = file_info["df"].copy()
        else:
            external_files.append((file_key, file_info, category))
    
    if base_df is None:
        st.warning("""
⚠️ **No base titles file detected.** 

Please upload a file with your show titles and sales data. It should contain 
at least a `show_title` column, and ideally `season_year` and sales columns.

You can also click the "Base Titles" tab above to download a template.
        """)
    else:
        st.success(f"✅ Using **{base_file}** as base data ({len(base_df)} rows)")
        
        # Ensure season_year exists
        if "season_year" not in base_df.columns:
            if "year" in base_df.columns:
                base_df["season_year"] = base_df["year"]
                st.info("Created `season_year` from `year` column.")
            elif "opening_date" in base_df.columns:
                try:
                    base_df["opening_date"] = pd.to_datetime(
                        base_df["opening_date"], errors="coerce"
                    )
                    base_df["season_year"] = base_df["opening_date"].dt.year
                    st.info("Created `season_year` from `opening_date` column.")
                except (ValueError, TypeError):
                    st.warning(
                        "Could not parse `opening_date`. Please add a "
                        "`season_year` column manually."
                    )
        
        # Merge external data
        merge_log = []
        for file_key, file_info, category in external_files:
            if category is None:
                merge_log.append(f"⏭️ Skipped **{file_key}**: No category detected")
                continue
            
            cat_info = DATA_CATEGORIES.get(category)
            if cat_info is None or cat_info.get("join_keys") is None:
                merge_log.append(f"⏭️ Skipped **{file_key}**: No join keys defined")
                continue
            
            ext_df = file_info["df"]
            join_keys = cat_info["join_keys"]
            
            # Check if base has required join keys
            missing_base_keys = [k for k in join_keys["base"] if k not in base_df.columns]
            missing_ext_keys = [k for k in join_keys["external"] if k not in ext_df.columns]
            
            if missing_base_keys:
                merge_log.append(
                    f"⚠️ **{file_key}**: Base missing join keys: {missing_base_keys}"
                )
                continue
            
            if missing_ext_keys:
                merge_log.append(
                    f"⚠️ **{file_key}**: File missing join keys: {missing_ext_keys}"
                )
                continue
            
            # Perform merge
            try:
                before_cols = set(base_df.columns)
                base_df = base_df.merge(
                    ext_df,
                    left_on=join_keys["base"],
                    right_on=join_keys["external"],
                    how="left",
                    suffixes=("", "_ext")
                )
                after_cols = set(base_df.columns)
                new_cols = list(after_cols - before_cols)
                
                merge_log.append(
                    f"✅ Merged **{file_key}** ({cat_info['name']}): "
                    f"+{len(new_cols)} columns"
                )
            except ValueError as e:
                merge_log.append(f"❌ **{file_key}**: Merge failed - {str(e)[:50]}")
        
        # Show merge log
        if merge_log:
            st.subheader("📝 Merge Log")
            for log_entry in merge_log:
                st.write(log_entry)
        
        st.markdown("---")
        
        # Preview merged data
        st.subheader("📊 Preview Merged Data")
        st.dataframe(base_df.head(20), use_container_width=True)
        st.caption(f"Total: {len(base_df)} rows × {len(base_df.columns)} columns")
        
        # Download button
        csv_buffer = StringIO()
        base_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Merged Feature CSV",
            data=csv_buffer.getvalue(),
            file_name="title_features_with_external_data.csv",
            mime="text/csv",
            key="download_merged"
        )
        
        # Store in session state for other pages
        st.session_state.base_df = base_df

st.markdown("---")

# =============================================================================
# HELP SECTION
# =============================================================================

with st.expander("❓ Troubleshooting & FAQ"):
    st.markdown("""
### Common Issues

**Q: My file wasn't detected correctly**  
A: Use the dropdown to manually select the correct category. The auto-detection 
   looks for keyword matches in column names.

**Q: The merge failed with "key not found"**  
A: Make sure your external data has the right join columns:
   - Economic/Arts data needs a `year` column
   - Demographics/Tourism need `year` and `city` columns
   - Per-show data needs `show_title` (must match base data exactly)

**Q: I don't have all the recommended columns**  
A: That's OK! Upload what you have. Missing columns will simply not be included 
   in the merged output. The ML model can work with partial data.

**Q: My city names don't match**  
A: Make sure cities are spelled consistently (e.g., always "Calgary" not 
   "calgary" or "CALGARY"). The system normalizes to lowercase internally, 
   but the values must still match.

**Q: How do I add new years of data?**  
A: Upload your updated files and re-run the merge. The download will include 
   all the latest data.

### Data Sources Reference

| Data Type | Where to Get It |
|-----------|-----------------|
| Economic | [StatsCan](https://www150.statcan.gc.ca/), [Alberta Economic Dashboard](https://economicdashboard.alberta.ca) |
| Demographics | [StatsCan Census](https://www12.statcan.gc.ca/census-recensement/) |
| Tourism | [Travel Alberta](https://industry.travelalberta.com/research) |
| Google Trends | [Google Trends](https://trends.google.com) |
| Oil Prices | [EIA](https://www.eia.gov/), [Bank of Canada](https://www.bankofcanada.ca) |
    """)
