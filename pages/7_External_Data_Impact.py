# pages/7_External_Data_Impact.py
"""
External Data Impact Dashboard

This page provides visibility into how external CSV data sources affect
estimated ticket numbers. It shows:
1. Summary of all loaded external data files
2. Current economic sentiment factor and its components
3. Impact breakdown showing how each data source contributes
4. Historical trends from external data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

# Configure page
st.set_page_config(
    page_title="External Data Impact — Alberta Ballet",
    page_icon="📊",
    layout="wide"
)

# Import data loaders
try:
    from data.loader import (
        load_oil_prices,
        load_unemployment_rates,
        get_economic_sentiment_factor,
        load_external_factors,
        DATA_DIR,
    )
    DATA_LOADERS_AVAILABLE = True
except ImportError as e:
    DATA_LOADERS_AVAILABLE = False
    st.error(f"Could not import data loaders: {e}")

# =============================================================================
# DATA FILE REGISTRY
# =============================================================================

# Define all external data files with their metadata
EXTERNAL_DATA_FILES = {
    # Economic indicators
    "commodity_price_index.csv": {
        "category": "Economics",
        "description": "Bank of Canada Commodity Price Index (BCPI) covering energy, metals, agriculture",
        "path": "economics/commodity_price_index.csv",
        "impact_type": "economic_sentiment",
        "key_columns": ["date", "A.BCPI", "A.ENER"],
    },
    "boc_annual_fx_rates.csv": {
        "category": "Economics",
        "description": "Bank of Canada annual exchange rates (CAD vs major currencies)",
        "path": "economics/boc_annual_fx_rates.csv",
        "impact_type": "economic_sentiment",
        "key_columns": ["date", "FXAUSDCAD"],
    },
    "boc_legacy_annual_rates.csv": {
        "category": "Economics",
        "description": "Historical Bank of Canada annual interest rates",
        "path": "economics/boc_legacy_annual_rates.csv",
        "impact_type": "economic_sentiment",
        "key_columns": ["date"],
    },
    "boc_cpi_monthly.csv": {
        "category": "Economics",
        "description": "Monthly Consumer Price Index from Bank of Canada",
        "path": "economics/boc_cpi_monthly.csv",
        "impact_type": "economic_sentiment",
        "key_columns": ["date"],
    },
    "oil_price.csv": {
        "category": "Economics",
        "description": "WCS and WTI oil prices (USD) - key Alberta economic indicator",
        "path": "economics/oil_price.csv",
        "impact_type": "oil_factor",
        "key_columns": ["date", "wcs_oil_price", "oil_series"],
    },
    "unemployment_by_city.csv": {
        "category": "Economics",
        "description": "Unemployment rates by city (Calgary, Edmonton, Alberta)",
        "path": "economics/unemployment_by_city.csv",
        "impact_type": "unemployment_factor",
        "key_columns": ["date", "unemployment_rate", "region"],
    },
    # Consumer sentiment
    "nanos_consumer_confidence.csv": {
        "category": "Consumer Sentiment",
        "description": "Nanos Bloomberg Consumer Confidence Index (BNCCI)",
        "path": "economics/nanos_consumer_confidence.csv",
        "impact_type": "consumer_confidence",
        "key_columns": ["category", "metric", "year_or_period", "value"],
    },
    "nanos_better_off.csv": {
        "category": "Consumer Sentiment",
        "description": "Nanos survey on Canadians feeling better/worse off financially",
        "path": "economics/nanos_better_off.csv",
        "impact_type": "consumer_confidence",
        "key_columns": ["category", "metric", "value"],
    },
    # Arts & Culture
    "nanos_arts_donors.csv": {
        "category": "Arts Engagement",
        "description": "Nanos survey on arts donation patterns and giving by demographic",
        "path": "audiences/nanos_arts_donors.csv",
        "impact_type": "arts_sector",
        "key_columns": ["section", "metric", "year_or_period", "value"],
    },
    "canada_council_scorecard.csv": {
        "category": "Arts Funding",
        "description": "Canada Council for the Arts annual funding scorecard",
        "path": "arts_funding/canada_council_scorecard.csv",
        "impact_type": "arts_sector",
        "key_columns": ["category", "subcategory", "metric", "year", "value"],
    },
    "canada_council_outreach.csv": {
        "category": "Arts Funding",
        "description": "Canada Council outreach program statistics",
        "path": "arts_funding/canada_council_outreach.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    "canada_council_touring.csv": {
        "category": "Arts Funding",
        "description": "Canada Council touring and travel support data",
        "path": "arts_funding/canada_council_touring.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    "canada_council_explore_create.csv": {
        "category": "Arts Funding",
        "description": "Canada Council Explore and Create grant programs",
        "path": "arts_funding/canada_council_explore_create.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    # Stats Canada Culture
    "stats_can_culture_sports_trade_provincial.csv": {
        "category": "Culture Statistics",
        "description": "Statistics Canada culture, sports & trade data by province",
        "path": "economics/stats_can_culture_sports_trade_provincial.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    "stats_can_culture_sports_trade_domain.csv": {
        "category": "Culture Statistics",
        "description": "Statistics Canada culture indicators by domain",
        "path": "economics/stats_can_culture_sports_trade_domain.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    "stats_can_culture_sports_trade_gdp.csv": {
        "category": "Culture Statistics",
        "description": "Culture sector GDP contribution from Statistics Canada",
        "path": "economics/stats_can_culture_sports_trade_gdp.csv",
        "impact_type": "arts_sector",
        "key_columns": [],
    },
    # Tourism
    "travel_alberta_tourism_impact.csv": {
        "category": "Tourism",
        "description": "Travel Alberta tourism impact and visitation data",
        "path": "economics/travel_alberta_tourism_impact.csv",
        "impact_type": "tourism",
        "key_columns": ["category", "subcategory", "metric_name", "metric_value"],
    },
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    if DATA_LOADERS_AVAILABLE:
        return DATA_DIR
    return Path(__file__).parent.parent / "data"


def load_external_file(file_info: dict) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load an external data file and return (df, error_message)."""
    data_dir = get_data_dir()
    file_path = data_dir / file_info["path"]
    
    if not file_path.exists():
        return None, f"File not found: {file_path}"
    
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, f"Error loading: {str(e)[:100]}"


def get_file_stats(df: pd.DataFrame) -> dict:
    """Get basic statistics about a dataframe."""
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_list": list(df.columns),
    }
    
    # Try to get date range
    for col in df.columns:
        if "date" in col.lower() or "year" in col.lower():
            try:
                if "date" in col.lower():
                    dates = pd.to_datetime(df[col], errors="coerce")
                    stats["date_range"] = f"{dates.min()} to {dates.max()}"
                elif "year" in col.lower():
                    years = pd.to_numeric(df[col], errors="coerce")
                    stats["date_range"] = f"{int(years.min())} to {int(years.max())}"
                break
            except Exception:
                continue
    
    return stats


def calculate_current_economic_impact() -> dict:
    """Calculate the current economic sentiment and its components."""
    impact = {
        "sentiment_factor": 1.0,
        "oil_factor": 1.0,
        "unemployment_factor": 1.0,
        "combined_factor": 1.0,
        "oil_price": None,
        "unemployment_rate": None,
        "baseline_oil": 60.0,
        "baseline_unemployment": 6.0,
    }
    
    if not DATA_LOADERS_AVAILABLE:
        return impact
    
    try:
        # Get current sentiment factor
        impact["sentiment_factor"] = get_economic_sentiment_factor(
            run_date=pd.Timestamp.now(),
            city=None,
            baseline_oil_price=60.0,
            baseline_unemployment=6.0,
            oil_weight=0.4,
            unemployment_weight=0.6,
            min_factor=0.85,
            max_factor=1.10
        )
        
        # Load oil prices for latest value
        oil_df = load_oil_prices()
        if not oil_df.empty and "date" in oil_df.columns:
            oil_df["date"] = pd.to_datetime(oil_df["date"], errors="coerce")
            wcs_df = oil_df[oil_df.get("oil_series", "") == "WCS"].copy()
            if wcs_df.empty:
                wcs_df = oil_df.copy()
            recent = wcs_df.sort_values("date", ascending=False)
            if not recent.empty:
                for col in ["wcs_oil_price", "oil_price", "price"]:
                    if col in recent.columns:
                        impact["oil_price"] = float(recent.iloc[0][col])
                        # Calculate oil factor component
                        if impact["oil_price"] > 0:
                            impact["oil_factor"] = 0.8 + 0.4 * min(1.5, max(0.5, impact["oil_price"] / 60.0))
                        break
        
        # Load unemployment for latest value
        unemp_df = load_unemployment_rates()
        if not unemp_df.empty and "date" in unemp_df.columns:
            unemp_df["date"] = pd.to_datetime(unemp_df["date"], errors="coerce")
            # Filter to Alberta
            if "region" in unemp_df.columns:
                alberta_df = unemp_df[unemp_df["region"] == "Alberta"]
                if alberta_df.empty:
                    alberta_df = unemp_df
            else:
                alberta_df = unemp_df
            recent = alberta_df.sort_values("date", ascending=False)
            if not recent.empty and "unemployment_rate" in recent.columns:
                impact["unemployment_rate"] = float(recent.iloc[0]["unemployment_rate"])
                # Calculate unemployment factor component
                if impact["unemployment_rate"] > 0:
                    impact["unemployment_factor"] = 0.8 + 0.4 * min(1.5, max(0.5, 6.0 / impact["unemployment_rate"]))
        
        # Calculate combined factor (weighted average)
        impact["combined_factor"] = 0.4 * impact["oil_factor"] + 0.6 * impact["unemployment_factor"]
        
    except Exception as e:
        st.warning(f"Error calculating economic impact: {e}")
    
    return impact


def get_nanos_consumer_confidence_summary() -> dict:
    """Extract key metrics from Nanos consumer confidence data."""
    summary = {}
    data_dir = get_data_dir()
    file_path = data_dir / "economics/nanos_consumer_confidence.csv"
    
    if not file_path.exists():
        return summary
    
    try:
        df = pd.read_csv(file_path)
        
        # Get headline index
        headline = df[(df["category"] == "BNCCI") & (df["subcategory"] == "Headline Index")]
        if not headline.empty:
            this_week = headline[headline["metric"] == "This week"]
            if not this_week.empty:
                summary["current_index"] = float(this_week.iloc[0]["value"])
            
            avg_2025 = headline[headline["metric"] == "2025 average"]
            if not avg_2025.empty:
                summary["avg_2025"] = float(avg_2025.iloc[0]["value"])
            
            overall_avg = headline[headline["metric"] == "Overall index average"]
            if not overall_avg.empty:
                summary["overall_avg"] = float(overall_avg.iloc[0]["value"])
            
            record_low = headline[headline["metric"] == "Record low"]
            if not record_low.empty:
                summary["record_low"] = float(record_low.iloc[0]["value"])
            
            record_high = headline[headline["metric"] == "Record high"]
            if not record_high.empty:
                summary["record_high"] = float(record_high.iloc[0]["value"])
        
        # Get regional breakdown (Prairies)
        regional = df[(df["category"] == "Demographics") & (df["subcategory"] == "Region")]
        prairies = regional[regional["metric"] == "Prairies"]
        if not prairies.empty:
            summary["prairies_index"] = float(prairies.iloc[0]["value"])
    
    except Exception as e:
        st.warning(f"Error parsing Nanos data: {e}")
    
    return summary


def get_arts_funding_summary() -> dict:
    """Extract key metrics from Canada Council data."""
    summary = {}
    data_dir = get_data_dir()
    file_path = data_dir / "arts_funding/canada_council_scorecard.csv"
    
    if not file_path.exists():
        return summary
    
    try:
        df = pd.read_csv(file_path)
        
        # Get total support funding
        total_support = df[
            (df["category"] == "Total support") & 
            (df["subcategory"] == "All recipients") &
            (df["metric"] == "Total support funding")
        ]
        if not total_support.empty:
            latest = total_support.sort_values("year", ascending=False).iloc[0]
            summary["total_funding"] = float(latest["value"])
            summary["funding_year"] = latest["year"]
        
        # Get granting programs data
        core_grants = df[
            (df["category"] == "Granting Programs") & 
            (df["subcategory"] == "Core Grants") &
            (df["metric"] == "Funding")
        ]
        if not core_grants.empty:
            latest = core_grants.sort_values("year", ascending=False).iloc[0]
            summary["core_grants"] = float(latest["value"])
    
    except Exception as e:
        st.warning(f"Error parsing Canada Council data: {e}")
    
    return summary


# =============================================================================
# PAGE CONTENT
# =============================================================================

st.title("📊 External Data Impact Dashboard")
st.caption("See how external CSV data sources affect estimated ticket numbers")

# Summary tabs
tab_overview, tab_economic, tab_sentiment, tab_arts, tab_files = st.tabs([
    "📈 Impact Overview",
    "💰 Economic Factors",
    "🎭 Consumer & Arts Sentiment",
    "🏛️ Arts Funding",
    "📁 All Data Files",
])

# =============================================================================
# IMPACT OVERVIEW TAB
# =============================================================================

with tab_overview:
    st.header("How External Data Affects Ticket Estimates")
    
    st.markdown("""
    The app uses external data sources to adjust ticket estimates based on Alberta's 
    economic conditions. Here's how the factors currently affect your estimates:
    """)
    
    # Calculate current impact
    impact = calculate_current_economic_impact()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment = impact["sentiment_factor"]
        delta_pct = (sentiment - 1.0) * 100
        st.metric(
            "Economic Sentiment Factor",
            f"×{sentiment:.3f}",
            delta=f"{delta_pct:+.1f}%",
            delta_color="normal" if delta_pct >= 0 else "inverse",
            help="Multiplier applied to all ticket estimates based on economic conditions"
        )
    
    with col2:
        oil_f = impact["oil_factor"]
        st.metric(
            "Oil Price Factor",
            f"×{oil_f:.3f}",
            delta=f"Weight: 40%",
            delta_color="off",
            help="Contribution from oil prices (higher = more spending power)"
        )
    
    with col3:
        unemp_f = impact["unemployment_factor"]
        st.metric(
            "Employment Factor",
            f"×{unemp_f:.3f}",
            delta=f"Weight: 60%",
            delta_color="off",
            help="Contribution from employment levels (lower unemployment = better)"
        )
    
    with col4:
        # Get consumer confidence if available
        nanos = get_nanos_consumer_confidence_summary()
        if "current_index" in nanos:
            ci = nanos["current_index"]
            avg = nanos.get("overall_avg", 54.92)
            st.metric(
                "Consumer Confidence",
                f"{ci:.1f}",
                delta=f"{ci - avg:+.1f} vs avg",
                help="Nanos Bloomberg Consumer Confidence Index (neutral = 50)"
            )
        else:
            st.metric(
                "Consumer Confidence",
                "N/A",
                help="Consumer confidence data not available"
            )
    
    st.divider()
    
    # Impact explanation
    st.subheader("📖 How This Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **The Economic Sentiment Factor** adjusts ticket estimates based on:
        
        1. **Oil Prices (40% weight)**: Alberta's economy is closely tied to oil.
           Higher prices generally mean more disposable income.
           - Baseline: $60 USD (WCS)
           - Current: ${:.2f} USD
           
        2. **Unemployment (60% weight)**: Lower unemployment means 
           more people employed and spending on entertainment.
           - Baseline: 6.0%
           - Current: {:.1f}%
        """.format(
            impact["oil_price"] or 0,
            impact["unemployment_rate"] or 0
        ))
    
    with col2:
        st.markdown("""
        **Example Impact on Ticket Estimates:**
        
        If a title's base estimate is **5,000 tickets**:
        
        - With sentiment factor of **×{:.3f}**
        - Adjusted estimate: **{:,.0f} tickets**
        - Difference: **{:+,.0f} tickets** ({:+.1f}%)
        
        The factor is applied after seasonality and remount 
        adjustments, as the final economic adjustment.
        """.format(
            impact["sentiment_factor"],
            5000 * impact["sentiment_factor"],
            5000 * (impact["sentiment_factor"] - 1),
            (impact["sentiment_factor"] - 1) * 100
        ))
    
    # Visual gauge
    st.subheader("📊 Economic Sentiment Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=impact["sentiment_factor"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Economic Sentiment Factor"},
        delta={"reference": 1.0, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0.85, 1.15], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "darkblue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0.85, 0.95], "color": "#ffcccc"},
                {"range": [0.95, 1.05], "color": "#ffffcc"},
                {"range": [1.05, 1.15], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 1.0,
            },
        },
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Legend**: Red zone (<0.95) = unfavorable economic conditions, 
    Yellow zone (0.95-1.05) = neutral, Green zone (>1.05) = favorable.
    A factor of 1.0 means no adjustment to estimates.
    """)


# =============================================================================
# ECONOMIC FACTORS TAB
# =============================================================================

with tab_economic:
    st.header("💰 Economic Indicators")
    
    # Load oil price data
    st.subheader("Oil Prices (WCS/WTI)")
    
    if DATA_LOADERS_AVAILABLE:
        oil_df = load_oil_prices()
        if not oil_df.empty:
            oil_df["date"] = pd.to_datetime(oil_df["date"], errors="coerce")
            oil_df = oil_df.dropna(subset=["date"])
            
            # Show recent data
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Chart
                fig = px.line(
                    oil_df, x="date", y="wcs_oil_price" if "wcs_oil_price" in oil_df.columns else oil_df.columns[1],
                    color="oil_series" if "oil_series" in oil_df.columns else None,
                    title="Historical Oil Prices"
                )
                fig.add_hline(y=60, line_dash="dash", line_color="red", 
                             annotation_text="Baseline ($60)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Recent Data**")
                recent_oil = oil_df.sort_values("date", ascending=False).head(10)
                st.dataframe(recent_oil, hide_index=True, height=300)
        else:
            st.info("Oil price data not available")
    else:
        st.info("Data loaders not available")
    
    st.divider()
    
    # Unemployment data
    st.subheader("Unemployment Rates")
    
    if DATA_LOADERS_AVAILABLE:
        unemp_df = load_unemployment_rates()
        if not unemp_df.empty:
            unemp_df["date"] = pd.to_datetime(unemp_df["date"], errors="coerce")
            unemp_df = unemp_df.dropna(subset=["date"])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.line(
                    unemp_df, x="date", y="unemployment_rate",
                    color="region" if "region" in unemp_df.columns else None,
                    title="Unemployment Rates by Region"
                )
                fig.add_hline(y=6.0, line_dash="dash", line_color="red",
                             annotation_text="Baseline (6.0%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Recent Data**")
                recent_unemp = unemp_df.sort_values("date", ascending=False).head(15)
                st.dataframe(recent_unemp, hide_index=True, height=300)
        else:
            st.info("Unemployment data not available")
    
    st.divider()
    
    # Commodity Price Index
    st.subheader("Commodity Price Index")
    data_dir = get_data_dir()
    cpi_path = data_dir / "economics/commodity_price_index.csv"
    
    if cpi_path.exists():
        cpi_df = pd.read_csv(cpi_path)
        cpi_df["date"] = pd.to_datetime(cpi_df["date"], errors="coerce")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot main indices
            cols_to_plot = ["A.BCPI", "A.ENER"]
            cols_available = [c for c in cols_to_plot if c in cpi_df.columns]
            
            if cols_available:
                fig = px.line(
                    cpi_df, x="date", y=cols_available,
                    title="Bank of Canada Commodity Price Index"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Index Components:**")
            st.markdown("""
            - **A.BCPI**: Total commodity index
            - **A.ENER**: Energy component
            - **A.MTLS**: Metals & minerals
            - **A.FOPR**: Forestry products
            - **A.AGRI**: Agriculture
            """)
    else:
        st.info("Commodity price data not available")


# =============================================================================
# CONSUMER & ARTS SENTIMENT TAB
# =============================================================================

with tab_sentiment:
    st.header("🎭 Consumer & Arts Sentiment")
    
    # Nanos Consumer Confidence
    st.subheader("Nanos Consumer Confidence Index")
    
    nanos = get_nanos_consumer_confidence_summary()
    
    if nanos:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if "current_index" in nanos:
                st.metric("Current Index", f"{nanos['current_index']:.1f}")
        with col2:
            if "prairies_index" in nanos:
                st.metric("Prairies Region", f"{nanos['prairies_index']:.1f}")
        with col3:
            if "overall_avg" in nanos:
                st.metric("Historical Average", f"{nanos['overall_avg']:.1f}")
        with col4:
            if "record_low" in nanos and "record_high" in nanos:
                st.metric("Range", f"{nanos['record_low']:.0f} - {nanos['record_high']:.0f}")
        
        st.markdown("""
        **About the BNCCI**: The Nanos Bloomberg Canadian Consumer Confidence Index 
        measures how Canadians feel about their financial future. A value of 50 is neutral;
        above 50 indicates optimism, below 50 indicates pessimism.
        
        **Prairies Impact**: The Prairies regional score is most relevant for Alberta Ballet's
        audience. Higher confidence typically correlates with higher discretionary spending.
        """)
    else:
        st.info("Nanos consumer confidence data not available")
    
    st.divider()
    
    # Arts Donors Survey
    st.subheader("Arts Giving Patterns (Nanos Survey)")
    
    data_dir = get_data_dir()
    arts_donors_path = data_dir / "audiences/nanos_arts_donors.csv"
    
    if arts_donors_path.exists():
        arts_df = pd.read_csv(arts_donors_path)
        
        # Get annual giving breakdown
        annual = arts_df[arts_df["section"] == "Annual giving breakdown"]
        
        if not annual.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Arts Share of Charitable Giving**")
                arts_share = annual[annual["subcategory"] == "Arts share"]
                if not arts_share.empty:
                    fig = px.bar(
                        arts_share, x="year_or_period", y="value",
                        title="Arts as % of Annual Charitable Giving"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Regional Arts Giving (2025)**")
                regional = arts_df[arts_df["section"] == "Annual giving by region"]
                if not regional.empty:
                    fig = px.bar(
                        regional, x="subcategory", y="value",
                        title="Arts Share by Region"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Arts donors survey data not available")


# =============================================================================
# ARTS FUNDING TAB
# =============================================================================

with tab_arts:
    st.header("🏛️ Arts Funding Overview")
    
    # Canada Council Summary
    funding = get_arts_funding_summary()
    
    if funding:
        st.subheader("Canada Council for the Arts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "total_funding" in funding:
                st.metric(
                    f"Total Support ({funding.get('funding_year', 'Latest')})",
                    f"${funding['total_funding']:,.0f}K"
                )
        with col2:
            if "core_grants" in funding:
                st.metric("Core Grants", f"${funding['core_grants']:,.0f}K")
        with col3:
            st.metric("Fiscal Year", funding.get("funding_year", "N/A"))
    
    st.divider()
    
    # Stats Can Culture data
    st.subheader("Statistics Canada Culture Indicators")
    
    data_dir = get_data_dir()
    
    # Check each Stats Can file
    stats_can_files = [
        ("stats_can_culture_sports_trade_provincial.csv", "Provincial Culture Trade"),
        ("stats_can_culture_sports_trade_domain.csv", "Culture by Domain"),
        ("stats_can_culture_sports_trade_gdp.csv", "Culture GDP Contribution"),
    ]
    
    for filename, title in stats_can_files:
        file_path = data_dir / "economics" / filename
        if file_path.exists():
            with st.expander(f"📊 {title}"):
                df = pd.read_csv(file_path)
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    
    st.divider()
    
    # Tourism Impact
    st.subheader("Tourism Impact (Travel Alberta)")
    
    tourism_path = data_dir / "economics/travel_alberta_tourism_impact.csv"
    if tourism_path.exists():
        tourism_df = pd.read_csv(tourism_path)
        
        # Show key metrics
        canada_overview = tourism_df[tourism_df["category"] == "Canada overview"]
        if not canada_overview.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Tourism Metrics**")
                st.dataframe(
                    tourism_df[["category", "metric_name", "metric_value", "unit"]].head(10),
                    hide_index=True
                )
            
            with col2:
                st.markdown("**Tourism-Arts Connection:**")
                st.markdown("""
                - Tourism drives cultural attendance
                - Visitor economy valued at $100B+ CAD (2024)
                - International tourism contributes to GDP
                - Higher tourism = more potential audience
                """)
    else:
        st.info("Travel Alberta tourism data not available")


# =============================================================================
# ALL DATA FILES TAB
# =============================================================================

with tab_files:
    st.header("📁 External Data Files Inventory")
    
    st.markdown("""
    This tab shows all external CSV files that can affect ticket estimates.
    Green checkmarks indicate files that are loaded and available.
    """)
    
    # Group files by category
    categories = {}
    for filename, info in EXTERNAL_DATA_FILES.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((filename, info))
    
    # Show files by category
    for category, files in sorted(categories.items()):
        st.subheader(f"{category}")
        
        for filename, info in files:
            df, error = load_external_file(info)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                if df is not None:
                    st.success(f"✅ **{filename}**")
                else:
                    st.error(f"❌ **{filename}**")
                st.caption(info["description"])
            
            with col2:
                if df is not None:
                    stats = get_file_stats(df)
                    st.caption(f"📊 {stats['rows']:,} rows × {stats['columns']} cols")
                    if "date_range" in stats:
                        st.caption(f"📅 {stats['date_range']}")
                else:
                    st.caption(f"⚠️ {error}" if error else "Not loaded")
            
            with col3:
                impact = info["impact_type"]
                badges = {
                    "oil_factor": "🛢️ Oil Factor",
                    "unemployment_factor": "👷 Employment",
                    "economic_sentiment": "📈 Economic",
                    "consumer_confidence": "🛒 Consumer",
                    "arts_sector": "🎭 Arts",
                    "tourism": "✈️ Tourism",
                }
                st.caption(badges.get(impact, impact))
            
            # Show preview in expander
            if df is not None:
                with st.expander(f"Preview {filename}"):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Columns: {', '.join(df.columns.tolist())}")
        
        st.divider()
    
    # Summary statistics
    st.subheader("📊 Data Summary")
    
    total_files = len(EXTERNAL_DATA_FILES)
    loaded_files = sum(1 for f, i in EXTERNAL_DATA_FILES.items() 
                       if load_external_file(i)[0] is not None)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Files Defined", total_files)
    with col2:
        st.metric("Files Available", loaded_files)
    with col3:
        st.metric("Files Missing", total_files - loaded_files)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("""
**External Data Impact Dashboard** — This page shows how external CSV data files 
affect ticket estimates in the Alberta Ballet Title Scoring App. The economic 
sentiment factor is the primary mechanism through which external data influences 
estimates, but other data sources provide context and inform strategic decisions.
""")
