# - Learns YYC/YEG splits from history.csv (or uploaded CSV)
# - Single ticket estimation only
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards
# - Economic sentiment factor integration for market-aware ticket estimation
# - Title scoring helper functionality integrated (Wikipedia, Google Trends,
#   YouTube, Spotify) into main app

from __future__ import annotations

import io
import math
import re
import sys
import time
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple, List, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from textwrap import dedent

from pytrends.request import TrendReq
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Economic / weather / live analytics
try:
    from data.loader import (
        get_economic_sentiment_factor,
        load_oil_prices,
        load_unemployment_rates,
        load_weather_calgary,
        load_weather_edmonton,
        get_weather_impact_factor,
        get_monthly_weather_summary,
        load_live_analytics_raw,
        get_live_analytics_category_factors,
        get_category_engagement_factor,
    )

    ECONOMIC_DATA_AVAILABLE = True
    WEATHER_DATA_AVAILABLE = True
    LIVE_ANALYTICS_AVAILABLE = True
except ImportError:
    ECONOMIC_DATA_AVAILABLE = False
    WEATHER_DATA_AVAILABLE = False
    LIVE_ANALYTICS_AVAILABLE = False

    def get_economic_sentiment_factor(*args, **kwargs):
        return 1.0

    def get_weather_impact_factor(*args, **kwargs):
        return 1.0

    def get_monthly_weather_summary(*args, **kwargs):
        return {}

    def get_category_engagement_factor(*args, **kwargs):
        return 1.0

    def get_live_analytics_category_factors(*args, **kwargs):
        return {}

    def load_weather_calgary(*args, **kwargs):
        return pd.DataFrame()

    def load_weather_edmonton(*args, **kwargs):
        return pd.DataFrame()

    def load_live_analytics_raw(*args, **kwargs):
        return pd.DataFrame()

# ML scoring (relaxed schema + intervals)
from ml.scoring import score_runs_for_planning

try:
    import yaml
except ImportError:
    yaml = None

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

SEGMENT_MULT: Dict[str, float] = {}
REGION_MULT: Dict[str, float] = {}
DEFAULT_BASE_CITY_SPLIT: Tuple[float, float] = (0.5, 0.5)
_CITY_CLIP_RANGE: Tuple[float, float] = (0.1, 0.9)
POSTCOVID_FACTOR: float = 1.0
TICKET_BLEND_WEIGHT: float = 0.7
DEFAULT_MARKETING_SPT_CITY: Dict[str, float] = {}

ML_CONFIG: Dict[str, Any] = {}
KNN_CONFIG: Dict[str, Any] = {}
CALIBRATION_CONFIG: Dict[str, Any] = {}


def load_config(path: str = "config.yaml") -> None:
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global DEFAULT_MARKETING_SPT_CITY
    global ML_CONFIG, KNN_CONFIG, CALIBRATION_CONFIG

    if yaml is None:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return

    if "segment_mult" in cfg:
        SEGMENT_MULT = cfg["segment_mult"]

    if "region_mult" in cfg:
        REGION_MULT = cfg["region_mult"]

    city_cfg = cfg.get("city_splits", {})
    DEFAULT_BASE_CITY_SPLIT = city_cfg.get("default_base_city_split", DEFAULT_BASE_CITY_SPLIT)
    _CITY_CLIP_RANGE = tuple(city_cfg.get("city_clip_range", _CITY_CLIP_RANGE))

    demand_cfg = cfg.get("demand", {})
    POSTCOVID_FACTOR = demand_cfg.get("postcovid_factor", POSTCOVID_FACTOR)
    TICKET_BLEND_WEIGHT = demand_cfg.get("ticket_blend_weight", TICKET_BLEND_WEIGHT)

def main():
    load_config("config.yaml")

    st.set_page_config(page_title="Alberta Ballet: Title Scoring and Forecasts", layout="wide")

    st.title("Alberta Ballet: Title Scoring and Ticket Forecasts")
    st.caption(
        "Plan productions by estimating single-ticket demand and exploring driver signals "
        "like title popularity, economics, weather, and marketing."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a view",
            ["Overview", "Model training (legacy)", "Diagnostics"],
            index=0,
        )

    if page == "Overview":
        overview_page()
    elif False:
        st.subheader("Overview")
        st.write(
            "This is a placeholder overview page. "
            "Once the full logic is wired back in, you'll see title scoring, "
            "ticket forecasts, and uncertainty bands here."
        )

        uploaded = st.file_uploader(
            "Upload a productions CSV (optional)",
            type=["csv"],
            help="Upload the same structure as productions_history.csv",
        )
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

    elif page == "Model training (legacy)":
        st.subheader("Legacy baseline training")
        st.warning(
            "This page runs the old baseline training pipeline. "
            "It is deprecated in favour of the safe modelling dataset pipeline."
        )
        if st.button("Run legacy baseline training (slow)"):
            with st.spinner("Training baseline model (this can take several minutes)..."):
                try:
                    from ml.training import train_baseline_model
                    result = train_baseline_model()
                    st.success("Baseline training finished.")
                    st.json(result)
                except Exception as exc:
                    st.error("Training failed: " + str(exc))

    elif page == "Diagnostics":
        st.subheader("Diagnostics")
        st.write("Configuration loaded from config.yaml:")
        st.json(
            {
                "SEGMENT_MULT": SEGMENT_MULT,
                "REGION_MULT": REGION_MULT,
                "DEFAULT_BASE_CITY_SPLIT": DEFAULT_BASE_CITY_SPLIT,
                "CITY_CLIP_RANGE": _CITY_CLIP_RANGE,
                "POSTCOVID_FACTOR": POSTCOVID_FACTOR,
                "TICKET_BLEND_WEIGHT": TICKET_BLEND_WEIGHT,
            }
        )

# --- Julius patch: helper loader functions for internal CSVs ---
import pandas as _pd
import os as _os

HISTORY_DEFAULT_PATH = "data/productions/history_city_sales.csv"
TITLE_SCORES_DEFAULT_PATH = "data/productions/title_scores_history.csv"

@st.cache_data(show_spinner=False)
def load_title_scores_history():
    if os.path.exists(TITLE_SCORES_DEFAULT_PATH):
        try:
            return pd.read_csv(TITLE_SCORES_DEFAULT_PATH)
        except Exception as e:
            st.warning("Could not read " + TITLE_SCORES_DEFAULT_PATH + ": " + str(e))
            return None
    return None

PRODUCTIONS_DEFAULT_PATH = "data/productions/productions_history.csv"

@st.cache_data(show_spinner=False)
def load_default_history():
    if _os.path.exists(HISTORY_DEFAULT_PATH):
        try:
            return _pd.read_csv(HISTORY_DEFAULT_PATH)
        except Exception as e:
            st.warning("Could not read " + HISTORY_DEFAULT_PATH + ": " + str(e))
            return None
    return None

@st.cache_data(show_spinner=False)
def load_default_productions():
    if _os.path.exists(PRODUCTIONS_DEFAULT_PATH):
        try:
            return _pd.read_csv(PRODUCTIONS_DEFAULT_PATH)
        except Exception as e:
            st.warning("Could not read " + PRODUCTIONS_DEFAULT_PATH + ": " + str(e))
            return None
    return None


def overview_page():
    st.subheader("Overview")
    st.write(
        "History and productions are loaded automatically from the repo. "
        "You can optionally upload CSVs to override them for what-if analysis."
    )

    default_hist = load_default_history()
    default_prods = load_default_productions()

    col1, col2 = st.columns(2)
    with col1:
        hist_file = st.file_uploader(
            "Upload history CSV (optional)",
            type=["csv"],
            key="hist_upload",
            help="If provided, overrides data/productions/history_city_sales.csv",
        )
    with col2:
        prods_file = st.file_uploader(
            "Upload productions CSV (optional)",
            type=["csv"],
            key="prods_upload",
            help="If provided, overrides data/productions/productions_history.csv",
        )

    if hist_file is not None:
        history_df = _pd.read_csv(hist_file)
    else:
        history_df = default_hist

    if prods_file is not None:
        prods_df = _pd.read_csv(prods_file)
    else:
        prods_df = default_prods

    if history_df is None and prods_df is None:
        st.warning(
            "No internal CSVs found yet. Add them under data/productions/ or upload here."
        )
        return

    if history_df is not None:
        st.markdown("**History sample**")
        st.dataframe(history_df.head())

    if prods_df is not None:
        st.markdown("**Productions sample**")
        st.dataframe(prods_df.head())

    st.info(
        "Downstream pages can now reuse these dataframes from st.session_state "
        "instead of forcing new uploads."
    )
    st.session_state["history_df"] = history_df
    st.session_state["productions_df"] = prods_df



def main():
    # Load configuration and set page meta
    load_config("config.yaml")

    st.set_page_config(
        page_title="Alberta Ballet: Title Scoring and Ticket Forecasts",
        layout="wide"
    )

    st.title("Alberta Ballet: Title Scoring and Ticket Forecasts")
    st.caption(
        "Estimate title demand and single-ticket forecasts using internal history, "
        "external signals (Wikipedia, Google, YouTube, Spotify), and market context."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a view",
            ["Overview", "Title Scoring", "Forecasts & Planning", "Diagnostics"],
            index=0,
        )

    if page == "Overview":
        overview_page()
    elif False:
        _page_overview()
    elif page == "Title Scoring":
        _page_title_scoring()
    elif page == "Forecasts & Planning":
        _page_forecasts_planning()
    else:
        _page_diagnostics()


@st.cache_data(show_spinner=False)
def load_baselines_cached():
    """Load pre-baked title scores from baselines.csv (instruction_files)."""
    possible_paths = [
        "data/productions/instruction_files/baselines.csv",
        "data/baselines.csv",
        "baselines.csv",
    ]
    for p in possible_paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                return df
        except Exception:
            continue
    return None


# -------------------------------------------------------------------------
# PAGE: Overview
# -------------------------------------------------------------------------
def _page_overview():
    st.subheader("Overview")
    # Optional: pre-baked title demand scores history
    title_scores_hist = load_title_scores_history()
    if title_scores_hist is not None and not title_scores_hist.empty:
        st.markdown("**Title scores history (sample)**")
        st.dataframe(title_scores_hist.head())
        st.session_state["title_scores_history_df"] = title_scores_hist


    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "Use the sidebar to switch between:\n"
            "- **Title Scoring** to see multi-source popularity signals for candidate titles.\n"
            "- **Forecasts & Planning** to estimate single-ticket demand for upcoming productions.\n"
        )

        uploaded = st.file_uploader(
            "Optionally upload a history CSV (productions with ticket sales)",
            type=["csv"],
            help="If provided, it will be used on the Forecasts & Planning page."
        )
        if uploaded is not None:
            try:
                hist_df = pd.read_csv(uploaded)
                st.session_state["history_df"] = hist_df
                st.success("History data loaded into session_state.")
                st.dataframe(hist_df.head())
            except Exception as exc:
                st.error("Could not read uploaded CSV: " + str(exc))

    with col2:
        st.markdown("**Quick Links**")
        st.markdown("- Config file: `config.yaml`")
        st.markdown("- Legacy baseline: see `pages/4_Model_Training.py` (deprecated)")


# -------------------------------------------------------------------------
# PAGE: Title Scoring (integrated helper)
# -------------------------------------------------------------------------
def _page_title_scoring():
    st.subheader("Title Scoring Helper")

    st.markdown(
        "Fetch and normalize 00 scores from Wikipedia, Google Trends, YouTube, "
        "and Spotify for title demand scoring."
    )

    titles_input = st.text_area(
        "Enter one title per line",
        height=150,
        placeholder="The Nutcracker\nRomeo and Juliet\nSwan Lake",
    )

    if not titles_input.strip():
        st.info("Enter at least one title above to see scores.")
        return

    titles = [t.strip() for t in titles_input.splitlines() if t.strip()]
    if not titles:
        st.info("No valid titles found after cleaning the input.")
        return

    # Prefer existing baseline scores from baselines.csv when available,
    # then fall back to live API calls for unknown titles.
    baselines_df = load_baselines_cached()
    baseline_lut = None
    if baselines_df is not None and not baselines_df.empty and "title" in baselines_df.columns:
        baselines_df = baselines_df.copy()
        baselines_df["_title_key"] = baselines_df["title"].astype(str).str.strip().str.lower()
        baseline_lut = baselines_df.set_index("_title_key").to_dict(orient="index")

    rows = []
    live_titles = []

    for t in titles:
        key = t.strip().lower()
        if baseline_lut is not None and key in baseline_lut:
            b = baseline_lut[key]
            row = {"title": t}

            # Raw metrics from baselines (these column names match your baselines.csv)
            raw_map = {
                "wiki": "wiki_raw",
                "trends": "trends_raw",
                "youtube": "youtube_raw",
                "spotify": "spotify_raw",
            }
            for src_col, dst_col in raw_map.items():
                if src_col in b:
                    row[dst_col] = b[src_col]

            # If baselines already store normalized 00 scores, map them;
            # otherwise we treat the stored values as scores directly.
            score_map = {
                "wiki": "wiki_score_0_100",
                "trends": "trends_score_0_100",
                "youtube": "youtube_score_0_100",
                "spotify": "spotify_score_0_100",
            }
            for src_col, dst_col in score_map.items():
                if src_col in b and (dst_col not in row or pd.isna(row.get(dst_col))):
                    try:
                        row[dst_col] = float(b[src_col])
                    except Exception:
                        pass

            # Aggregate if not explicitly provided
            if "aggregate_score_0_100" in b:
                try:
                    row["aggregate_score_0_100"] = float(b["aggregate_score_0_100"])
                except Exception:
                    row["aggregate_score_0_100"] = np.nan
            else:
                valid_scores = [
                    s for s in [
                        row.get("wiki_score_0_100"),
                        row.get("trends_score_0_100"),
                        row.get("youtube_score_0_100"),
                        row.get("spotify_score_0_100"),
                    ]
                    if s is not None and not pd.isna(s)
                ]
                if valid_scores:
                    row["aggregate_score_0_100"] = float(sum(valid_scores)) / len(valid_scores)
                else:
                    row["aggregate_score_0_100"] = np.nan

            rows.append(row)
        else:
            live_titles.append(t)

    # For titles not in baselines, use live API calls
    if live_titles:
        with st.spinner("Fetching signals from Wikipedia / Google / YouTube / Spotify for new titles..."):
            wiki_vals = []
            trends_vals = []
            yt_vals = []
            sp_vals = []

            for t in live_titles:
                wiki_val = fetch_wikipedia_views(t)
                trend_val = fetch_google_trends_score(t)
                yt_val = fetch_youtube_views(t)
                sp_val = fetch_spotify_metric(t)

                wiki_vals.append(wiki_val)
                trends_vals.append(trend_val)
                yt_vals.append(yt_val)
                sp_vals.append(sp_val)

            wiki_norm = normalize_0_100(wiki_vals)
            trends_norm = normalize_0_100(trends_vals)
            yt_norm = normalize_0_100(yt_vals)
            sp_norm = normalize_0_100(sp_vals)

            for i, t in enumerate(live_titles):
                row = {
                    "title": t,
                    "wiki_raw": wiki_vals[i],
                    "trends_raw": trends_vals[i],
                    "youtube_raw": yt_vals[i],
                    "spotify_raw": sp_vals[i],
                    "wiki_score_0_100": wiki_norm[i],
                    "trends_score_0_100": trends_norm[i],
                    "youtube_score_0_100": yt_norm[i],
                    "spotify_score_0_100": sp_norm[i],
                }
                valid_scores = [
                    s for s in [
                        wiki_norm[i],
                        trends_norm[i],
                        yt_norm[i],
                        sp_norm[i],
                    ]
                    if not math.isnan(s)
                ]
                if valid_scores:
                    row["aggregate_score_0_100"] = float(sum(valid_scores)) / len(valid_scores)
                else:
                    row["aggregate_score_0_100"] = np.nan
                rows.append(row)

    df_scores = pd.DataFrame(rows)

    st.markdown("**Title demand scores (00)**")
    st.dataframe(df_scores[[
        "title",
        "aggregate_score_0_100",
        "wiki_score_0_100",
        "trends_score_0_100",
        "youtube_score_0_100",
        "spotify_score_0_100",
    ]])

    csv_bytes = df_scores.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scores as CSV",
        data=csv_bytes,
        file_name="title_scores_export.csv",
        mime="text/csv",
    )

# -------------------------------------------------------------------------
# PAGE: Forecasts & Planning
# -------------------------------------------------------------------------
def _page_forecasts_planning():
    st.subheader("Forecasts & Planning")

    st.markdown(
        "This page will use the trained model (safe modelling pipeline) to estimate "
        "single-ticket demand for planned productions, optionally informed by title scores."
    )

    history_df = st.session_state.get("history_df")
    if history_df is None:
        st.info(
            "No history data in session. Upload a history CSV on the Overview page, "
            "or use the existing internal data pipeline."
        )

    # Here we just show placeholders; wiring to `score_runs_for_planning` depends
    # on your ml/scoring module and feature expectations.
    try:
        from ml.scoring import score_runs_for_planning  # type: ignore
        scoring_available = True
    except Exception:
        scoring_available = False

    if not scoring_available:
        st.warning(
            "The safe model scoring function `score_runs_for_planning` is not available "
            "or failed to import. Check ml/scoring.py."
        )
        return

    st.markdown("**Upcoming productions input**")

    upcoming_csv = st.file_uploader(
        "Upload a CSV of upcoming productions (with title, dates, venue, etc.)",
        type=["csv"],
        key="upcoming_runs_uploader",
    )

    if upcoming_csv is None:
        st.info("Upload a CSV of upcoming productions to run the planning model.")
        return

    try:
        upcoming_df = pd.read_csv(upcoming_csv)
    except Exception as exc:
        st.error("Could not read upcoming productions CSV: " + str(exc))
        return

    st.markdown("Preview of upcoming productions data:")
    st.dataframe(upcoming_df.head())

    # Run safe scoring
    if st.button("Run ticket demand model"):
        with st.spinner("Scoring upcoming productions with safe model..."):
            try:
                result_df, meta = score_runs_for_planning(
                    upcoming_df,
                    history_df=history_df,
                )
            except Exception as exc:
                st.error("Error while scoring with safe model: " + str(exc))
                return

        st.markdown("**Model results (head)**")
        st.dataframe(result_df.head())

        if "city" in result_df.columns and "predicted_single_tickets" in result_df.columns:
            st.markdown("**Predicted single tickets by production and city**")
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            pivot = (
                result_df
                .groupby(["run_id", "city"])["predicted_single_tickets"]
                .sum()
                .reset_index()
            )
            for city in sorted(pivot["city"].unique()):
                sub = pivot[pivot["city"] == city]
                ax2.bar(sub["run_id"].astype(str), sub["predicted_single_tickets"], label=city)
            ax2.set_xlabel("Run ID")
            ax2.set_ylabel("Predicted single tickets")
            ax2.legend()
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig2)

        st.session_state["planning_results_df"] = result_df


# -------------------------------------------------------------------------
# PAGE: Diagnostics
# -------------------------------------------------------------------------
def _page_diagnostics():
    st.subheader("Diagnostics")

    st.markdown("**Global config values**")
    st.json(
        {
            "SEGMENT_MULT": SEGMENT_MULT,
            "REGION_MULT": REGION_MULT,
            "DEFAULT_BASE_CITY_SPLIT": DEFAULT_BASE_CITY_SPLIT,
            "CITY_CLIP_RANGE": _CITY_CLIP_RANGE,
            "POSTCOVID_FACTOR": POSTCOVID_FACTOR,
            "TICKET_BLEND_WEIGHT": TICKET_BLEND_WEIGHT,
        }
    )


if __name__ == "__main__":
    main()


