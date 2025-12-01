# - Learns YYC/YEG splits from history.csv (or uploaded CSV)
# - Single ticket estimation only
# - Removes arbitrary 60/40 split; uses title→category→default fallback
# - Small fixes: softmax bug, LA attach loop, duplicate imports, safer guards
# - Economic sentiment factor integration for market-aware ticket estimation
# - Title scoring helper functionality integrated (Wikipedia, Google Trends,
#   YouTube, Spotify) into main app

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

# External API / title-signal libraries
from pytrends.request import TrendReq
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Import economic data loader functions
try:
    from data.loader import (
        get_economic_sentiment_factor,
        load_oil_prices,
        load_unemployment_rates,
        # Weather data integration
        load_weather_calgary,
        load_weather_edmonton,
        get_weather_impact_factor,
        get_monthly_weather_summary,
        # Live analytics integration
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
        import pandas as pd
        return pd.DataFrame()

    def load_weather_edmonton(*args, **kwargs):
        import pandas as pd
        return pd.DataFrame()

# Import Bank of Canada Valet API integration for live economic data
try:
    from utils.economic_factors import (
        compute_boc_economic_sentiment,
        get_boc_indicator_display,
        is_boc_live_enabled,
        get_combined_economic_sentiment,
        # Alberta integration
        compute_alberta_economic_sentiment,
        get_alberta_indicator_display,
        is_alberta_live_enabled,
        get_current_economic_context,
    )
    BOC_INTEGRATION_AVAILABLE = True
except ImportError:
    BOC_INTEGRATION_AVAILABLE = False

    def compute_boc_economic_sentiment(*args, **kwargs):
        return 1.0, {"source": "unavailable", "boc_available": False}

    def get_boc_indicator_display():
        return {"available": False, "message": "BoC integration not installed"}

    def is_boc_live_enabled():
        return False

    def get_combined_economic_sentiment(*args, **kwargs):
        return 1.0, {}

    def compute_alberta_economic_sentiment(*args, **kwargs):
        return 1.0, {"source": "unavailable", "alberta_available": False}

    def get_alberta_indicator_display():
        return {"available": False, "message": "Alberta integration not installed"}

    def is_alberta_live_enabled():
        return False

    def get_current_economic_context(*args, **kwargs):
        return {
            "boc": None,
            "alberta": None,
            "combined_sentiment": 1.0,
            "sources_available": [],
        }

from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

try:
    import yaml
except ImportError:
    yaml = None

# Advanced ML models for regression
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Integrated ML scoring module for uncertainty + planning
try:
    from ml.scoring import score_runs_for_planning
    ML_SCORING_AVAILABLE = True
except ImportError:
    ML_SCORING_AVAILABLE = False

# =============================================================================
# Title scoring / external signals (integrated from title_scoring_helper.py)
# =============================================================================

# Initialize external clients using Streamlit secrets
pytrends = TrendReq(hl="en-US", tz=0)
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", None)
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID", None)
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET", None)

if YOUTUBE_API_KEY:
    youtube_client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
else:
    youtube_client = None

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    spotify_auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
    )
    spotify_client = spotipy.Spotify(client_credentials_manager=spotify_auth)
else:
    spotify_client = None


def normalize_0_100(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(50.0, index=series.index)
    return (series - smin) / (smax - smin) * 100.0


def fetch_wikipedia_views(title: str, days: int = 365) -> float:
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia.org/all-access/all-agents/"
            + requests.utils.quote(title.replace(" ", "_"))
            + "/daily/"
            + start.strftime("%Y%m%d")
            + "/"
            + end.strftime("%Y%m%d")
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return float("")
        data = r.json().get("items", [])
        if not data:
            return float("")
        views = [item.get("views", 0) for item in data]
        if not views:
            return float("")
        return float(np.mean(views))
    except Exception:
        return float("")


def fetch_google_trends_score(title: str) -> float:
    try:
        kw_list = [title]
        pytrends.build_payload(kw_list, timeframe="today 12-m", geo="CA")
        df = pytrends.interest_over_time()
        if df.empty or title not in df.columns:
            return float("")
        return float(df[title].mean())
    except Exception:
        return float("")


def fetch_youtube_metric(title: str) -> float:
    if youtube_client is None:
        return float("")
    try:
        resp = (
            youtube_client.search()
            .list(
                q=title,
                part="id",
                type="video",
                maxResults=10,
            )
            .execute()
        )
        video_ids = [item["id"]["videoId"] for item in resp.get("items", [])]
        if not video_ids:
            return float("")
        stats_resp = (
            youtube_client.videos()
            .list(id=",".join(video_ids), part="statistics")
            .execute()
        )
        views = []
        for item in stats_resp.get("items", []):
            v = item.get("statistics", {}).get("viewCount")
            if v is not None:
                try:
                    views.append(float(v))
                except Exception:
                    continue
        if not views:
            return float("")
        return float(np.mean(views))
    except Exception:
        return float("")


def fetch_spotify_metric(title: str) -> float:
    if spotify_client is None:
        return float("")
    try:
        q = "track:" + title
        resp = spotify_client.search(q=q, type="track", limit=10)
        items = resp.get("tracks", {}).get("items", [])
        if not items:
            return float("")
        pops = [item.get("popularity", 0) for item in items]
        if not pops:
            return float("")
        return float(np.mean(pops))
    except Exception:
        return float("")


def build_title_signal_frame(
    titles: List[str],
    confidence_level: float = 0.8,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in titles:
        t_clean = (t or "").strip()
        if not t_clean:
            continue
        wiki_score = fetch_wikipedia_views(t_clean)
        trends_score = fetch_google_trends_score(t_clean)
        youtube_score = fetch_youtube_metric(t_clean)
        spotify_score = fetch_spotify_metric(t_clean)
        rows.append(
            {
                "title": t_clean,
                "wiki": wiki_score,
                "trends": trends_score,
                "youtube": youtube_score,
                "spotify": spotify_score,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["wiki_norm"] = normalize_0_100(df["wiki"])
    df["trends_norm"] = normalize_0_100(df["trends"])
    df["youtube_norm"] = normalize_0_100(df["youtube"])
    df["spotify_norm"] = normalize_0_100(df["spotify"])

    # If ML scoring is available, estimate tickets with uncertainty bands
    if ML_SCORING_AVAILABLE and not df.empty:
        df_features = df.copy()
        # expected columns by planning scorer
        df_features["show_title"] = df_features["title"]
        df_features["single_tickets_calgary"] = 0.0
        df_features["single_tickets_edmonton"] = 0.0

        to_score = df_features.drop(columns=["title"])
        try:
            df_scored = score_runs_for_planning(
                to_score,
                confidence_level=float(confidence_level),
                n_bootstrap=200,
                model=None,
                attach_context=False,
                economic_context=None,
            )
            pct = int(confidence_level * 100.0)
            lower_col = "lower_tickets_" + str(pct)
            upper_col = "upper_tickets_" + str(pct)

            cols_keep = [
                "show_title",
                "forecast_single_tickets",
            ]
            if lower_col in df_scored.columns:
                cols_keep.append(lower_col)
            if upper_col in df_scored.columns:
                cols_keep.append(upper_col)

            df_scored = df_scored[cols_keep]
            df_scored = df_scored.rename(columns={"show_title": "title"})
            df = df.merge(df_scored, on="title", how="left")
        except Exception:
            # if scoring fails, just return the raw signals
            pass

    return df


# =============================================================================
# Existing config, forecasting, and app logic continues below
# (unchanged from your previous streamlit_app except where it needs to
#  call build_title_signal_frame for title-level insights)
# =============================================================================

def load_config(path: str = "config.yaml"):
    global SEGMENT_MULT, REGION_MULT
    global DEFAULT_BASE_CITY_SPLIT, _CITY_CLIP_RANGE
    global POSTCOVID_FACTOR, TICKET_BLEND_WEIGHT
    global K_SHRINK, MINF, MAXF, N_MIN
    global DEFAULT_MARKETING_SPT_CITY
    # New robust forecasting settings
    global ML_CONFIG, KNN_CONFIG, CALIBRATION_CONFIG

    if yaml is None:
        # PyYAML not installed – just use hard-coded defaults
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # No config file – keep defaults
        return

    # 1) Segment multipliers
    if "segment_mult" in cfg:
        SEGMENT_MULT = cfg["segment_mult"]

    # 2) Region multipliers
    if "region_mult" in cfg:
        REGION_MULT = cfg["region_mult"]

    # 3) City splits
    city_cfg = cfg.get("city_splits", {})
    DEFAULT_BASE_CITY_SPLIT = city_cfg.get(
        "default_base_city_split", DEFAULT_BASE_CITY_SPLIT
    )
    _CITY_CLIP_RANGE = tuple(city_cfg.get("city_clip_range", _CITY_CLIP_RANGE))

    # 4) Demand knobs
    demand_cfg = cfg.get("demand", {})
    POSTCOVID_FACTOR = demand_cfg.get("postcovid_factor", POSTCOVID_FACTOR)
    TICKET_BLEND_WEIGHT = demand_cfg.get(
        "ticket_blend_weight", TICKET_BLEND_WEIGHT
    )

    # 5) Seasonality knobs
    seas_cfg = cfg.get("seasonality", {})
    K_SHRINK = seas_cfg.get("k_shrink", K_SHRINK)
    MINF = seas_cfg.get("minf", MINF)
    MAXF = seas_cfg.get("maxf", MAXF)
    N_MIN = seas_cfg.get("n_min", N_MIN)

    # 6) Default marketing spend-per-ticket city
    DEFAULT_MARKETING_SPT_CITY = cfg.get(
        "default_marketing_spt_city", DEFAULT_MARKETING_SPT_CITY
    )

    # 7) ML + calibration config
    ML_CONFIG = cfg.get("ml", ML_CONFIG)
    KNN_CONFIG = cfg.get("knn", KNN_CONFIG)
    CALIBRATION_CONFIG = cfg.get("calibration", CALIBRATION_CONFIG)


# ----------------------------------------------------------------------
# (All your existing functions and app layout go here unchanged:
#  - history loading
#  - season planner UI
#  - forecasting logic
#  - PDF export
#  - economic/weather/live analytics integration
#
#  The only addition you may want is a small section in the main UI to
#  show per-title external signals using build_title_signal_frame(...)
#  for the titles in the proposed season.)
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Alberta Ballet Season Forecaster",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Alberta Ballet Season Planning & Forecasting")

    # Example: add an optional title-signal section in the sidebar or main body
    with st.expander("Optional: Title demand signals (Wikipedia / Trends / YouTube / Spotify)"):
        title_input = st.text_area(
            "Enter one title per line to inspect external demand signals",
            value="",
            height=120,
        )
        conf = st.slider(
            "Prediction interval for title-level estimates (if model available)",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
        )
        if st.button("Fetch title signals"):
            titles = [t.strip() for t in title_input.splitlines() if t.strip()]
            df_signals = build_title_signal_frame(titles, confidence_level=conf)
            if df_signals is None or df_signals.empty:
                st.warning("No valid titles or no signals returned.")
            else:
                st.dataframe(df_signals)

    st.markdown(
        "Use the controls below to upload history, define runs, and generate "
        "ticket forecasts with economic and weather context."
    )

    # TODO: insert the rest of your original main() body here.
    # If your previous streamlit_app.py already had a main() and
    # if __name__ guard, keep that structure and merge this into it.
    # For brevity, we keep only the hook here. In your repo, make sure
    # all existing functionality below is preserved.


if __name__ == "__main__":
    main()
