from __future__ import annotations
from typing import List, Dict, Any, Optional
import math
import logging

import requests
import pandas as pd
import streamlit as st

from pytrends.request import TrendReq
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from ml.scoring import score_runs_for_planning

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Title Scoring Helper", layout="wide")

st.title("Title Scoring Helper")
st.caption(
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube, and Spotify "
    "for title demand scoring, then estimate ticket demand with uncertainty bands."
)

# -----------------------------------------------------------------------------
# External API clients
# -----------------------------------------------------------------------------

pytrends = TrendReq(hl="en-US", tz=0)

YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", None)
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID", None)
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET", None)

if YOUTUBE_API_KEY:
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
else:
    youtube = None

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    spotify_auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    spotify = spotipy.Spotify(client_credentials_manager=spotify_auth)
else:
    spotify = None

# -----------------------------------------------------------------------------
# Fetch helpers
# -----------------------------------------------------------------------------

def fetch_wikipedia_views(title: str) -> float:
    """
    Very rough Wikipedia signal: pageviews over recent days.
    Falls back to a small constant if anything fails.
    """
    try:
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia.org/all-access/user/"
            + requests.utils.quote(title.replace(" ", "_"))
            + "/daily/20230101/20231231"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return 1.0
        vals = [it.get("views", 0) for it in items]
        return float(sum(vals)) / max(len(vals), 1)
    except Exception as exc:
        logger.warning("Wikipedia fetch failed for " + title + ": " + str(exc))
        return 1.0


def fetch_google_trends_score(title: str) -> float:
    """
    Google Trends interest_over_time average, scaled 0–100 by Google.
    Returns a mean over recent period; falls back to small constant if empty.
    """
    try:
        pytrends.build_payload([title], timeframe="today 12-m")
        df = pytrends.interest_over_time()
        if df.empty or title not in df.columns:
            return 1.0
        vals = df[title].tolist()
        if not vals:
            return 1.0
        return float(sum(vals)) / max(len(vals), 1)
    except Exception as exc:
        logger.warning("Google Trends fetch failed for " + title + ": " + str(exc))
        return 1.0


def fetch_youtube_metric(title: str) -> float:
    """
    YouTube search-based rough metric: views of the top result.
    If YouTube is not configured or fails, return a small fallback.
    """
    if youtube is None:
        return 1.0
    try:
        search_response = (
            youtube.search()
            .list(
                q=title,
                part="id,snippet",
                maxResults=1,
                type="video",
                safeSearch="moderate",
            )
            .execute()
        )
        items = search_response.get("items", [])
        if not items:
            return 1.0
        vid_id = items[0]["id"]["videoId"]
        stats_resp = (
            youtube.videos()
            .list(id=vid_id, part="statistics")
            .execute()
        )
        vitems = stats_resp.get("items", [])
        if not vitems:
            return 1.0
        views = float(vitems[0]["statistics"].get("viewCount", 0))
        return max(views, 1.0)
    except Exception as exc:
        logger.warning("YouTube fetch failed for " + title + ": " + str(exc))
        return 1.0


def fetch_spotify_metric(title: str) -> float:
    """
    Spotify search-based rough metric: popularity of the top track.
    If Spotify is not configured or fails, return a small fallback.
    """
    if spotify is None:
        return 1.0
    try:
        res = spotify.search(q=title, type="track", limit=1)
        tracks = res.get("tracks", {}).get("items", [])
        if not tracks:
            return 1.0
        popularity = tracks[0].get("popularity", 0)
        return float(popularity)
    except Exception as exc:
        logger.warning("Spotify fetch failed for " + title + ": " + str(exc))
        return 1.0

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def normalize_0_100(values: List[float]) -> List[float]:
    """
    Rescale a list of values to 0–100. If all values are identical,
    return 50 for all to avoid divide-by-zero.
    """
    if not values:
        return []
    v = [0.0 if (val is None or math.isnan(val)) else float(val) for val in values]
    v_min = min(v)
    v_max = max(v)
    if v_max <= v_min:
        return [50.0 for _ in v]
    return [100.0 * (x - v_min) / (v_max - v_min) for x in v]

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.subheader("Step 1 – Enter Titles")

titles_raw = st.text_area(
    "Paste one title per line:",
    height=150,
    placeholder="The Nutcracker\nSwan Lake\nCinderella",
)

col_a, col_b = st.columns(2)
with col_a:
    fetch_button = st.button("Fetch & Normalize Scores", type="primary")
with col_b:
    confidence_level = st.selectbox(
        "Forecast interval confidence",
        options=[0.8, 0.9, 0.95],
        format_func=lambda x: str(int(x * 100)) + "%",
        index=0,
    )

titles: List[str] = [t.strip() for t in titles_raw.splitlines() if t.strip()]

if fetch_button and titles:
    with st.spinner("Fetching external signals and building title features…"):
        rows: List[Dict[str, Any]] = []
        for title in titles:
            wiki_val = fetch_wikipedia_views(title)
            trends_val = fetch_google_trends_score(title)
            yt_val = fetch_youtube_metric(title)
            sp_val = fetch_spotify_metric(title)

            rows.append(
                {
                    "title": title,
                    "wiki_raw": wiki_val,
                    "trends_raw": trends_val,
                    "youtube_raw": yt_val,
                    "spotify_raw": sp_val,
                }
            )

        df_raw = pd.DataFrame(rows)

        wiki_norm = normalize_0_100(df_raw["wiki_raw"].tolist())
        trends_norm = normalize_0_100(df_raw["trends_raw"].tolist())
        youtube_norm = normalize_0_100(df_raw["youtube_raw"].tolist())
        spotify_norm = normalize_0_100(df_raw["spotify_raw"].tolist())

        df_raw["wiki"] = wiki_norm
        df_raw["trends"] = trends_norm
        df_raw["youtube"] = youtube_norm
        df_raw["spotify"] = spotify_norm

    st.subheader("Step 2 – Normalized Signals (0–100)")
    st.dataframe(
        df_raw[["title", "wiki", "trends", "youtube", "spotify"]],
        use_container_width=True,
    )

    st.subheader("Step 3 – Ticket Forecasts")

    default_genre = st.selectbox(
        "Default genre for all titles (can be refined later in season planning CSV)",
        options=["classical", "contemporary", "family", "mixed"],
        index=0,
    )
    default_season = st.selectbox(
        "Season label (used as a feature)",
        options=["2024-25", "2025-26", "2026-27"],
        index=1,
    )

    feature_rows: List[Dict[str, Any]] = []
    for _, row in df_raw.iterrows():
        feature_rows.append(
            {
                "title": row["title"],
                "wiki": row["wiki"],
                "trends": row["trends"],
                "youtube": row["youtube"],
                "spotify": row["spotify"],
                "genre": default_genre,
                "season": default_season,
            }
        )

    df_features = pd.DataFrame(feature_rows)

    # Drop non-feature columns when scoring; keep title for joining back
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
    except Exception as exc:
        st.error("Error while scoring titles: " + str(exc))
        st.stop()

    # Attach titles back for display
    df_scored["title"] = df_features["title"].values

    # Reorder columns: title first, then forecast + intervals
    forecast_cols = [
        c for c in df_scored.columns
        if "forecast" in c or "lower" in c or "upper" in c
    ]
    other_cols = [c for c in df_scored.columns if c not in forecast_cols + ["title"]]
    display_cols = ["title"] + forecast_cols + other_cols

    st.dataframe(df_scored[display_cols], use_container_width=True)

    csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast CSV",
        data=csv_bytes,
        file_name="title_forecasts.csv",
        mime="text/csv",
    )
