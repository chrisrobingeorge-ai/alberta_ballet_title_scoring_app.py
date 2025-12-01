from __future__ import annotations
from typing import List, Dict, Any
import math

import requests
import pandas as pd
import streamlit as st

from pytrends.request import TrendReq
pytrends = TrendReq(hl="en-US", tz=0)
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from ml.scoring import score_runs_for_planning

st.set_page_config(page_title="Title Scoring Helper", layout="wide")

st.title("Title Scoring Helper")
st.caption(
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube, and Spotify "
    "for title demand scoring, then estimate ticket demand with uncertainty bands."
)

# -----------------------------------------------------------------------------
# CONFIG / KEYS
# -----------------------------------------------------------------------------

YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", None)
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID", None)
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET", None)

if YOUTUBE_API_KEY:
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
else:
    youtube = None

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    spotify_auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=spotify_auth)
else:
    sp = None


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def fetch_wikipedia_views(title: str) -> float:
    """Fetch simple Wikipedia pageview metric as a proxy for awareness."""
    try:
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia.org/all-access/all-agents/"
            f"{requests.utils.quote(title)}/daily/20230101/20231231"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        views = [item["views"] for item in data.get("items", [])]
        if not views:
            return 0.0
        return float(sum(views))
    except Exception:
        return 0.0


def fetch_google_trends_score(title: str) -> float:
    """Fetch Google Trends 0–100 score for the last 12 months."""
    try:
        kw_list = [title]
        pytrends.build_payload(kw_list, timeframe="today 12-m")
        data = pytrends.interest_over_time()
        if data.empty:
            return 0.0
        return float(data[title].mean())
    except Exception:
        return 0.0


def fetch_youtube_metric(title: str) -> float:
    """Fetch a simple YouTube relevance metric."""
    if youtube is None:
        return 0.0
    try:
        req = youtube.search().list(
            q=title,
            part="snippet",
            maxResults=10,
            type="video",
        )
        resp = req.execute()
        return float(len(resp.get("items", [])))
    except Exception:
        return 0.0


def fetch_spotify_metric(title: str) -> float:
    """Fetch a simple Spotify search hit count metric."""
    if sp is None:
        return 0.0
    try:
        results = sp.search(q=title, limit=10, type="track")
        items = results.get("tracks", {}).get("items", [])
        return float(len(items))
    except Exception:
        return 0.0


def normalize_0_100(values: List[float]) -> List[float]:
    """Normalize a list of raw values to the 0–100 range."""
    if not values:
        return []
    v = [0.0 if (x is None or math.isnan(x)) else float(x) for x in values]
    v_min = min(v)
    v_max = max(v)
    if v_max == v_min:
        # all equal or single item → all 50
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
        format_func=lambda x: f"{int(x*100)}%",
        index=0,
    )

titles: List[str] = [
    t.strip() for t in titles_raw.splitlines() if t.strip()
]

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

        # Normalize each channel to 0–100
        wiki_norm = normalize_0_100(df_raw["wiki_raw"].tolist())
        trends_norm = normalize_0_100(df_raw["trends_raw"].tolist())
        youtube_norm = normalize_0_100(df_raw["youtube_raw"].tolist())
        spotify_norm = normalize_0_100(df_raw["spotify_raw"].tolist())

        df_raw["wiki"] = wiki_norm
        df_raw["trends"] = trends_norm
        df_raw["youtube"] = youtube_norm
        df_raw["spotify"] = spotify_norm
