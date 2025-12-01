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
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube, and Spotify for title demand scoring."
)


# =============================================================================
# API HELPERS
# =============================================================================


def _safe_get_env(name: str) -> str | None:
    import os
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value


def _build_youtube_client():
    api_key = _safe_get_env("YOUTUBE_API_KEY")
    if not api_key:
        return None
    return build("youtube", "v3", developerKey=api_key)


def _build_spotify_client():
    client_id = _safe_get_env("SPOTIFY_CLIENT_ID")
    client_secret = _safe_get_env("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_wikipedia_pageviews(title: str) -> float:
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/all-agents/"
        + requests.utils.quote(title.replace(" ", "_"))
        + "/daily/20240101/20241231"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return 0.0
    data = resp.json()
    views = [item.get("views", 0) for item in data.get("items", [])]
    if not views:
        return 0.0
    return float(sum(views)) / float(len(views))


def fetch_google_trends_score(title: str) -> float:
    try:
        pytrends.build_payload([title], cat=0, timeframe="today 12-m", geo="", gprop="")
        data = pytrends.interest_over_time()
        if data.empty:
            return 0.0
        series = data[title]
        return float(series.mean())
    except Exception:
        return 0.0


def fetch_youtube_score(title: str, yt_client) -> float:
    if yt_client is None:
        return 0.0
    try:
        resp = yt_client.search().list(
            q=title,
            part="snippet",
            maxResults=5,
            type="video",
        ).execute()
        items = resp.get("items", [])
        if not items:
            return 0.0
        score = len(items)
        return float(score)
    except Exception:
        return 0.0


def fetch_spotify_score(title: str, sp_client) -> float:
    if sp_client is None:
        return 0.0
    try:
        results = sp_client.search(q=title, limit=5, type="track")
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return 0.0
        popularity_vals = [track.get("popularity", 0) for track in items]
        if not popularity_vals:
            return 0.0
        return float(sum(popularity_vals)) / float(len(popularity_vals))
    except Exception:
        return 0.0


def normalize_0_100(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        return [50.0 for _ in vals]
    return [
        100.0 * (val - vmin) / float(vmax - vmin)
        for val in vals
    ]


# =============================================================================
# MODEL SCORING WRAPPER FOR SINGLE TITLE
# =============================================================================


def score_single_run(
    features: Dict[str, Any],
    confidence_level: float = 0.8,
) -> Dict[str, float]:
    """
    Score a single hypothetical run from a dict of features.

    This expects `features` to use the same feature names as the training
    dataset (e.g. wiki, trends, youtube, spotify, genre, season, etc.).
    """
    df = pd.DataFrame([features])
    df_scored = score_runs_for_planning(
        df,
        confidence_level=confidence_level,
        n_bootstrap=200,
        model=None,
        attach_context=False,
        economic_context=None,
    )
    row = df_scored.iloc[0]

    pct = int(confidence_level * 100.0)
    lower_col = "lower_tickets_" + str(pct)
    upper_col = "upper_tickets_" + str(pct)

    return {
        "pred_tickets": float(row.get("forecast_single_tickets", 0.0)),
        "lower_tickets": float(row.get(lower_col, 0.0)),
        "upper_tickets": float(row.get(upper_col, 0.0)),
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================


def main() -> None:
    st.subheader("Step 1 – Enter Titles")

    titles_raw = st.text_area(
        "Enter one title per line",
        value="The Nutcracker\nSwan Lake\nNew Contemporary Work",
        height=120,
    )

    if not titles_raw.strip():
        st.stop()

    titles = [t.strip() for t in titles_raw.split("\n") if t.strip()]
    if not titles:
        st.stop()

    st.subheader("Step 2 – Fetch Signals")

    col1, col2 = st.columns(2)

    with col1:
        run_fetch = st.button("Fetch Scores", type="primary")
    with col2:
        confidence_level = st.slider(
            "Prediction interval confidence",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
        )

    if not run_fetch:
        st.info("Enter titles and click Fetch Scores to continue.")
        st.stop()

    yt_client = _build_youtube_client()
    sp_client = _build_spotify_client()

    rows: List[Dict[str, Any]] = []

    for title in titles:
        wiki_val = fetch_wikipedia_pageviews(title)
        trends_val = fetch_google_trends_score(title)
        yt_val = fetch_youtube_score(title, yt_client)
        sp_val = fetch_spotify_score(title, sp_client)

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

    st.subheader("Step 3 – Normalized Scores (0–100)")

    st.dataframe(
        df_raw[["title", "wiki", "trends", "youtube", "spotify"]],
        use_container_width=True,
    )

    st.subheader("Step 4 – Ticket Forecasts")

    default_genre = st.selectbox(
        "Default genre for all titles (can override later offline)",
        options=["classical", "contemporary", "family", "mixed"],
        index=0,
    )
    default_season = st.selectbox(
        "Season label (for model feature)",
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

    df_scored = score_runs_for_planning(
        df_features.drop(columns=["title"]),
        confidence_level=confidence_level,
        n_bootstrap=200,
        model=None,
        attach_context=False,
        economic_context=None,
    )

    pct = int
