from __future__ import annotations
from typing import List, Dict, Any
import math

import requests
import pandas as pd
import streamlit as st

from pytrends.request import TrendReq
from googleapiclient.discovery import build
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="Title Scoring Helper", layout="wide")

st.title("Title Scoring Helper")
st.caption(
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube and Spotify "
    "for each title, then paste the generated BASELINES dict into your main app."
)

# ------------------------------------------------------------------
# 1) Sidebar: titles + defaults + API keys
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Titles")
    raw_titles = st.text_area(
        "Titles (one per line)",
        value="Cinderella\nAlice in Wonderland\nSwan Lake",
        height=200,
    )

    st.header("Defaults (for category/gender)")
    default_category = st.selectbox(
        "Default category",
        options=[
            "family_classic",
            "classic_romance",
            "classic_comedy",
            "romantic_tragedy",
            "romantic_comedy",
            "contemporary",
            "dramatic",
            "pop_ip",
        ],
        index=0,
    )
    default_gender = st.selectbox(
        "Default gender",
        options=["female", "male", "co", "na"],
        index=0,
    )

    st.header("API keys / credentials")
    yt_api_key = st.text_input(
        "YouTube Data API key",
        type="password",
        help="From Google Cloud; used to fetch view counts.",
    )
    sp_client_id = st.text_input(
        "Spotify client ID",
        type="password",
        help="From Spotify developer dashboard.",
    )
    sp_client_secret = st.text_input(
        "Spotify client secret",
        type="password",
    )

    st.markdown(
        "_Keys are only used in this helper app and not stored anywhere._"
    )

    run_button = st.button("Run scoring")

titles: List[str] = [t.strip() for t in raw_titles.splitlines() if t.strip()]
if not titles:
    st.info("Enter at least one title in the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------
# 2) Fetchers
# ------------------------------------------------------------------

def fetch_wiki_raw(title: str) -> float:
    """
    Use Wikipedia pageviews (last 365 days, enwiki) as a raw familiarity signal.
    """
    try:
        from datetime import datetime, timedelta

        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/"
            f"{requests.utils.quote(title.replace(' ', '_'))}/daily/{start}/{end}"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return 0.0
        items = r.json().get("items", [])
        views = [it.get("views", 0) for it in items]
        return float(sum(views) / 365.0) if views else 0.0
    except Exception:
        return 0.0


def fetch_trends_raw(pytrend: TrendReq, title: str) -> float:
    """
    Google Trends: average interest over last 5 years.
    """
    try:
        pytrend.build_payload([title], cat=0, timeframe="today 5-y", geo="", gprop="")
        df = pytrend.interest_over_time()
        if df.empty or title not in df.columns:
            return 0.0
        return float(df[title].mean())
    except Exception:
        return 0.0


def fetch_youtube_raw(youtube, title: str) -> float:
    """
    YouTube: log-transformed sum of view counts for top few results.
    """
    if youtube is None:
        return 0.0
    try:
        search_resp = youtube.search().list(
            q=title,
            part="snippet",
            type="video",
            maxResults=5,
        ).execute()
        video_ids = [
            item["id"]["videoId"]
            for item in search_resp.get("items", [])
            if "id" in item and "videoId" in item["id"]
        ]
        if not video_ids:
            return 0.0
        stats_resp = youtube.videos().list(
            id=",".join(video_ids),
            part="statistics",
        ).execute()
        total_views = 0
        for item in stats_resp.get("items", []):
            vc = item.get("statistics", {}).get("viewCount")
            if vc is not None:
                total_views += int(vc)
        # log so huge channels don't dominate
        return float(math.log10(total_views + 1.0))
    except Exception:
        return 0.0


def fetch_spotify_raw(sp: spotipy.Spotify | None, title: str) -> float:
    """
    Spotify: max track popularity (0–100) among top results.
    """
    if sp is None:
        return 0.0
    try:
        res = sp.search(q=title, type="track", limit=5)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return 0.0
        return float(max(track.get("popularity", 0) for track in items))
    except Exception:
        return 0.0


def normalize_0_100_log(values: List[float]) -> List[int]:
    """
    Take raw positive values and normalize to 0–100 across the batch,
    with log(1+x) compression for skewed distributions.
    """
    if not values:
        return []
    logs = [math.log1p(max(v, 0.0)) for v in values]
    v_min = min(logs)
    v_max = max(logs)
    if v_max <= v_min:
        # all equal (or all zero)
        return [0 for _ in logs]
    return [
        int(round(100 * (lv - v_min) / (v_max - v_min)))
        for lv in logs
    ]


# ------------------------------------------------------------------
# 3) Run scoring
# ------------------------------------------------------------------
if run_button:
    with st.spinner("Fetching Wikipedia/Trends/YouTube/Spotify…"):
        # Build clients
        pytrend = TrendReq(hl="en-US", tz=0)

        youtube = build("youtube", "v3", developerKey=yt_api_key) if yt_api_key else None

        spotify = None
        if sp_client_id and sp_client_secret:
            auth = SpotifyClientCredentials(
                client_id=sp_client_id,
                client_secret=sp_client_secret,
            )
            spotify = spotipy.Spotify(client_credentials_manager=auth)

        raw_wiki = []
        raw_trends = []
        raw_youtube = []
        raw_spotify = []

        for t in titles:
            raw_wiki.append(fetch_wiki_raw(t))
            raw_trends.append(fetch_trends_raw(pytrend, t))
            raw_youtube.append(fetch_youtube_raw(youtube, t))
            raw_spotify.append(fetch_spotify_raw(spotify, t))

        wiki_scores = normalize_0_100_log(raw_wiki)
        trends_scores = normalize_0_100_log(raw_trends)
        youtube_scores = normalize_0_100_log(raw_youtube)
        spotify_scores = normalize_0_100_log(raw_spotify)

    df = pd.DataFrame({
        "title": titles,
        "wiki_raw": raw_wiki,
        "trends_raw": raw_trends,
        "youtube_raw": raw_youtube,
        "spotify_raw": raw_spotify,
        "wiki": wiki_scores,
        "trends": trends_scores,
        "youtube": youtube_scores,
        "spotify": spotify_scores,
        "category": default_category,
        "gender": default_gender,
    })

    st.subheader("Raw & normalized scores")
    st.dataframe(
        df[[
            "title",
            "wiki_raw", "trends_raw", "youtube_raw", "spotify_raw",
            "wiki", "trends", "youtube", "spotify",
            "category", "gender",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # 4) Generate BASELINES dict snippet
    # ------------------------------------------------------------------
    st.subheader("Generated BASELINES dict (copy into main app)")

    lines: List[str] = ["BASELINES = {"]

    for _, row in df.iterrows():
        title = str(row["title"]).replace('"', '\\"')
        line = (
            f'    "{title}": {{'
            f'"wiki": {int(row["wiki"])}, '
            f'"trends": {int(row["trends"])}, '
            f'"youtube": {int(row["youtube"])}, '
            f'"spotify": {int(row["spotify"])}, '
            f'"category": "{row["category"]}", '
            f'"gender": "{row["gender"]}"'
            f"}},"
        )
        lines.append(line)

    lines.append("}")
    baselines_text = "\n".join(lines)
    st.code(baselines_text, language="python")

    st.download_button(
        "⬇️ Download scores as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="show_baselines_generated.csv",
        mime="text/csv",
    )

else:
    st.info("Set your titles and API keys (optional), then click **Run scoring** in the sidebar.")
