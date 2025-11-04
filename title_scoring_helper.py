from __future__ import annotations
from typing import List, Dict
import os
import math
import requests
import pandas as pd
import streamlit as st

# Optional external libraries if you want to wire them up:
# from pytrends.request import TrendReq      # Google Trends
# from googleapiclient.discovery import build  # YouTube Data API
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

st.set_page_config(page_title="Title Scoring Helper", layout="wide")

st.title("Title Scoring Helper")
st.caption(
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube and Spotify "
    "for each title. Copy the generated BASELINES dict into your main app."
)

# ----------------------------
# 1. Inputs
# ----------------------------
with st.sidebar:
    st.header("Input titles")
    raw_titles = st.text_area(
        "Titles (one per line)",
        value="Cinderella\nAlice in Wonderland\nSwan Lake",
        height=200,
    )

    st.header("Defaults")
    default_category = st.selectbox(
        "Default category (if you don't fill manually later)",
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
        "Default gender (if you don't fill manually later)",
        options=["female", "male", "co", "na"],
        index=0,
    )

    run_button = st.button("Run scoring")

titles: List[str] = [
    t.strip() for t in raw_titles.splitlines() if t.strip()
]

if not titles:
    st.info("Enter at least one title in the sidebar to begin.")
    st.stop()

# ----------------------------
# 2. Fetcher stubs / simple implementations
# ----------------------------

def fetch_wiki_raw(title: str) -> float:
    """
    Example: use Wikipedia pageviews as a raw signal.
    This is a minimal implementation (30 days pageviews).
    You can refine it as you like.
    """
    try:
        # 30-day pageviews via REST API; using en.wikipedia
        # NOTE: you'll likely want to adjust dates for real usage.
        # Here we just give structure; you can refine date ranges.
        session = requests.Session()
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/"
            f"{requests.utils.quote(title.replace(' ', '_'))}/daily/20240101/20240131"
        )
        r = session.get(url, timeout=5)
        if r.status_code != 200:
            return 0.0
        data = r.json().get("items", [])
        views = sum(item.get("views", 0) for item in data)
        return float(views)
    except Exception:
        return 0.0


def fetch_trends_raw(title: str) -> float:
    """
    Placeholder: if you wire Google Trends via pytrends,
    return something like average interest over last 12 months.
    For now, just return 0 and you can implement later.
    """
    # Example of structure (commented out):
    # pytrends = TrendReq(hl="en-US", tz=0)
    # pytrends.build_payload([title], cat=0, timeframe="today 12-m")
    # df = pytrends.interest_over_time()
    # if df.empty:
    #     return 0.0
    # return float(df[title].mean())
    return 0.0


def fetch_youtube_raw(title: str) -> float:
    """
    Placeholder: if you wire YouTube Data API, you might:
    - search for the title
    - pick top result's viewCount
    For now, return 0.
    """
    # api_key = os.environ.get("YOUTUBE_API_KEY")
    # if not api_key:
    #     return 0.0
    # yt = build("youtube", "v3", developerKey=api_key)
    # res = yt.search().list(q=title, part="snippet", type="video", maxResults=1).execute()
    # ...
    return 0.0


def fetch_spotify_raw(title: str) -> float:
    """
    Placeholder: if you wire Spotify via spotipy, you might:
    - search for a track / show title
    - take 'popularity' (0–100) directly.
    For now, return 0.
    """
    # client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    # client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    # if not (client_id and client_secret):
    #     return 0.0
    # auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    # sp = spotipy.Spotify(auth_manager=auth)
    # res = sp.search(q=title, type="track", limit=1)
    # ...
    return 0.0


def normalize_to_0_100(values: List[float]) -> List[int]:
    """
    Take raw positive values and normalize to 0–100 across the batch.
    Uses log(1+x) scaling for skewed distributions.
    If all zeros, returns all zeros.
    """
    if not values:
        return []

    logs = [math.log1p(max(v, 0.0)) for v in values]
    m = min(logs)
    M = max(logs)
    if M <= m:
        return [0 for _ in values]
    return [int(round(100 * (lv - m) / (M - m))) for lv in logs]


# ----------------------------
# 3. Run scoring
# ----------------------------
if run_button:
    with st.spinner("Fetching & normalizing scores..."):
        raw_wiki = [fetch_wiki_raw(t) for t in titles]
        raw_trends = [fetch_trends_raw(t) for t in titles]
        raw_youtube = [fetch_youtube_raw(t) for t in titles]
        raw_spotify = [fetch_spotify_raw(t) for t in titles]

        wiki_scores = normalize_to_0_100(raw_wiki)
        trends_scores = normalize_to_0_100(raw_trends)
        youtube_scores = normalize_to_0_100(raw_youtube)
        spotify_scores = normalize_to_0_100(raw_spotify)

    df = pd.DataFrame({
        "title": titles,
        "wiki": wiki_scores,
        "trends": trends_scores,
        "youtube": youtube_scores,
        "spotify": spotify_scores,
        "category": default_category,
        "gender": default_gender,
    })

    st.subheader("Raw scores (you can export this to CSV too)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Generate BASELINES dict snippet
    st.subheader("Generated BASELINES dict (copy into main app)")
    lines: List[str] = ["BASELINES = {"]

    for _, row in df.iterrows():
        title = row["title"].replace('"', '\\"')
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
        "Download scores as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="show_baselines_generated.csv",
        mime="text/csv",
    )
else:
    st.info("Click **Run scoring** in the sidebar to fetch scores and generate the BASELINES dict.")
