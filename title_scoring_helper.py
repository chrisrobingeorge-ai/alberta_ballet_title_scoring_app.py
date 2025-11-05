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

st.set_page_config(page_title="Title Scoring Helper", layout="wide")

st.title("Title Scoring Helper")
st.caption(
    "Fetch and normalize 0–100 scores from Wikipedia, Google Trends, YouTube and Spotify "
    "for each title, then paste the generated BASELINES dict into your main app."
)

# ------------------------------------------------------------------
# 0) Same infer_gender_and_category as your main app
# ------------------------------------------------------------------
def infer_gender_and_category(title: str) -> tuple[str, str]:
    t = title.lower()
    gender = "na"
    female_keys = [
        "cinderella", "sleeping", "beauty and the beast", "beauty",
        "giselle", "swan", "widow", "alice", "juliet", "sylphide"
    ]
    male_keys = [
        "pinocchio", "peter pan", "don quixote", "hunchback",
        "hamlet", "frankenstein", "romeo", "nijinsky"
    ]
    if "romeo" in t and "juliet" in t:
        gender = "co"
    elif any(k in t for k in female_keys):
        gender = "female"
    elif any(k in t for k in male_keys):
        gender = "male"

    if any(k in t for k in ["wizard", "peter pan", "pinocchio", "hansel", "frozen", "beauty", "alice"]):
        cat = "family_classic"
    elif any(k in t for k in ["swan", "sleeping", "cinderella", "giselle", "sylphide"]):
        cat = "classic_romance"
    elif any(k in t for k in ["romeo", "hunchback", "notre dame", "hamlet", "frankenstein"]):
        cat = "romantic_tragedy"
    elif any(k in t for k in ["don quixote", "merry widow"]):
        cat = "classic_comedy"
    elif any(k in t for k in [
        "contemporary", "boyz", "ballet boyz", "momix", "complexions",
        "grimm", "nijinsky", "shadowland", "deviate", "phi"
    ]):
        cat = "contemporary"
    elif any(k in t for k in ["taj", "tango", "harlem", "tragically hip", "l cohen", "leonard cohen"]):
        cat = "pop_ip"
    else:
        cat = "dramatic"
    return gender, cat

# ------------------------------------------------------------------
# 1) Sidebar: titles + fallback defaults + API keys
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Titles")
    raw_titles = st.text_area(
        "Titles (one per line)",
        value="Cinderella\nAlice in Wonderland\nSwan Lake",
        height=200,
    )

    st.header("Fallback defaults (used only if heuristic is weak)")
    default_category = st.selectbox(
        "Default category (fallback)",
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
        "Default gender (fallback)",
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
import math
import requests
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{page}/daily/{start}/{end}"
)

def wiki_search_best_title(query: str) -> str | None:
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        headers = {
            "User-Agent": "AB-TitleScorer/1.0 (contact: your-email@example.com)"
        }
        r = requests.get(WIKI_API, params=params, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        items = r.json().get("query", {}).get("search", [])
        return items[0]["title"] if items else None
    except Exception:
        return None

WIKI_PAGE_OVERRIDE = {
    "Peter Pan": "Peter Pan",                     # main franchise page
    "Alice in Wonderland": "Alice's Adventures in Wonderland",
    "Nijinsky": "Vaslav Nijinsky",
    # add any others you care about
}

def fetch_wiki_raw(title: str) -> float:
    try:
        # Use override first, then search, then raw title
        override = WIKI_PAGE_OVERRIDE.get(title.strip())
        page_title = override or wiki_search_best_title(title) or title
        page_slug = page_title.replace(" ", "_")

        headers = {
            "User-Agent": "AB-TitleScorer/1.0 (contact: your-email@example.com)"
        }

        url = WIKI_PAGEVIEW.format(
            page=page_slug,
            start="20240101",
            end="20240131",
        )
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            # API not cooperating → fall back
            raise RuntimeError(f"Wiki pageviews HTTP {r.status_code}")

        data = r.json()
        items = data.get("items", [])
        views = [it.get("views", 0) for it in items]
        total = float(sum(views))

        # If still zero, fall back
        if total <= 0:
            raise RuntimeError("Wiki pageviews zero")

        return total
    except Exception:
        # Fallback heuristic if API is blocked: scale by title length
        return float(len(title) * 10.0)


def fetch_trends_raw(title: str) -> float:
    """
    Google Trends, Alberta-only (CA-AB), avg interest over last 12 months.
    """
    try:
        kw = title.strip()
        if not kw:
            return 0.0

        # note geo="CA-AB" instead of ""
        pytrends.build_payload([kw], cat=0, timeframe="today 12-m",
                               geo="CA-AB", gprop="")
        df = pytrends.interest_over_time()

        if df.empty or kw not in df.columns:
            return 0.0

        series = df[kw].astype(float)
        if series.sum() == 0:
            return 0.0

        return float(series.mean())
    except Exception:
        return 0.0

def fetch_trends_city_raw(title: str, city_name: str) -> float:
    """
    Google Trends city-level interest (e.g. 'Calgary', 'Edmonton'),
    relative 0–100 within Canada over last 12 months.
    """
    try:
        kw = title.strip()
        if not kw:
            return 0.0

        pytrends.build_payload([kw], cat=0, timeframe="today 12-m",
                               geo="CA", gprop="")
        df = pytrends.interest_by_region(
            resolution="CITY",
            inc_low_vol=True,
            inc_geo_code=False,
        )
        if df.empty or kw not in df.columns:
            return 0.0

        mask = df.index.str.contains(city_name, case=False, na=False)
        sub = df.loc[mask, kw]
        if sub.empty:
            return 0.0

        return float(sub.iloc[0])  # already 0–100
    except Exception:
        return 0.0


def fetch_youtube_raw(youtube, title: str) -> float:
    """
    YouTube: log-transformed total view counts for top 5 results.
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
        return float(math.log10(total_views + 1.0))
    except Exception:
        return 0.0


def fetch_spotify_raw(sp: spotipy.Spotify | None, title: str) -> float:
    """
    Spotify: max track popularity (0–100) among top 5 results.
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
        youtube = build("youtube", "v3", developerKey=yt_api_key) if yt_api_key else None

        spotify = None
        if sp_client_id and sp_client_secret:
            auth = SpotifyClientCredentials(
                client_id=sp_client_id,
                client_secret=sp_client_secret,
            )
            spotify = spotipy.Spotify(client_credentials_manager=auth)

        raw_wiki = []
        raw_trends = []          # Alberta-wide (CA-AB)
        raw_trends_cgy = []      # Calgary
        raw_trends_yeg = []      # Edmonton
        raw_youtube = []
        raw_spotify = []
        genders = []
        categories = []

        for t in titles:
            g_infer, c_infer = infer_gender_and_category(t)
            gender = g_infer if g_infer != "na" else default_gender
            category = c_infer if c_infer != "dramatic" else default_category

            genders.append(gender)
            categories.append(category)

            raw_wiki.append(fetch_wiki_raw(t))
            raw_trends.append(fetch_trends_raw(t))                 # Alberta-only (CA-AB) ONCE
            raw_trends_cgy.append(fetch_trends_city_raw(t, "Calgary"))
            raw_trends_yeg.append(fetch_trends_city_raw(t, "Edmonton"))
            raw_youtube.append(fetch_youtube_raw(youtube, t))
            raw_spotify.append(fetch_spotify_raw(spotify, t))

        wiki_scores = normalize_0_100_log(raw_wiki)
        trends_scores = normalize_0_100_log(raw_trends)
        youtube_scores = normalize_0_100_log(raw_youtube)
        spotify_scores = normalize_0_100_log(raw_spotify)

    df = pd.DataFrame({
        "title": titles,
        "wiki_raw": raw_wiki,
        "trends_raw": raw_trends,               # Alberta
        "trends_calgary_raw": raw_trends_cgy,
        "trends_edmonton_raw": raw_trends_yeg,
        "youtube_raw": raw_youtube,
        "spotify_raw": raw_spotify,
        "wiki": wiki_scores,
        "trends": trends_scores,
        "youtube": youtube_scores,
        "spotify": spotify_scores,
        "category": categories,
        "gender": genders,
    })

    st.subheader("Raw & normalized scores (with inferred category/gender)")
    st.dataframe(
        df[[
            "title",
            "wiki_raw",
            "trends_raw",
            "trends_calgary_raw",
            "trends_edmonton_raw",
            "youtube_raw",
            "spotify_raw",
            "wiki", "trends", "youtube", "spotify",
            "category", "gender",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # 4) Generated BASELINES dict
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
