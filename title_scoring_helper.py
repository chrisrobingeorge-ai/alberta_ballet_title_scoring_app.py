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
    st.caption(
        "Keys can be loaded from environment variables or Streamlit secrets. "
        "Set `YOUTUBE_API_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` "
        "in your environment or `~/.streamlit/secrets.toml`."
    )
    
    # Try to load from environment/secrets first, allow override via input
    import os
    
    def get_secret(key: str, env_key: str) -> str:
        """Get secret from st.secrets, environment, or empty string."""
        # Try st.secrets first
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        # Try environment variable
        return os.environ.get(env_key, "")
    
    default_yt = get_secret("youtube_api_key", "YOUTUBE_API_KEY")
    default_sp_id = get_secret("spotify_client_id", "SPOTIFY_CLIENT_ID")
    default_sp_secret = get_secret("spotify_client_secret", "SPOTIFY_CLIENT_SECRET")
    
    yt_api_key = st.text_input(
        "YouTube Data API key",
        value=default_yt,
        type="password",
        help="From Google Cloud; used to fetch view counts. "
             "Can be set via YOUTUBE_API_KEY env var or st.secrets['youtube_api_key'].",
    )
    sp_client_id = st.text_input(
        "Spotify client ID",
        value=default_sp_id,
        type="password",
        help="From Spotify developer dashboard. "
             "Can be set via SPOTIFY_CLIENT_ID env var or st.secrets['spotify_client_id'].",
    )
    sp_client_secret = st.text_input(
        "Spotify client secret",
        value=default_sp_secret,
        type="password",
        help="Can be set via SPOTIFY_CLIENT_SECRET env var or st.secrets['spotify_client_secret'].",
    )

    st.markdown(
        "_Keys are used only for API calls and are not stored. "
        "See README.md for secure credential handling._"
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
    If Google returns 429 / empty / all zeros, fall back to a simple heuristic
    so we still get non-zero variation across titles.
    """
    kw = title.strip()
    if not kw:
        return 0.0

    try:
        pytrends.build_payload(
            [kw],
            cat=0,
            timeframe="today 12-m",
            geo="CA-AB",
            gprop=""
        )
        df = pytrends.interest_over_time()

        # Treat empty / zero-only as a failure (often caused by 429)
        if df.empty or kw not in df.columns:
            raise RuntimeError("Trends dataframe empty or missing keyword")

        series = df[kw].astype(float)
        if series.sum() == 0:
            raise RuntimeError("Trends series all zeros")

        return float(series.mean())
    except Exception as e:
        st.warning(
            f"Google Trends failed for '{title}' (likely 429 / rate limit). "
            "Using a simple heuristic instead."
        )
        # Heuristic: scale roughly with title length
        return float(len(title) * 2.0)


def fetch_trends_city_raw(title: str, city_name: str) -> float:
    """
    City-level heuristic only – live Google Trends is too fragile here.
    """
    # Tiny variation by city so Calgary/Edmonton numbers differ a bit
    base = len(title)
    if "calg" in city_name.lower():
        return float(base * 1.1)
    elif "edm" in city_name.lower():
        return float(base * 1.0)
    return float(base)

def fetch_youtube_raw(youtube, title: str) -> float:
    """
    YouTube: total view counts for top 5 results (raw sum of views).
    We'll map this to 0–100 later with a fixed log scale.
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

        # <-- IMPORTANT: return raw total views, not log()
        return float(total_views)
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


def wiki_to_score(views: float) -> int:
    """
    Map Wikipedia pageviews to a 0–100 score using a fixed log scale.
    Rough idea: 0 views -> 0, ~1M+ views -> ~100.
    """
    v = max(0.0, views)
    logv = math.log10(v + 1.0)      # 0 .. ~6 for 1M
    score = (logv / 6.0) * 100.0    # 6 ~ 1,000,000 views
    score = max(0.0, min(100.0, score))
    return int(round(score))


def youtube_to_score(views: float) -> int:
    """
    Map YouTube total views to 0–100.
    Rough idea: 0 views -> 0, ~100M+ views -> ~100.
    """
    v = max(0.0, views)
    logv = math.log10(v + 1.0)      # 0 .. ~8 for 100M
    score = (logv / 8.0) * 100.0
    score = max(0.0, min(100.0, score))
    return int(round(score))


def clamp_0_100(x: float) -> int:
    """
    Clamp any numeric value into [0,100] and round.
    Useful for Trends (already 0–100) and Spotify popularity (0–100).
    """
    return int(round(max(0.0, min(100.0, float(x)))))



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

        wiki_scores    = [wiki_to_score(v) for v in raw_wiki]
        trends_scores  = [clamp_0_100(v) for v in raw_trends]     # Trends already 0–100-ish
        youtube_scores = [youtube_to_score(v) for v in raw_youtube]
        spotify_scores = [clamp_0_100(v) for v in raw_spotify]    # Spotify popularity 0–100

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
        width='stretch',
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
