# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation Scorer
# Full replacement version with:
# - Robust Google Trends (AB) + global fallback
# - Wikipedia "(ballet)" and variant fallbacks
# - Smart benchmark normalization (skips flat columns)
# - Diagnostics table
# - One-click PDF brief (table, quadrant, bars)
# - Optional YouTube/Spotify signals (keys optional)

import os, math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pytrends.request import TrendReq
import requests

# Optional APIs
try:
    from googleapiclient.discovery import build  # YouTube Data API v3
except Exception:
    build = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

# -------------------------
# CONFIG & WEIGHTS
# -------------------------
AB_GEO = "CA-AB"            # Alberta for Google Trends
TRENDS_TIMEFRAME = "today 5-y"

# Weights are relative; function normalizes automatically
WEIGHTS = {
    "search_trends": 0.35,   # Google Trends
    "youtube":       0.25,   # YouTube search + top view (optional)
    "spotify":       0.10,   # Music recognition (optional)
    "wikipedia":     0.05,   # Pageviews
    # remaining weight is implicitly distributed by normalization
}

EPS = 1e-9

# -------------------------
# UTILITIES
# -------------------------
def normalize_series(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = float(min(values)), float(max(values))
    if abs(vmax - vmin) < EPS:
        return [50.0 for _ in values]
    return [ (float(v) - vmin) * 100.0 / (vmax - vmin) for v in values ]

def capped_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = sorted([float(v) for v in values])
    n = len(arr)
    k = max(1, int(0.05 * n)) if n >= 20 else 0  # trim 5% if enough points
    trimmed = arr[k: n-k] if k > 0 else arr
    return float(sum(trimmed)) / len(trimmed)

def has_variance(series: List[float]) -> bool:
    return (max(series) - min(series)) > 1e-6 if series else False

# -------------------------
# DATA SOURCES (cached)
# -------------------------
@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_google_trends_score(title: str, use_global_fallback: bool) -> float:
    """
    Query Google Trends independently per title to avoid per-batch normalization.
    Try multiple phrasing variants; pick the strongest Alberta signal, else global.
    Returns a 0..100 average interest score.
    """
    variants = [title, f"{title} ballet", f"{title} story", f"{title} music"]
    scores = []

    def _query(kw: str, geo: str) -> float:
        try:
            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload([kw], geo=geo, timeframe=TRENDS_TIMEFRAME)
            df = pytrends.interest_over_time()
            if df.empty or kw not in df.columns:
                return 0.0
            return capped_mean(df[kw].astype(float).tolist())  # already 0..100
        except Exception:
            return 0.0

    # Alberta first
    for kw in variants:
        s = _query(kw, AB_GEO)
        if s > 0:
            scores.append(s)

    # Global fallback if Alberta is flat and toggle is enabled
    if not scores and use_global_fallback:
        for kw in variants:
            s = _query(kw, "")
            if s > 0:
                scores.append(s)

    return max(scores) if scores else 0.0

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_wikipedia_views(title: str) -> float:
    """
    Average daily pageviews over last 12 months for likely pages.
    Tries title, 'Title (ballet)', and common 'The Title' variant.
    """
    def _pageviews(page_name: str) -> float:
        try:
            end = datetime.utcnow().strftime("%Y%m%d")
            start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
            page = page_name.replace(" ", "_")
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                return 0.0
            items = r.json().get("items", [])
            views = [it.get("views", 0) for it in items]
            return (sum(views) / 365.0) if views else 0.0
        except Exception:
            return 0.0

    candidates = [
        title,
        f"{title} (ballet)",
        f"The {title}",
        f"The {title} (ballet)"
    ]
    return max(_pageviews(c) for c in candidates)

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float]:
    """
    Returns (search_count, top_view_sqrt) as proxies.
    Requires YouTube API key; otherwise (0.0, 0.0).
    """
    if not api_key or build is None:
        return 0.0, 0.0
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        search = yt.search().list(q=title, part="id", type="video", maxResults=50).execute()
        ids = [item["id"]["videoId"] for item in search.get("items", [])]
        search_count = len(ids)
        if not ids:
            return 0.0, 0.0
        chunks = [ids[i:i+50] for i in range(0, len(ids), 50)]
        top_views = 0
        for ch in chunks:
            stats = yt.videos().list(part="statistics", id=",".join(ch)).execute()
            for it in stats.get("items", []):
                vc = int(it.get("statistics", {}).get("viewCount", 0))
                top_views = max(top_views, vc)
        return float(search_count), math.sqrt(float(top_views))
    except Exception:
        return 0.0, 0.0

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_spotify_popularity(title: str, client_id: Optional[str], client_secret: Optional[str]) -> float:
    """
    Max popularity (0..100) among top tracks matching the title.
    Requires Spotify creds; otherwise 0.
    """
    if not client_id or not client_secret or spotipy is None:
        return 0.0
    try:
        auth_mgr = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_mgr)
        res = sp.search(q=title, type="track", limit=10)
        pops = [t.get("popularity", 0) for t in res.get("tracks", {}).get("items", [])]
        return max(pops) if pops else 0.0
    except Exception:
        return 0.0

# -------------------------
# SCORING
# -------------------------
def weighted_average(components: Dict[str, float], weights: Dict[str, float]) -> float:
    num, den = 0.0, 0.0
    for key, val in components.items():
        if val is None:
            continue
        w = float(weights.get(key, 0.0))
        num += float(val) * w
        den += w
    return (num / den) if den > 0 else 0.0  # stays in 0..100 if components are 0..100

def score_titles(titles: List[str],
                 yt_key: Optional[str],
                 sp_id: Optional[str],
                 sp_secret: Optional[str],
                 use_global_trends: bool) -> pd.DataFrame:

    trends_scores, wiki_scores, yt_counts, yt_topsqrt, sp_pops = [], [], [], [], []

    for t in titles:
        st.write(f"Fetching metrics for **{t}** …")
        ts = fetch_google_trends_score(t, use_global_trends)
        ws = fetch_wikipedia_views(t)
        yc, yt = fetch_youtube_metrics(t, yt_key)
        sp = fetch_spotify_popularity(t, sp_id, sp_secret)

        trends_scores.append(ts)      # already 0..100
        wiki_scores.append(ws)
        yt_counts.append(yc)
        yt_topsqrt.append(yt)
        sp_pops.append(sp)

    # Normalize per metric (0..100)
    wiki_n = normalize_series(wiki_scores)
    yt_count_n = normalize_series(yt_counts)
    yt_top_n = normalize_series(yt_topsqrt)
    yt_combo = [capped_mean([a, b]) for a, b in zip(yt_count_n, yt_top_n)]
    sp_n = normalize_series(sp_pops)

    rows = []
    for i, t in enumerate(titles):
        fam = weighted_average(
            {"search_trends": trends_scores[i], "wikipedia": wiki_n[i], "spotify": sp_n[i]},
            WEIGHTS
        )
        mot = weighted_average(
            {"search_trends": trends_scores[i], "youtube": yt_combo[i], "spotify": sp_n[i]},
            WEIGHTS
        )
        rows.append({
            "Title": t,
            "Familiarity": round(fam, 1),
            "Motivation": round(mot, 1),
            "GoogleTrends": round(trends_scores[i], 1),
            "WikipediaN": round(wiki_n[i], 1),
            "YouTubeN": round(yt_combo[i], 1),
            "SpotifyN": round(sp_n[i], 1),
        })

    df = pd.DataFrame(rows)
    # Sort AFTER potential normalization (done later)
    return df

def apply_benchmark(df: pd.DataFrame, benchmark_title: str) -> pd.DataFrame:
    df = df.copy()
    if benchmark_title not in df["Title"].values:
        return df
    b = df[df["Title"] == benchmark_title].iloc[0]
    cols = ["Familiarity","Motivation","GoogleTrends","WikipediaN","YouTubeN","SpotifyN"]
    for col in cols:
        series = df[col].astype(float).tolist()
        if not has_variance(series):
            continue  # skip flat columns
        bench = float(b[col])
        if abs(bench) > EPS:
            df[col] = (df[col].astype(float) / bench) * 100.0
    return df

# -------------------------
# PLOTTING & PDF
# -------------------------
def quadrant_plot(df: pd.DataFrame, title: str = "Familiarity vs Motivation"):
    fig = plt.figure()
    x, y = df["Familiarity"].values, df["Motivation"].values
    plt.scatter(x, y)
    for _, r in df.iterrows():
        plt.annotate(r["Title"], (r["Familiarity"], r["Motivation"]),
                     fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.axvline(np.median(x), linestyle="--")
    plt.axhline(np.median(y), linestyle="--")
    plt.xlabel("Familiarity"); plt.ylabel("Motivation"); plt.title(title)
    return fig

def bar_chart(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    order = df.sort_values(by=col, ascending=True)
    plt.barh(order["Title"], order[col])
    plt.title(title); plt.xlabel(col)
    return fig

def generate_pdf_brief(df: pd.DataFrame, file_path: str):
    with PdfPages(file_path) as pdf:
        # Page 1: table
        fig1 = plt.figure(); plt.axis('off'); plt.title("Alberta Ballet — Title Scores", pad=20)
        tab = plt.table(cellText=df.round(1).values, colLabels=df.columns, loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1, 1.2)
        pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)
        # Quadrant
        fig2 = quadrant_plot(df, "Familiarity vs Motivation (Quadrant Map)")
        pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
        # Bars
        fig3 = bar_chart(df, "Familiarity", "Familiarity by Title"); pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)
        fig4 = bar_chart(df, "Motivation", "Motivation by Title");   pdf.savefig(fig4, bbox_inches='tight'); plt.close(fig4)
        # Notes
        fig5 = plt.figure(); plt.axis('off')
        memo = ("Notes:\n"
                "- Signals: Google Trends (AB) with optional global fallback; Wikipedia; optional YouTube & Spotify.\n"
                "- Benchmark normalization scales variable columns so benchmark = 100.\n"
                "- Diagnostics in app show metric dispersion (min/max/std).")
        plt.text(0.01, 0.99, memo, va='top')
        pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Viability", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer")

st.markdown("Paste titles, score them, optionally normalize to a benchmark, and export a PDF brief.")

with st.expander("🔑 API Configuration (optional)"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")

with st.expander("⚙️ Options"):
    use_global_trends = st.checkbox("Use global Google Trends fallback if Alberta is flat", value=True)

default_titles = [
    "The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio",
    "The Merry Widow","The Hunchback of Notre Dame","Frozen",
    "Beauty and the Beast","Alice in Wonderland","Peter Pan"
]
titles_input = st.text_area("Enter titles (one per line):", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

colA, colB = st.columns(2)
with colA:
    do_benchmark = st.checkbox("Normalize to a benchmark title?")
with colB:
    benchmark_title = st.selectbox("Benchmark title", options=titles, index=0 if "The Nutcracker" in titles else 0)

if st.button("Score Titles"):
    with st.spinner("Scoring titles…"):
        df = score_titles(titles, yt_key.strip() or None, sp_id.strip() or None, sp_secret.strip() or None, use_global_trends)
        if do_benchmark and benchmark_title:
            df = apply_benchmark(df, benchmark_title)

        # Sort for display AFTER normalization decision
        df = df.sort_values(by=["Motivation","Familiarity"], ascending=False).reset_index(drop=True)

        # Diagnostics
        st.subheader("Diagnostics")
        diag_cols = ["Familiarity","Motivation","GoogleTrends","WikipediaN","YouTubeN","SpotifyN"]
        diag = pd.DataFrame({
            "metric": diag_cols,
            "min":    [float(df[c].min()) for c in diag_cols],
            "max":    [float(df[c].max()) for c in diag_cols],
            "std":    [float(df[c].std(ddof=0)) for c in diag_cols],
        })
        st.dataframe(diag, use_container_width=True)

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="title_scores.csv", mime="text/csv")

        # PDF brief
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"title_scores_brief_{ts}.pdf"
        if st.button("📄 Generate PDF Brief"):
            generate_pdf_brief(df, pdf_path)
            st.success("PDF created.")
            st.download_button("⬇️ Download PDF Brief",
                               data=open(pdf_path, "rb").read(),
                               file_name=pdf_path, mime="application/pdf")
