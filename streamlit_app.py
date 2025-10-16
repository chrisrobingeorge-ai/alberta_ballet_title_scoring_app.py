# --- Alberta Ballet — Title Familiarity & Motivation Scorer (Streamlit) ---
# Mobile-friendly single-file version for Streamlit Cloud
# Signals: Google Trends (AB), Wikipedia pageviews; optional YouTube + Spotify if keys provided
# Includes: Benchmark normalization, PDF brief, quadrant map

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

REGION_GEO = "CA-AB"              # Alberta
TRENDS_TIMEFRAME = "today 5-y"    # last 5 years

WEIGHTS = {
    "search_trends": 0.35,
    "youtube": 0.25,
    "social": 0.00,      # placeholder for future integrations
    "spotify": 0.10,
    "wikipedia": 0.05,
    "safety_buffer": 0.25
}

def normalize_series(values: List[float]) -> List[float]:
    if not values: return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin: return [50.0 for _ in values]
    return [(v - vmin) * 100.0 / (vmax - vmin) for v in values]

def capped_mean(values: List[float]) -> float:
    if not values: return 0.0
    arr = sorted(values); n = len(arr); k = max(1, int(0.05 * n))
    trimmed = arr[k:n-k] if n > 2*k else arr
    return float(sum(trimmed)) / len(trimmed)

# ----------- data sources -----------
def fetch_google_trends_score(title: str) -> float:
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(kw_list=[title], geo=REGION_GEO, timeframe=TRENDS_TIMEFRAME)
        df = pytrends.interest_over_time()
        if df.empty: return 0.0
        return capped_mean(df[title].astype(float).tolist())
    except Exception:
        return 0.0

def fetch_wikipedia_views(title: str) -> float:
    try:
        page = title.replace(" ", "_")
        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return 0.0
        views = [it.get("views", 0) for it in r.json().get("items", [])]
        return float(sum(views)) / 365.0 if views else 0.0
    except Exception:
        return 0.0

def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float]:
    if not api_key or build is None: return 0.0, 0.0
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        search = yt.search().list(q=title, part="id", type="video", maxResults=50).execute()
        ids = [it["id"]["videoId"] for it in search.get("items", [])]
        search_count = len(ids)
        if not ids: return 0.0, 0.0
        chunks = [ids[i:i+50] for i in range(0, len(ids), 50)]
        top_views = 0
        for ch in chunks:
            stats = yt.videos().list(part="statistics", id=",".join(ch)).execute()
            for it in stats.get("items", []):
                vc = int(it.get("statistics", {}).get("viewCount", 0))
                top_views = max(top_views, vc)
        return float(search_count), math.sqrt(top_views)
    except Exception:
        return 0.0, 0.0

def fetch_spotify_popularity(title: str, client_id: Optional[str], client_secret: Optional[str]) -> float:
    if not client_id or not client_secret or spotipy is None: return 0.0
    try:
        auth_mgr = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_mgr)
        results = sp.search(q=title, type="track,album", limit=10)
        pops = [it.get("popularity", 0) for it in results.get("tracks", {}).get("items", [])]
        return max(pops) if pops else 0.0
    except Exception:
        return 0.0

def weighted_sum(components: Dict[str, float]) -> float:
    pairs, total_w = [], 0.0
    for k, v in components.items():
        w = {"trends":WEIGHTS["search_trends"], "youtube":WEIGHTS["youtube"], "spotify":WEIGHTS["spotify"],
             "wikipedia":WEIGHTS["wikipedia"], "social":WEIGHTS["social"]}.get(k, 0.0)
        pairs.append((v, w)); total_w += w
    if total_w <= 0: return 0.0
    extra = WEIGHTS["safety_buffer"] * (len(pairs) / 3.0)
    adj = total_w + extra
    return sum(v * (w / adj) for v, w in pairs)

def score_titles(titles: List[str], yt_key: Optional[str], sp_id: Optional[str], sp_secret: Optional[str]) -> pd.DataFrame:
    trends, wiki, yt_search, yt_top, sp = [], [], [], [], []
    for t in titles:
        st.write(f"Fetching metrics for **{t}** …")
        trends.append(fetch_google_trends_score(t))
        wiki.append(fetch_wikipedia_views(t))
        s, v = fetch_youtube_metrics(t, yt_key); yt_search.append(s); yt_top.append(v)
        sp.append(fetch_spotify_popularity(t, sp_id, sp_secret))
    wiki_n = normalize_series(wiki)
    yt_search_n = normalize_series(yt_search)
    yt_top_n = normalize_series(yt_top)
    yt_combo = [capped_mean([a,b]) for a,b in zip(yt_search_n, yt_top_n)]
    sp_n = normalize_series(sp)

    rows = []
    for i, t in enumerate(titles):
        fam = weighted_sum({"trends":trends[i], "wikipedia":wiki_n[i], "spotify":sp_n[i]})
        mot = weighted_sum({"youtube":yt_combo[i], "trends":trends[i], "spotify":sp_n[i]})
        rows.append({
            "Title": t,
            "Familiarity": round(fam*100, 1),
            "Motivation": round(mot*100, 1),
            "GoogleTrends": round(trends[i], 1),
            "WikipediaN": round(wiki_n[i], 1),
            "YouTubeN": round(yt_combo[i], 1),
            "SpotifyN": round(sp_n[i], 1)
        })
    return pd.DataFrame(rows).sort_values(by=["Motivation","Familiarity"], ascending=False)

def apply_benchmark(df: pd.DataFrame, bench: str) -> pd.DataFrame:
    if bench not in df["Title"].values: return df
    b = df.loc[df["Title"]==bench].iloc[0]
    for col in ["Familiarity","Motivation","GoogleTrends","WikipediaN","YouTubeN","SpotifyN"]:
        if b[col] and b[col] != 0: df[col] = (df[col]/b[col])*100.0
    return df

def quadrant_plot(df: pd.DataFrame, ttl="Familiarity vs Motivation"):
    fig = plt.figure()
    x, y = df["Familiarity"].values, df["Motivation"].values
    plt.scatter(x,y)
    for _, r in df.iterrows():
        plt.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.axvline(np.median(x), linestyle="--")
    plt.axhline(np.median(y), linestyle="--")
    plt.xlabel("Familiarity"); plt.ylabel("Motivation"); plt.title(ttl)
    return fig

def bar_chart(df: pd.DataFrame, col: str, ttl: str):
    fig = plt.figure()
    order = df.sort_values(by=col)
    plt.barh(order["Title"], order[col]); plt.title(ttl); plt.xlabel(col)
    return fig

def generate_pdf(df: pd.DataFrame, path: str):
    with PdfPages(path) as pdf:
        fig1 = plt.figure(); plt.axis('off'); plt.title("Alberta Ballet — Title Scores", pad=20)
        tab = plt.table(cellText=df.round(1).values, colLabels=df.columns, loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.2)
        pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)
        pdf.savefig(quadrant_plot(df), bbox_inches='tight'); plt.close()
        pdf.savefig(bar_chart(df, "Familiarity", "Familiarity by Title"), bbox_inches='tight'); plt.close()
        pdf.savefig(bar_chart(df, "Motivation", "Motivation by Title"), bbox_inches='tight'); plt.close()
        fig5 = plt.figure(); plt.axis('off'); memo = "Signals: Google Trends (AB), Wikipedia; optional YouTube/Spotify via keys. If normalized, benchmark=100."
        plt.text(0.01, 0.99, memo, va='top'); pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# ------------- UI -------------
st.set_page_config(page_title="Alberta Ballet — Title Scorer", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer")

with st.expander("🔑 API keys (optional)"):
    yt_key   = st.text_input("YouTube Data API key", type="password")
    sp_id    = st.text_input("Spotify Client ID", type="password")
    sp_sec   = st.text_input("Spotify Client Secret", type="password")

default_titles = ["The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio","The Merry Widow",
                  "The Hunchback of Notre Dame","Frozen","Beauty and the Beast","Alice in Wonderland","Peter Pan"]
titles = st.text_area("Titles (one per line)", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles.splitlines() if t.strip()]

c1, c2 = st.columns(2)
with c1: do_norm = st.checkbox("Normalize to a benchmark?")
with c2: bench = st.selectbox("Benchmark title", options=titles, index=0 if "The Nutcracker" in titles else 0)

if st.button("Score Titles"):
    with st.spinner("Scoring…"):
        df = score_titles(titles, yt_key.strip() or None, sp_id.strip() or None, sp_sec.strip() or None)
        if do_norm and bench: df = apply_benchmark(df, bench)
        st.success("Done.")
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="title_scores.csv", mime="text/csv")

        if st.button("📄 Generate PDF Brief"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = f"title_scores_brief_{ts}.pdf"
            generate_pdf(df, pdf_path)
            st.download_button("⬇️ Download PDF", open(pdf_path,"rb").read(), file_name=pdf_path, mime="application/pdf")
