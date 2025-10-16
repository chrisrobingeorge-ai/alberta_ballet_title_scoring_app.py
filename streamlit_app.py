# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v4)
# - Robust Familiarity (log Wikipedia pageviews -> percentile rank) so scores don't collapse
# - Optional Google Trends, YouTube, Spotify inputs
# - Smarter benchmark normalization (skips flat columns)
# - Diagnostics table + PDF export

import os, math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
TRENDS_TIMEFRAME = "today 5-y"

# Weights for the composite indices
WEIGHTS = {
    "fam_wiki": 0.55,        # Robust Wikipedia familiarity (log -> percentile)
    "fam_trends": 0.30,      # Google Trends familiarity (if enabled)
    "fam_spotify": 0.15,     # Spotify recognition (if enabled)

    "mot_youtube": 0.45,     # YouTube search/view proxy (if enabled)
    "mot_trends": 0.25,      # Trends supports motivation if enabled
    "mot_spotify": 0.15,     # Music recall
    "mot_wiki": 0.15,        # Wikipedia salience assists motivation (lighter)
}

# -------------------------
# UTILITIES
# -------------------------
def normalize_series(values):
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [50.0 for _ in values]
    return [(v - vmin) * 100.0 / (vmax - vmin) for v in values]

def capped_mean(values):
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    k = max(1, int(0.05 * n))
    trimmed = arr[k: n-k] if n > 2*k else arr
    return float(sum(trimmed)) / len(trimmed)

def percentile_normalize(vals):
    """Turn a list into 0..100 by percentile rank; robust when ranges are tight."""
    if not vals:
        return []
    s = pd.Series(vals, dtype="float64")
    ranks = s.rank(method="average", pct=True)
    return (ranks * 100.0).tolist()

# -------------------------
# GOOGLE TRENDS (optional)
# -------------------------
def fetch_google_trends_score(title: str, region_geo: str = "CA-AB") -> float:
    """Uses pytrends only if available; returns 0.0 on error or rate-limit."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360)
        variants = [title, f"{title} ballet", f"{title} story"]
        scores = []
        for kw in variants:
            try:
                pytrends.build_payload([kw], geo=region_geo, timeframe=TRENDS_TIMEFRAME)
                df = pytrends.interest_over_time()
                if not df.empty and kw in df.columns:
                    scores.append(capped_mean(df[kw].astype(float).tolist()))
            except Exception:
                continue
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0

# -------------------------
# WIKIPEDIA (robust search + pageviews)
# -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"

def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        r = requests.get(WIKI_API, params=params, timeout=10)
        if r.status_code != 200:
            return None
        items = r.json().get("query", {}).get("search", [])
        if not items:
            return None
        for it in items:
            t = it.get("title", "")
            if t.lower() == query.lower() or query.lower() in t.lower():
                return t
        return items[0].get("title")
    except Exception:
        return None

def fetch_wikipedia_views_for_page(page_title: str) -> float:
    try:
        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        page = page_title.replace(" ", "_")
        url = WIKI_PAGEVIEW.format(page=page, start=start, end=end)
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return 0.0
        items = r.json().get("items", [])
        views = [it.get("views", 0) for it in items]
        return (sum(views) / 365.0) if views else 0.0
    except Exception:
        return 0.0

def fetch_wikipedia_views(query: str) -> Tuple[str, float]:
    candidates = [
        query,
        f"{query} (ballet)",
        f"{query} (novel)",
        f"{query} (film)",
        f"{query} (opera)",
        f"{query} (play)",
    ]
    best = wiki_search_best_title(query)
    if best:
        v = fetch_wikipedia_views_for_page(best)
        if v > 0:
            return best, v
    best_title, best_val = None, 0.0
    for c in candidates:
        t = wiki_search_best_title(c) or c
        v = fetch_wikipedia_views_for_page(t)
        if v > best_val:
            best_title, best_val = t, v
    return (best_title or query), best_val

# -------------------------
# YOUTUBE / SPOTIFY (optional)
# -------------------------
def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float]:
    """(search_count, sqrt(top_view_count)) — returns 0,0 without key."""
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
                if vc > top_views:
                    top_views = vc
        return float(search_count), float(math.sqrt(top_views))
    except Exception:
        return 0.0, 0.0

def fetch_spotify_popularity(title: str, client_id: Optional[str], client_secret: Optional[str]) -> float:
    if not client_id or not client_secret or spotipy is None:
        return 0.0
    try:
        auth_mgr = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_mgr)
        results = sp.search(q=title, type="track,album", limit=10)
        pops = [item.get("popularity", 0) for item in results.get("tracks", {}).get("items", [])]
        return max(pops) if pops else 0.0
    except Exception:
        return 0.0

# -------------------------
# SCORING
# -------------------------
def score_titles(titles, yt_key, sp_id, sp_secret, use_trends, trends_region):
    trends_scores, wiki_titles, wiki_scores, yt_search_counts, yt_top_view_sqrts, sp_pops = [], [], [], [], [], []

    for t in titles:
        st.write(f"Fetching metrics for **{t}** …")
        # Trends (optional)
        ts = fetch_google_trends_score(t, region_geo=trends_region) if use_trends else 0.0
        trends_scores.append(ts)

        # Wikipedia (robust)
        w_title, w_score = fetch_wikipedia_views(t)
        wiki_titles.append(w_title)
        wiki_scores.append(w_score)

        # YouTube / Spotify (optional)
        ysc, ytv = fetch_youtube_metrics(t, yt_key)
        sp = fetch_spotify_popularity(t, sp_id, sp_secret)

        yt_search_counts.append(ysc)
        yt_top_view_sqrts.append(ytv)
        sp_pops.append(sp)

    # ---- Robust Familiarity base from Wikipedia:
    # Use raw pageviews -> log transform -> percentile rank (0..100)
    wiki_log = [math.log1p(v) for v in wiki_scores]
    wiki_fam = percentile_normalize(wiki_log)  # robust 0..100 familiarity signal

    # Other sources normalized 0..100 for blending
    trends_n   = trends_scores if use_trends else [0.0] * len(titles)  # already 0..100 if present
    sp_n       = normalize_series(sp_pops)
    yt_searchn = normalize_series(yt_search_counts)
    yt_topn    = normalize_series(yt_top_view_sqrts)
    yt_combo   = [capped_mean([a, b]) for a, b in zip(yt_searchn, yt_topn)]  # 0..100 motivation proxy

    # Compose Familiarity & Motivation
    rows = []
    for i, t in enumerate(titles):
        # Familiarity = weighted blend (robust wiki, + optional trends/spotify)
        fam = (
            WEIGHTS["fam_wiki"]   * (wiki_fam[i] or 0.0) +
            WEIGHTS["fam_trends"] * (trends_n[i] or 0.0) +
            WEIGHTS["fam_spotify"]* (sp_n[i] or 0.0)
        )

        # Motivation = weighted blend (youtube + trends + spotify + light wiki)
        mot = (
            WEIGHTS["mot_youtube"]* (yt_combo[i] or 0.0) +
            WEIGHTS["mot_trends"] * (trends_n[i] or 0.0) +
            WEIGHTS["mot_spotify"]* (sp_n[i] or 0.0) +
            WEIGHTS["mot_wiki"]   * (wiki_fam[i] or 0.0)
        )

        rows.append({
            "Title": t,
            "ResolvedWikiPage": wiki_titles[i],
            "Familiarity": round(fam, 1),
            "Motivation": round(mot, 1),
            "GoogleTrends": round(trends_n[i], 1),
            "WikipediaRawAvgDaily": round(wiki_scores[i], 2),
            "WikipediaFamiliarity": round(wiki_fam[i], 1),
            "YouTubeN": round(yt_combo[i], 1),
            "SpotifyN": round(sp_n[i], 1),
        })

    df = pd.DataFrame(rows).sort_values(by=["Motivation", "Familiarity"], ascending=False)
    return df

def apply_benchmark(df: pd.DataFrame, benchmark_title: str) -> pd.DataFrame:
    df = df.copy()
    if benchmark_title not in df["Title"].values:
        return df
    b_row = df[df["Title"] == benchmark_title].iloc[0]
    cols = ["Familiarity","Motivation","GoogleTrends","WikipediaFamiliarity","YouTubeN","SpotifyN"]
    for col in cols:
        series = df[col].astype(float)
        if series.std(ddof=0) < 1e-6:
            continue
        bench = float(b_row[col])
        if bench and abs(bench) > 1e-9:
            df[col] = (series / bench) * 100.0
    return df

# -------------------------
# PLOTTING & PDF
# -------------------------
def quadrant_plot(df: pd.DataFrame, title: str = "Familiarity vs Motivation"):
    fig = plt.figure()
    x = df["Familiarity"].values
    y = df["Motivation"].values
    plt.scatter(x, y)
    for _, r in df.iterrows():
        plt.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.axvline(np.median(x), linestyle="--")
    plt.axhline(np.median(y), linestyle="--")
    plt.xlabel("Familiarity")
    plt.ylabel("Motivation")
    plt.title(title)
    return fig

def bar_chart(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    order = df.sort_values(by=col, ascending=True)
    plt.barh(order["Title"], order[col])
    plt.title(title)
    plt.xlabel(col)
    return fig

def generate_pdf_brief(df: pd.DataFrame, file_path: str):
    with PdfPages(file_path) as pdf:
        fig1 = plt.figure(); plt.axis('off')
        plt.title("Alberta Ballet — Title Scores", pad=20)
        tab = plt.table(cellText=df.round(1).values, colLabels=df.columns, loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1, 1.2)
        pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)

        fig2 = quadrant_plot(df, "Familiarity vs Motivation (Quadrant Map)")
        pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

        fig3 = bar_chart(df, "Familiarity", "Familiarity by Title")
        pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)

        fig4 = bar_chart(df, "Motivation", "Motivation by Title")
        pdf.savefig(fig4, bbox_inches='tight'); plt.close(fig4)

        fig5 = plt.figure(); plt.axis('off')
        memo = (
            "Familiarity = log Wikipedia pageviews -> percentile (robust), blended with optional Trends/Spotify.\n"
            "Motivation = YouTube + Trends + Spotify + light Wikipedia.\n"
            "Benchmark normalization scales varying columns so the benchmark = 100."
        )
        plt.text(0.01, 0.99, memo, va='top')
        pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Viability", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v4)")

st.markdown("Robust Familiarity via Wikipedia (log → percentile). Optional Trends/YouTube/Spotify enhance separation.")

with st.expander("🔑 API Configuration (optional)"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")

with st.expander("⚙️ Options"):
    use_trends = st.checkbox("Use Google Trends (may be blocked on Cloud)", value=False)
    trends_region = st.selectbox("Trends region", ["CA-AB","CA",""], index=0)
    st.caption("Tip: start with Trends OFF to confirm Familiarity varies. Then experiment with AB/CA/global.")

default_titles = [
    "The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio",
    "The Merry Widow","The Hunchback of Notre Dame","Frozen",
    "Beauty and the Beast","Alice in Wonderland","Peter Pan"
]
titles_input = st.text_area("Enter titles (one per line):", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

colA, colB, colC = st.columns(3)
with colA:
    do_benchmark = st.checkbox("Normalize to a benchmark title?")
with colB:
    benchmark_title = st.selectbox("Benchmark title", options=titles, index=0 if "The Nutcracker" in titles else 0)
with colC:
    run_btn = st.button("Score Titles", type="primary")

if run_btn:
    with st.spinner("Scoring titles…"):
        df = score_titles(titles, yt_key.strip() or None, sp_id.strip() or None, sp_secret.strip() or None, use_trends, trends_region)
        if do_benchmark and benchmark_title:
            df = apply_benchmark(df, benchmark_title)
        st.success("Done.")
        st.dataframe(df, use_container_width=True)

        # Diagnostics — raw sources and dispersion
        st.subheader("Diagnostics")
        diag_cols = ["Familiarity","Motivation","GoogleTrends","WikipediaRawAvgDaily","WikipediaFamiliarity","YouTubeN","SpotifyN"]
        diag = pd.DataFrame({
            "metric": diag_cols,
            "min":    [float(df[c].min()) for c in diag_cols],
            "max":    [float(df[c].max()) for c in diag_cols],
            "std":    [float(df[c].std(ddof=0)) for c in diag_cols],
        })
        st.dataframe(diag, use_container_width=True)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "title_scores.csv", "text/csv")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"title_scores_brief_{ts}.pdf"
        if st.button("📄 Generate PDF Brief"):
            generate_pdf_brief(df, pdf_path)
            st.success("PDF created.")
            st.download_button("⬇️ Download PDF Brief", data=open(pdf_path, "rb").read(),
                               file_name=pdf_path, mime="application/pdf")
