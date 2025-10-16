# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v7.2)
# Adds: API Self-Test, hardened Wikipedia requests, clearer YouTube/Trends diagnostics, caching

import os, math, time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests
from requests.adapters import HTTPAdapter, Retry

# Optional APIs
try:
    from googleapiclient.discovery import build  # YouTube Data API v3
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception

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

WEIGHTS = {
    "fam_wiki": 0.55,
    "fam_trends": 0.30,
    "fam_spotify": 0.15,

    "mot_youtube": 0.45,
    "mot_trends": 0.25,
    "mot_spotify": 0.15,
    "mot_wiki": 0.15,
}

SEGMENT_RULES = {
    "Core Classical (Female 35–64)": {
        "fam": {"female_lead": +0.12, "romantic": +0.08, "classic_canon": +0.10,
                "male_lead": -0.04, "contemporary": -0.05, "tragic": -0.02},
        "mot": {"female_lead": +0.10, "romantic": +0.06, "classic_canon": +0.08,
                "male_lead": -0.03, "contemporary": -0.04, "spectacle": +0.02}
    },
    "Family (Parents w/ kids)": {
        "fam": {"female_lead": +0.15, "family_friendly": +0.20, "pop_ip": +0.10,
                "male_lead": -0.05, "tragic": -0.12},
        "mot": {"female_lead": +0.10, "family_friendly": +0.18, "spectacle": +0.06,
                "pop_ip": +0.10, "tragic": -0.15}
    },
    "Emerging Adults (18–34)": {
        "fam": {"contemporary": +0.18, "spectacle": +0.08, "pop_ip": +0.10,
                "classic_canon": -0.04},
        "mot": {"contemporary": +0.22, "spectacle": +0.10, "pop_ip": +0.10,
                "female_lead": +0.02, "male_lead": +0.02}
    }
}

ATTR_COLUMNS = [
    "female_lead","male_lead","family_friendly","romantic","tragic",
    "contemporary","classic_canon","spectacle","pop_ip"
]

# -------------------------
# UTILITIES / SESSION SETUP
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
    if not vals:
        return []
    s = pd.Series(vals, dtype="float64")
    ranks = s.rank(method="average", pct=True)
    return (ranks * 100.0).tolist()

def requests_session():
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=frozenset(["GET", "POST"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "AlbertaBallet-TitleScorer/1.0 (contact: info@albertaballet.ca)"
    })
    return s

# -------------------------
# GOOGLE TRENDS (province/city with retries & fallbacks)
# -------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _pytrends():
    from pytrends.request import TrendReq
    # retries/backoff inside TrendReq
    return TrendReq(hl="en-US", tz=360, retries=5, backoff_factor=0.5)

def _trends_kw_variants(title: str) -> List[str]:
    return [title, f"{title} ballet", f"{title} story"]

def fetch_trends_province_or_global(title: str, region_geo: str) -> Tuple[float, str]:
    """
    Returns (score, source_label) — source_label tells which region worked.
    """
    try:
        pytrends = _pytrends()
        def _get_score(geo: str) -> float:
            vals = []
            for kw in _trends_kw_variants(title):
                try:
                    pytrends.build_payload([kw], geo=geo, timeframe=TRENDS_TIMEFRAME)
                    df = pytrends.interest_over_time()
                    if not df.empty and kw in df.columns:
                        vals.append(float(df[kw].mean()))
                    time.sleep(0.2)  # be gentle
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else 0.0

        score = _get_score(region_geo)
        if score > 0:
            return score, region_geo
        if region_geo != "CA":
            score = _get_score("CA")
            if score > 0:
                return score, "CA"
        score = _get_score("")
        if score > 0:
            return score, "GLOBAL"
        return 0.0, "NONE"
    except Exception as e:
        return 0.0, f"ERR:{e}"

def fetch_trends_city(title: str, city_name: str) -> Tuple[float, str]:
    """
    City-level via interest_by_region(resolution='CITY'). Falls back to AB/CA/Global.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = _pytrends()

        def _city_score(kw: str) -> float:
            try:
                pytrends.build_payload([kw], geo="CA", timeframe=TRENDS_TIMEFRAME)
                df = pytrends.interest_by_region(resolution="CITY", inc_low_vol=True)
                if df is None or df.empty or kw not in df.columns:
                    return 0.0
                idx = df.index.astype(str)
                mask_city = idx.str.contains(city_name, case=False, na=False)
                mask_ab = idx.str.contains(r"\(AB\)", case=False, na=False)
                sel = df.loc[mask_city & mask_ab, kw]
                if sel.empty:
                    sel = df.loc[mask_city, kw]
                if sel.empty:
                    return 0.0
                return float(sel.mean())
            except Exception:
                return 0.0

        vals = []
        for kw in _trends_kw_variants(title):
            v = _city_score(kw)
            if v > 0:
                vals.append(v); time.sleep(0.2)
        if vals:
            return float(np.mean(vals)), f"CITY:{city_name}"
        score, src = fetch_trends_province_or_global(title, "CA-AB")
        return score, src
    except Exception as e:
        return 0.0, f"ERR:{e}"

def fetch_google_trends_score(title: str, market: str) -> Tuple[float, str]:
    if market == "Calgary":
        s, src = fetch_trends_city(title, "Calgary")
        if s > 0:
            return s, src
        return fetch_trends_province_or_global(title, "CA-AB")
    if market == "Edmonton":
        s, src = fetch_trends_city(title, "Edmonton")
        if s > 0:
            return s, src
        return fetch_trends_province_or_global(title, "CA-AB")
    return fetch_trends_province_or_global(title, "CA-AB")

# -------------------------
# WIKIPEDIA (robust search + pageviews) + caching + UA
# -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"

@st.cache_data(show_spinner=False, ttl=86400)
def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        s = requests_session()
        params = {"action": "query","list": "search","srsearch": query,"format": "json","srlimit": 5}
        r = s.get(WIKI_API, params=params, timeout=10)
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

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_wikipedia_views_for_page(page_title: str) -> float:
    try:
        s = requests_session()
        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        page = page_title.replace(" ", "_")
        url = WIKI_PAGEVIEW.format(page=page, start=start, end=end)
        r = s.get(url, timeout=12)
        if r.status_code != 200:
            return 0.0
        items = r.json().get("items", [])
        views = [it.get("views", 0) for it in items]
        return (sum(views) / 365.0) if views else 0.0
    except Exception:
        return 0.0

def fetch_wikipedia_views(query: str) -> Tuple[str, float]:
    candidates = [query, f"{query} (ballet)", f"{query} (novel)", f"{query} (film)", f"{query} (opera)", f"{query} (play)"]
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
# YOUTUBE / SPOTIFY (optional) + caching
# -------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float, str]:
    """(search_count, sqrt(top_view_count), status_msg) — returns zeros + reason if key/quota missing."""
    if not api_key:
        return 0.0, 0.0, "NO_KEY"
    if build is None:
        return 0.0, 0.0, "LIB_MISSING"
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        search = yt.search().list(q=title, part="id", type="video", maxResults=50).execute()
        ids = [item["id"]["videoId"] for item in search.get("items", [])]
        search_count = len(ids)
        if not ids:
            return 0.0, 0.0, "NO_RESULTS"
        chunks = [ids[i:i+50] for i in range(0, len(ids), 50)]
        top_views = 0
        for ch in chunks:
            stats = yt.videos().list(part="statistics", id=",".join(ch)).execute()
            for it in stats.get("items", []):
                vc = int(it.get("statistics", {}).get("viewCount", 0))
                if vc > top_views:
                    top_views = vc
        return float(search_count), float(math.sqrt(top_views)), "OK"
    except HttpError as he:
        return 0.0, 0.0, f"HTTP_ERROR:{he}"
    except Exception as e:
        return 0.0, 0.0, f"ERR:{e}"

@st.cache_data(show_spinner=False, ttl=86400)
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
# SEGMENT TAGGING & BOOSTS
# -------------------------
def infer_title_attributes(title: str) -> Dict[str, bool]:
    t = title.lower()
    attr = {k: False for k in ATTR_COLUMNS}
    female_titles = ["nutcracker","sleeping beauty","cinderella","beauty and the beast","alice","giselle","swan lake","merry widow","romeo and juliet"]
    male_titles = ["pinocchio","peter pan","don quixote","hunchback","notre dame","romeo and juliet"]
    family_list = ["nutcracker","cinderella","beauty and the beast","alice","frozen","peter pan","pinocchio","wizard of oz"]
    romantic_list = ["swan lake","cinderella","sleeping beauty","romeo and juliet","merry widow","beauty and the beast","giselle"]
    tragic_list = ["swan lake","romeo and juliet","hunchback","notre dame","giselle"]
    classic_list = ["nutcracker","swan lake","sleeping beauty","cinderella","romeo and juliet","don quixote","giselle","merry widow"]
    spectacle_list = ["wizard of oz","peter pan","pinocchio","frozen","notre dame","hunchback","don quixote"]
    popip_list = ["frozen","beauty and the beast","wizard of oz","bridgerton","harry potter","star wars","avengers","barbie"]

    if any(k in t for k in female_titles): attr["female_lead"] = True
    if any(k in t for k in male_titles):   attr["male_lead"] = True
    if any(k in t for k in family_list):   attr["family_friendly"] = True
    if any(k in t for k in romantic_list): attr["romantic"] = True
    if any(k in t for k in tragic_list):   attr["tragic"] = True
    if any(k in t for k in classic_list):  attr["classic_canon"] = True
    if any(k in t for k in spectacle_list):attr["spectacle"] = True
    if any(k in t for k in popip_list):    attr["pop_ip"] = True
    if any(k in t for k in ["contemporary","composers","mixed bill","forsythe","balanchine","jerome robbins","nijinsky","grimm","1000 tales","once upon a time"]):
        attr["contemporary"] = True
    if "romeo" in t:
        attr["female_lead"] = True
        attr["male_lead"] = True
    return attr

def segment_boosts(attrs: Dict[str,bool], segment_name: str) -> Tuple[float,float]:
    rules = SEGMENT_RULES.get(segment_name, {})
    fam_delta = 0.0
    mot_delta = 0.0
    for k, v in rules.get("fam", {}).items():
        if attrs.get(k, False): fam_delta += v
    for k, v in rules.get("mot", {}).items():
        if attrs.get(k, False): mot_delta += v
    fam_delta = max(-0.30, min(0.40, fam_delta))
    mot_delta = max(-0.30, min(0.40, mot_delta))
    return fam_delta, mot_delta

# -------------------------
# SCORING
# -------------------------
def score_titles(titles, yt_key, sp_id, sp_secret, use_trends, market, segment_name, attrs_df):
    user_attrs = {}
    for _, row in attrs_df.iterrows():
        nm = str(row["Title"]).strip()
        if not nm: continue
        user_attrs[nm.lower()] = {col: bool(row.get(col, False)) for col in ATTR_COLUMNS}

    trends_scores, trend_srcs, wiki_titles, wiki_scores, yt_search_counts, yt_top_view_sqrts, yt_status, sp_pops = [], [], [], [], [], [], [], []
    fam_boosts, mot_boosts = [], []

    for t in titles:
        st.write(f"Fetching metrics for **{t}** …")

        # Trends
        if use_trends:
            ts, src = fetch_google_trends_score(t, market=market)
        else:
            ts, src = 0.0, "OFF"
        trends_scores.append(ts); trend_srcs.append(src)

        # Wikipedia
        w_title, w_score = fetch_wikipedia_views(t)
        wiki_titles.append(w_title); wiki_scores.append(w_score)

        # YouTube / Spotify
        ysc, ytv, ystat = fetch_youtube_metrics(t, yt_key)
        yt_search_counts.append(ysc); yt_top_view_sqrts.append(ytv); yt_status.append(ystat)
        sp = fetch_spotify_popularity(t, sp_id, sp_secret)
        sp_pops.append(sp)

        # Segment boost
        base_attrs = infer_title_attributes(t)
        over = user_attrs.get(t.lower(), {})
        base_attrs.update(over)
        fdelta, mdelta = segment_boosts(base_attrs, segment_name)
        fam_boosts.append(fdelta); mot_boosts.append(mdelta)

    wiki_log = [math.log1p(v) for v in wiki_scores]
    wiki_fam = percentile_normalize(wiki_log)  # 0..100

    trends_n   = trends_scores if use_trends else [0.0] * len(titles)
    sp_n       = normalize_series(sp_pops)
    yt_searchn = normalize_series(yt_search_counts)
    yt_topn    = normalize_series(yt_top_view_sqrts)
    yt_combo   = [capped_mean([a, b]) for a, b in zip(yt_searchn, yt_topn)]

    rows = []
    for i, t in enumerate(titles):
        fam = (
            WEIGHTS["fam_wiki"]   * (wiki_fam[i] or 0.0) +
            WEIGHTS["fam_trends"] * (trends_n[i] or 0.0) +
            WEIGHTS["fam_spotify"]* (sp_n[i] or 0.0)
        )
        mot = (
            WEIGHTS["mot_youtube"]* (yt_combo[i] or 0.0) +
            WEIGHTS["mot_trends"] * (trends_n[i] or 0.0) +
            WEIGHTS["mot_spotify"]* (sp_n[i] or 0.0) +
            WEIGHTS["mot_wiki"]   * (wiki_fam[i] or 0.0)
        )

        fam_adj = fam * (1.0 + fam_boosts[i])
        mot_adj = mot * (1.0 + mot_boosts[i])

        rows.append({
            "Title": t,
            "ResolvedWikiPage": wiki_titles[i],
            "Market": market,
            "TargetSegment": segment_name,
            "Familiarity": round(fam, 1),
            "Motivation": round(mot, 1),
            "FamiliarityAdj": round(fam_adj, 1),
            "MotivationAdj": round(mot_adj, 1),
            "SegBoostF%": round(fam_boosts[i]*100.0, 1),
            "SegBoostM%": round(mot_boosts[i]*100.0, 1),
            "GoogleTrends": round(trends_n[i], 1),
            "TrendsSource": trend_srcs[i],
            "WikipediaRawAvgDaily": round(wiki_scores[i], 2),
            "WikipediaFamiliarity": round(wiki_fam[i], 1),
            "YouTubeN": round(yt_combo[i], 1),
            "YouTubeStatus": yt_status[i],
            "SpotifyN": round(sp_n[i], 1),
        })

    df = pd.DataFrame(rows)
    return df

def apply_benchmark(df: pd.DataFrame, benchmark_title: str, use_adjusted: bool) -> pd.DataFrame:
    df = df.copy()
    if benchmark_title not in df["Title"].values:
        return df
    b_row = df[df["Title"] == benchmark_title].iloc[0]
    cols = ["GoogleTrends","WikipediaFamiliarity","YouTubeN","SpotifyN"]
    if use_adjusted:
        cols = ["FamiliarityAdj","MotivationAdj"] + cols
    else:
        cols = ["Familiarity","Motivation"] + cols
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
def quadrant_plot(df: pd.DataFrame, colx: str, coly: str, title: str):
    fig = plt.figure()
    x = df[colx].values
    y = df[coly].values
    plt.scatter(x, y)
    for _, r in df.iterrows():
        plt.annotate(r["Title"], (float(r[colx]), float(r[coly])), fontsize=8, xytext=(3,3), textcoords="offset points")
    plt.axvline(np.median(x), linestyle="--")
    plt.axhline(np.median(y), linestyle="--")
    plt.xlabel(colx)
    plt.ylabel(coly)
    plt.title(title)
    return fig

def bar_chart(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    order = df.sort_values(by=col, ascending=True)
    plt.barh(order["Title"], order[col])
    plt.title(title)
    plt.xlabel(col)
    return fig

def generate_pdf_brief(df: pd.DataFrame, use_adjusted: bool, file_path: str):
    fx, fy = ("FamiliarityAdj","MotivationAdj") if use_adjusted else ("Familiarity","Motivation")
    with PdfPages(file_path) as pdf:
        fig1 = plt.figure(); plt.axis('off')
        plt.title("Alberta Ballet — Title Scores", pad=20)
        tab_cols = ["Title","Market","TargetSegment",fx,fy,"SegBoostF%","SegBoostM%","GoogleTrends","TrendsSource","WikipediaFamiliarity","YouTubeN","SpotifyN","ResolvedWikiPage"]
        tab = plt.table(cellText=df[tab_cols].round(1).values, colLabels=tab_cols, loc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(6.5); tab.scale(1, 1.1)
        pdf.savefig(fig1, bbox_inches='tight'); plt.close(fig1)

        fig2 = quadrant_plot(df, fx, fy, f"{fx} vs {fy} (Quadrant Map)")
        pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)

        fig3 = bar_chart(df, fx, f"{fx} by Title")
        pdf.savefig(fig3, bbox_inches='tight'); plt.close(fig3)

        fig4 = bar_chart(df, fy, f"{fy} by Title")
        pdf.savefig(fig4, bbox_inches='tight'); plt.close(fig4)

        fig5 = plt.figure(); plt.axis('off')
        memo = (
            "Self-test verifies each data source. Trends falls back City→AB→CA→Global. "
            "Wikipedia uses pageviews with caching & proper User-Agent. "
            "YouTube shows status (OK/NO_KEY/HTTP_ERROR)."
        )
        plt.text(0.01, 0.99, memo, va='top')
        pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Viability", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v7.2)")

st.markdown("Now with **API Self-Test**, hardened Wikipedia requests, clearer YouTube/Trends diagnostics, and caching.")

# --- API keys auto-load (secrets/env) ---
def load_api_keys_from_env_and_secrets():
    yt = st.secrets.get("YOUTUBE_API_KEY") if hasattr(st, "secrets") else None
    sp_id = st.secrets.get("SPOTIFY_CLIENT_ID") if hasattr(st, "secrets") else None
    sp_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET") if hasattr(st, "secrets") else None
    yt = yt or os.environ.get("YOUTUBE_API_KEY")
    sp_id = sp_id or os.environ.get("SPOTIFY_CLIENT_ID")
    sp_secret = sp_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
    return yt, sp_id, sp_secret

with st.expander("🔑 API Configuration"):
    auto_yt, auto_sp_id, auto_sp_secret = load_api_keys_from_env_and_secrets()
    if auto_yt or auto_sp_id or auto_sp_secret:
        st.markdown("**Status:** Using keys from secrets/env (you can override below). ✅")
    yt_key = st.text_input("YouTube Data API v3 Key", type="password", value=auto_yt or "")
    sp_id = st.text_input("Spotify Client ID", type="password", value=auto_sp_id or "")
    sp_secret = st.text_input("Spotify Client Secret", type="password", value=auto_sp_secret or "")

with st.expander("🧪 API Self-Test"):
    st.caption("Runs a quick probe on Google Trends, Wikipedia, and YouTube using 'The Nutcracker'.")
    colt1, colt2 = st.columns(2)
    with colt1:
        run_test = st.button("Run Self-Test")
    with colt2:
        clear_cache = st.button("Clear Cached Lookups")
    if clear_cache:
        _pytrends.clear()
        wiki_search_best_title.clear()
        fetch_wikipedia_views_for_page.clear()
        fetch_youtube_metrics.clear()
        fetch_spotify_popularity.clear()
        st.success("Caches cleared.")
    if run_test:
        # Wikipedia test
        wp_title, wp_views = fetch_wikipedia_views("The Nutcracker")
        st.write(f"**Wikipedia**: page='{wp_title}', avg_daily_views={wp_views:.2f} → {'OK' if wp_views>0 else 'ZERO'}")

        # Trends test (AB & City fallback)
        t_score, t_src = fetch_google_trends_score("The Nutcracker", market="Calgary")
        st.write(f"**Google Trends**: score={t_score:.2f}, source={t_src} → {'OK' if t_score>0 else 'ZERO'}")

        # YouTube test
        ysc, ytv, ystat = fetch_youtube_metrics("The Nutcracker ballet", yt_key.strip() or None)
        st.write(f"**YouTube**: search_count_norm_src≈{ysc:.0f}, top_view_sqrt≈{ytv:.0f}, status={ystat}")

        if ystat.startswith("HTTP_ERROR"):
            st.info("If you see HTTP_ERROR, confirm in Google Cloud Console that **YouTube Data API v3** is enabled for your key, and the key isn't restricted to wrong referrers.")
        if t_src.startswith("ERR"):
            st.info("Trends error: pytrends was blocked or failed. Try again in a minute, or switch Market to Alberta/Global.")

with st.expander("⚙️ Options"):
    use_trends = st.checkbox("Use Google Trends", value=True)
    market = st.selectbox("Market", ["Alberta (province)", "Calgary (city)", "Edmonton (city)"], index=0)
    market_key = {"Alberta (province)":"AB","Calgary (city)":"Calgary","Edmonton (city)":"Edmonton"}[market]
    segment_name = st.selectbox("Target Segment (applies demographic boosts)", list(SEGMENT_RULES.keys()), index=0)
    use_adjusted = st.checkbox("Use Segment-Adjusted scores in charts/exports", value=True)
    st.caption("If city data is sparse, Trends falls back to Alberta → Canada → Global automatically.")

default_titles = [
    "The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio",
    "The Merry Widow","The Hunchback of Notre Dame","Frozen",
    "Beauty and the Beast","Alice in Wonderland","Peter Pan",
    "Romeo and Juliet","Swan Lake","Don Quixote","Contemporary Composers",
    "Nijinsky","Notre Dame de Paris","Wizard of Oz","Grimm"
]
titles_input = st.text_area("Enter titles (one per line):", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

# --- Title attributes editor ---
st.subheader("Title Attributes (edit to fit your production/angle)")
attr_rows = []
for t in titles:
    inferred = infer_title_attributes(t)
    row = {"Title": t}
    row.update(inferred)
    attr_rows.append(row)
attrs_df = pd.DataFrame(attr_rows, columns=["Title"]+ATTR_COLUMNS)
attrs_df = st.data_editor(attrs_df, use_container_width=True, hide_index=True)

colA, colB, colC, colD = st.columns(4)
with colA:
    do_benchmark = st.checkbox("Normalize to a benchmark title?")
with colB:
    benchmark_title = st.selectbox("Benchmark title", options=titles, index=0 if "The Nutcracker" in titles else 0)
with colC:
    run_btn = st.button("Score Titles", type="primary")
with colD:
    gen_pdf = st.checkbox("Prepare PDF after scoring", value=False)

if run_btn:
    with st.spinner("Scoring titles…"):
        df = score_titles(titles, yt_key.strip() or None, sp_id.strip() or None, sp_secret.strip() or None,
                          use_trends, market_key, segment_name, attrs_df)

        sort_x, sort_y = ("FamiliarityAdj","MotivationAdj") if use_adjusted else ("Familiarity","Motivation")
        df = df.sort_values(by=[sort_y, sort_x], ascending=False)

        if do_benchmark and benchmark_title:
            df = apply_benchmark(df, benchmark_title, use_adjusted)

        st.success("Done.")
        st.dataframe(df, use_container_width=True)

        # Diagnostics — dispersion & source status
        st.subheader("Diagnostics")
        diag_cols = [sort_x, sort_y, "GoogleTrends","TrendsSource","WikipediaRawAvgDaily","WikipediaFamiliarity","YouTubeN","YouTubeStatus","SpotifyN"]
        diag = pd.DataFrame({
            "metric": diag_cols,
            "min":    [float(df[c].min()) if c not in ["TrendsSource","YouTubeStatus"] else "—" for c in diag_cols],
            "max":    [float(df[c].max()) if c not in ["TrendsSource","YouTubeStatus"] else "—" for c in diag_cols],
            "std":    [float(df[c].std(ddof=0)) if c not in ["TrendsSource","YouTubeStatus"] else "—" for c in diag_cols],
            "notes":  [", ".join(sorted(df[c].astype(str).unique())) if c in ["TrendsSource","YouTubeStatus"] else "" for c in diag_cols]
        })
        st.dataframe(diag, use_container_width=True)

        # Charts
        st.subheader("Charts")
        figQ = quadrant_plot(df, sort_x, sort_y, f"{sort_x} vs {sort_y} (Quadrant Map)")
        st.pyplot(figQ)
        figF = bar_chart(df, sort_x, f"{sort_x} by Title")
        st.pyplot(figF)
        figM = bar_chart(df, sort_y, f"{sort_y} by Title")
        st.pyplot(figM)

        # CSV / PDF
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "title_scores.csv", "text/csv")
        if gen_pdf:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = f"title_scores_brief_{ts}.pdf"
            generate_pdf_brief(df, use_adjusted, pdf_path)
            st.success("PDF created.")
            st.download_button("⬇️ Download PDF Brief", data=open(pdf_path, "rb").read(),
                               file_name=pdf_path, mime="application/pdf")
