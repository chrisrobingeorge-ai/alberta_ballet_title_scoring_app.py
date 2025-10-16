# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation + City-level Sales Model (v8.0)
# - Markets: Alberta/Calgary/Edmonton (Google Trends with robust fallbacks)
# - Live APIs: Wikipedia pageviews, Google Trends, YouTube (low quota), Spotify
# - Segment boosts (Core Classical / Family / Emerging Adults)
# - Title attributes inference + editable table
# - Charts + CSV/PDF export
# - Train city-level models (Calgary/Edmonton), exclude 2021 pandemic season
# - Session-safe training/prediction (no "pane bounce")

import os, math, time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import streamlit as st
st.set_page_config(page_title="Alberta Ballet — Title Viability", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests
from requests.adapters import HTTPAdapter, Retry

# Optional APIs (YouTube / Spotify)
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

# Try scikit-learn for the training module
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -------------------------
# CONFIG
# -------------------------
TRENDS_TIMEFRAME = "today 5-y"

ATTR_COLUMNS = [
    "female_lead","male_lead","family_friendly","romantic","tragic",
    "contemporary","classic_canon","spectacle","pop_ip"
]

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
# GOOGLE TRENDS (robust fallbacks)
# -------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _pytrends():
    from pytrends.request import TrendReq
    return TrendReq(hl="en-US", tz=360, retries=5, backoff_factor=0.5)

def _trends_kw_variants(title: str) -> List[str]:
    return [title, f"{title} ballet", f"{title} story"]

def fetch_trends_province_or_global(title: str, region_geo: str) -> Tuple[float, str]:
    """Returns (score, source_label). Tries region -> CA -> Global."""
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
                    time.sleep(0.2)
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else 0.0

        score = _get_score(region_geo)
        if score > 0: return score, region_geo
        if region_geo != "CA":
            score = _get_score("CA")
            if score > 0: return score, "CA"
        score = _get_score("")
        if score > 0: return score, "GLOBAL"
        return 0.0, "NONE"
    except Exception as e:
        return 0.0, f"ERR:{e}"

def fetch_trends_city(title: str, city_name: str) -> Tuple[float, str]:
    """City via interest_by_region(resolution='CITY'), falls back to AB/CA/Global."""
    try:
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
        if s > 0: return s, src
        return fetch_trends_province_or_global(title, "CA-AB")
    if market == "Edmonton":
        s, src = fetch_trends_city(title, "Edmonton")
        if s > 0: return s, src
        return fetch_trends_province_or_global(title, "CA-AB")
    return fetch_trends_province_or_global(title, "CA-AB")

# -------------------------
# WIKIPEDIA (robust search + pageviews)
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
# YOUTUBE / SPOTIFY (low-quota YT) + caching
# -------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float, str]:
    """
    Low-quota version: one search call, no videos.list. Returns (#results_seen, 0, status).
    """
    if not api_key:
        return 0.0, 0.0, "NO_KEY"
    if build is None:
        return 0.0, 0.0, "LIB_MISSING"
    try:
        yt = build("youtube", "v3", developerKey=api_key)
        resp = yt.search().list(q=title, part="id", type="video", maxResults=5).execute()
        ids = [item["id"]["videoId"] for item in resp.get("items", [])]
        return float(len(ids)), 0.0, ("OK" if ids else "NO_RESULTS")
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
    fam_delta = 0.0; mot_delta = 0.0
    for k, v in rules.get("fam", {}).items():
        if attrs.get(k, False): fam_delta += v
    for k, v in rules.get("mot", {}).items():
        if attrs.get(k, False): mot_delta += v
    fam_delta = max(-0.30, min(0.40, fam_delta))
    mot_delta = max(-0.30, min(0.40, mot_delta))
    return fam_delta, mot_delta

# -------------------------
# SCORING ENGINE
# -------------------------
def score_titles(titles, yt_key, sp_id, sp_secret, use_trends, market, segment_name, attrs_df,
                 w_fam_trends, w_fam_wiki, w_fam_spot, w_mot_trends, w_mot_yt, w_mot_spot, w_mot_wiki):
    # Map title -> attrs from editor
    user_attrs = {}
    for _, row in attrs_df.iterrows():
        nm = str(row["Title"]).strip()
        if not nm: continue
        user_attrs[nm.lower()] = {col: bool(row.get(col, False)) for col in ATTR_COLUMNS}

    trends_scores, trend_srcs, wiki_titles, wiki_scores = [], [], [], []
    yt_search_counts, yt_status, sp_pops = [], [], []
    fam_boosts, mot_boosts = [], []

    for t in titles:
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
        ysc, _, ystat = fetch_youtube_metrics(t, yt_key)
        yt_search_counts.append(ysc); yt_status.append(ystat)
        sp = fetch_spotify_popularity(t, sp_id, sp_secret)
        sp_pops.append(sp)

        # Segment boost
        base_attrs = infer_title_attributes(t)
        over = user_attrs.get(t.lower(), {})
        base_attrs.update(over)
        fdelta, mdelta = segment_boosts(base_attrs, segment_name)
        fam_boosts.append(fdelta); mot_boosts.append(mdelta)

    # Familiarity base from Wikipedia: log -> percentile
    wiki_log = [math.log1p(v) for v in wiki_scores]
    wiki_fam = percentile_normalize(wiki_log)  # 0..100

    # Other sources normalized 0..100
    trends_n   = trends_scores if use_trends else [0.0] * len(titles)
    sp_n       = normalize_series(sp_pops)
    yt_searchn = normalize_series(yt_search_counts)

    # Dynamic weights + auto-rebalance on failure
    def _rebalance(d):
        s = sum(d.values())
        return {k: (v/s if s>0 else 0.0) for k,v in d.items()}

    fam_w = {
        "wiki": (w_fam_wiki/100.0),
        "trends": (w_fam_trends/100.0) if use_trends and any(trends_n) else 0.0,
        "spotify": (w_fam_spot/100.0) if any(sp_pops) else 0.0,
    }
    mot_w = {
        "youtube": (w_mot_yt/100.0) if any(yt_search_counts) and ("OK" in set(yt_status) or "NO_RESULTS" in set(yt_status)) else 0.0,
        "trends": (w_mot_trends/100.0) if use_trends and any(trends_n) else 0.0,
        "spotify": (w_mot_spot/100.0) if any(sp_pops) else 0.0,
        "wiki": (w_mot_wiki/100.0),
    }
    fam_w = _rebalance(fam_w)
    mot_w = _rebalance(mot_w)

    rows = []
    for i, t in enumerate(titles):
        fam = (
            fam_w["wiki"]   * (wiki_fam[i] or 0.0) +
            fam_w["trends"] * (trends_n[i] or 0.0) +
            fam_w["spotify"]* (sp_n[i] or 0.0)
        )
        mot = (
            mot_w["youtube"]* (yt_searchn[i] or 0.0) +
            mot_w["trends"] * (trends_n[i] or 0.0) +
            mot_w["spotify"]* (sp_n[i] or 0.0) +
            mot_w["wiki"]   * (wiki_fam[i] or 0.0)
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
            "YouTubeN": round(yt_searchn[i], 1),
            "YouTubeStatus": yt_status[i],
            "SpotifyN": round(sp_n[i], 1),
        })
    return pd.DataFrame(rows)

def apply_benchmark(df: pd.DataFrame, benchmark_title: str, use_adjusted: bool) -> pd.DataFrame:
    df = df.copy()
    if benchmark_title not in df["Title"].values:
        return df
    b_row = df[df["Title"] == benchmark_title].iloc[0]
    cols = (["FamiliarityAdj","MotivationAdj"] if use_adjusted else ["Familiarity","Motivation"]) + \
           ["GoogleTrends","WikipediaFamiliarity","YouTubeN","SpotifyN"]
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
    plt.xlabel(colx); plt.ylabel(coly); plt.title(title)
    return fig

def bar_chart(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure()
    order = df.sort_values(by=col, ascending=True)
    plt.barh(order["Title"], order[col])
    plt.title(title); plt.xlabel(col)
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
            "Trends falls back City→AB→CA→Global; Wikipedia uses pageviews (log→percentile). "
            "YouTube is low-quota (search only). Source weights auto-rebalance if a source fails."
        )
        plt.text(0.01, 0.99, memo, va='top')
        pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# -------------------------
# UI — SCORING
# -------------------------
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation (v8.0)")
st.markdown("Market-aware Trends + robust Wikipedia + low-quota YouTube. Segment boosts reflect Calgary/Edmonton gender/age realities. Weights auto-rebalance when a source fails.")

# API keys from secrets/env
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
    st.caption("Quick probe on Wikipedia, Google Trends, and YouTube using 'The Nutcracker'.")
    c1, c2 = st.columns(2)
    with c1:
        run_test = st.button("Run Self-Test")
    with c2:
        clear_cache = st.button("Clear Cached Lookups")
    if clear_cache:
        _pytrends.clear(); wiki_search_best_title.clear()
        fetch_wikipedia_views_for_page.clear()
        fetch_youtube_metrics.clear(); fetch_spotify_popularity.clear()
        st.success("Caches cleared.")
    if run_test:
        wp_title, wp_views = fetch_wikipedia_views("The Nutcracker")
        st.write(f"**Wikipedia**: page='{wp_title}', avg_daily_views={wp_views:.2f} → {'OK' if wp_views>0 else 'ZERO'}")
        t_score, t_src = fetch_google_trends_score("The Nutcracker", market="Calgary")
        st.write(f"**Google Trends**: score={t_score:.2f}, source={t_src} → {'OK' if t_score>0 else 'ZERO'}")
        ysc, _, ystat = fetch_youtube_metrics("The Nutcracker ballet", yt_key.strip() or None)
        st.write(f"**YouTube**: results≈{ysc:.0f}, status={ystat}")
        if str(ystat).startswith("HTTP_ERROR"):
            st.info("If you see HTTP_ERROR, ensure YouTube Data API v3 is enabled and the key isn't restricted incorrectly.")

with st.expander("⚙️ Options"):
    use_trends = st.checkbox("Use Google Trends", value=True)
    market = st.selectbox("Market", ["Alberta (province)", "Calgary (city)", "Edmonton (city)"], index=0)
    market_key = {"Alberta (province)":"AB","Calgary (city)":"Calgary","Edmonton (city)":"Edmonton"}[market]
    segment_name = st.selectbox("Target Segment (applies demographic boosts)", list(SEGMENT_RULES.keys()), index=0)
    use_adjusted = st.checkbox("Use Segment-Adjusted scores in charts/exports", value=True)
    st.caption("If city data is sparse, Trends falls back to Alberta → Canada → Global automatically.")

with st.expander("🔩 Source Weights (advanced)"):
    st.caption("If a source fails (e.g., Trends=NONE, YouTube quota), weights auto-rebalance.")
    w_fam_trends = st.slider("Weight: Familiarity ← Google Trends", 0, 100, 30)
    w_fam_wiki   = st.slider("Weight: Familiarity ← Wikipedia", 0, 100, 55)
    w_fam_spot   = st.slider("Weight: Familiarity ← Spotify", 0, 100, 15)
    w_mot_trends = st.slider("Weight: Motivation ← Google Trends", 0, 100, 25)
    w_mot_yt     = st.slider("Weight: Motivation ← YouTube", 0, 100, 45)
    w_mot_spot   = st.slider("Weight: Motivation ← Spotify", 0, 100, 15)
    w_mot_wiki   = st.slider("Weight: Motivation ← Wikipedia", 0, 100, 15)

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
    row = {"Title": t}; row.update(inferred)
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
        df = score_titles(
            titles,
            yt_key.strip() or None,  # YouTube
            sp_id.strip() or None, sp_secret.strip() or None,  # Spotify
            use_trends, market_key, segment_name, attrs_df,
            w_fam_trends, w_fam_wiki, w_fam_spot, w_mot_trends, w_mot_yt, w_mot_spot, w_mot_wiki
        )
        sort_x, sort_y = ("FamiliarityAdj","MotivationAdj") if use_adjusted else ("Familiarity","Motivation")
        df = df.sort_values(by=[sort_y, sort_x], ascending=False)

        if do_benchmark and benchmark_title:
            df = apply_benchmark(df, benchmark_title, use_adjusted)

        st.success("Done.")
        st.dataframe(df, use_container_width=True)

        # Diagnostics
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
        st.pyplot(quadrant_plot(df, sort_x, sort_y, f"{sort_x} vs {sort_y} (Quadrant Map)"))
        st.pyplot(bar_chart(df, sort_x, f"{sort_x} by Title"))
        st.pyplot(bar_chart(df, sort_y, f"{sort_y} by Title"))

        # CSV / PDF
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "title_scores.csv", "text/csv")
        if gen_pdf:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = f"title_scores_brief_{ts}.pdf"
            generate_pdf_brief(df, use_adjusted, pdf_path)
            st.success("PDF created.")
            st.download_button("⬇️ Download PDF Brief", data=open(pdf_path, "rb").read(),
                               file_name=pdf_path, mime="application/pdf")

# ============================================
# 📈 Train Sales Model (city-level, tickets + revenue) — session_state safe
# ============================================
from io import StringIO

st.markdown("---")
st.header("📈 Train Sales Model (City-Level)")

# ---- helpers ----
def _long_city_rows(df):
    rows = []
    for _, r in df.iterrows():
        season = r.get("season", None)
        title  = str(r.get("title", "")).strip()
        rows.append({"season": season, "title": title, "city": "Calgary",
                     "tickets_city": r.get("yyc_tickets", None),
                     "revenue_city": r.get("yyc_revenue", None)})
        rows.append({"season": season, "title": title, "city": "Edmonton",
                     "tickets_city": r.get("yeg_tickets", None),
                     "revenue_city": r.get("yeg_revenue", None)})
    out = pd.DataFrame(rows)
    for c in ["tickets_city","revenue_city","season"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["title"] = out["title"].astype(str)
    out["city"] = out["city"].astype(str)
    return out

def _attach_title_attributes(df_titles: pd.DataFrame) -> pd.DataFrame:
    attr_records = []
    for t in df_titles["title"].astype(str):
        attrs = infer_title_attributes(t)
        rec = {"title": t}
        rec.update({k: bool(v) for k,v in attrs.items()})
        attr_records.append(rec)
    attrs_df2 = pd.DataFrame(attr_records).drop_duplicates(subset=["title"])
    return df_titles.merge(attrs_df2, on="title", how="left")

def _build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    long = _long_city_rows(raw_df)
    long = long[long["season"].astype("Int64") != 2021]  # exclude 2021 entirely
    long = _attach_title_attributes(long)
    return long

def _time_split(dfX, dfY, season_col="season"):
    max_season = int(dfX[season_col].max())
    train_idx = dfX[season_col] < max_season
    test_idx  = dfX[season_col] == max_season
    return train_idx, test_idx, max_season

def _random_split(dfX, dfY, by_col="title", p_train=0.8, seed=42):
    rng = np.random.default_rng(seed)
    titles_u = dfX[by_col].drop_duplicates().values
    rng.shuffle(titles_u)
    cut = int(len(titles_u)*p_train)
    train_titles = set(titles_u[:cut])
    train_idx = dfX[by_col].isin(train_titles)
    test_idx  = ~train_idx
    return train_idx, test_idx, None

def _train_one_target(feat_df: pd.DataFrame, target_col: str, holdout_mode: str):
    data = feat_df.copy()
    data = data[pd.notna(data[target_col]) & (data[target_col] > 0)]

    attr_cols = [c for c in data.columns if c in ATTR_COLUMNS]
    feature_cols_num = ["season"]           # numeric
    feature_cols_cat = ["city"]             # categorical
    feature_cols_bin = attr_cols            # booleans -> numeric

    X = data[feature_cols_num + feature_cols_cat + feature_cols_bin].copy()
    for c in feature_cols_bin:
        X[c] = X[c].astype(float).fillna(0.0).astype(int)
    y = data[target_col].astype(float)

    if holdout_mode.startswith("Last season"):
        tr_idx, te_idx, max_season = _time_split(X.assign(_season=data["season"]), y, season_col="_season")
        split_note = f"Hold-out season: {max_season}"
    else:
        tr_idx, te_idx, _ = _random_split(X.assign(_title=data["title"]), y, by_col="_title")
        split_note = "Random 80/20 split by production"

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn not installed. Add scikit-learn to requirements.txt and redeploy.")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), feature_cols_num + feature_cols_bin),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ], remainder="drop"
    )
    model = Pipeline(steps=[("pre", pre), ("ridge", Ridge(alpha=1.0, random_state=42))])
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    else:
        r2, mae = np.nan, np.nan

    report = {"target": target_col, "r2": r2, "mae": mae, "split": split_note,
              "n_train": int(tr_idx.sum()), "n_test": int(te_idx.sum())}
    meta = {"feature_cols_num": feature_cols_num, "feature_cols_cat": feature_cols_cat, "feature_cols_bin": feature_cols_bin}
    return model, report, meta

def _predict_for_titles(model, meta, titles_list, city_list, attrs_df_editable):
    """Generate predictions by city for given titles using trained model."""
    now_year = datetime.now().year + 1
    attr_map = {}
    for _, row in attrs_df_editable.iterrows():
        nm = str(row["Title"]).strip()
        if not nm:
            continue
        attr_map[nm.lower()] = {k: bool(row.get(k, False)) for k in meta["feature_cols_bin"]}

    rows = []
    for city in city_list:
        for t in titles_list:
            attrs = infer_title_attributes(t)
            if t.lower() in attr_map:
                attrs.update(attr_map[t.lower()])
            rec = {"title": t, "city": city, "season": now_year}
            for k in meta["feature_cols_bin"]:
                rec[k] = int(bool(attrs.get(k, False)))
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["title", "city", "season", "prediction"])

    pred_df = pd.DataFrame(rows)

    # Safety: ensure required columns exist
    X_cols = meta["feature_cols_num"] + meta["feature_cols_cat"] + meta["feature_cols_bin"]
    for col in X_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0

    try:
        y_hat = model.predict(pred_df[X_cols])
        pred_df["prediction"] = y_hat
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        pred_df["prediction"] = np.nan

    if "season" not in pred_df.columns:
        pred_df["season"] = now_year

    return pred_df

# --------- session_state to survive reruns ----------
with st.expander("Upload & Configure", expanded=True):
    for k in ["sales_df", "feat_df", "models", "model_meta", "reports"]:
        if k not in st.session_state:
            st.session_state[k] = None if k not in ["models","model_meta","reports"] else {}

    sales_file = st.file_uploader(
        "Upload cleaned sales dataset (e.g., alberta_ballet_productions_2017_2025_clean_positive.csv)",
        type=["csv","xlsx"],
        key="uploader_sales"
    )

    target_choice = st.multiselect(
        "Targets to train (both recommended):",
        ["tickets", "revenue"],
        default=["tickets", "revenue"],
        key="targets_sel"
    )

    holdout_mode = st.selectbox(
        "Validation split",
        ["Last season hold-out (time-based)", "80/20 random (by production)"],
        index=0, key="holdout_sel"
    )

    with st.form("train_form", clear_on_submit=False):
        st.caption("2021 is excluded automatically from training and validation.")
        train_submit = st.form_submit_button("Train Model(s)")

    if sales_file is not None:
        if sales_file.name.lower().endswith(".csv"):
            st.session_state.sales_df = pd.read_csv(sales_file)
        else:
            st.session_state.sales_df = pd.read_excel(sales_file)
        st.write("Preview:", st.session_state.sales_df.head(8))

    if train_submit:
        if st.session_state.sales_df is None:
            st.error("Please upload the cleaned sales CSV first.")
        elif not SKLEARN_OK:
            st.error("scikit-learn is not installed. Add it to requirements.txt and redeploy.")
        else:
            st.session_state.feat_df = _build_feature_table(st.session_state.sales_df)
            st.write("Training rows (city-level):", st.session_state.feat_df.head(8))

            st.session_state.models = {}
            st.session_state.model_meta = {}
            st.session_state.reports = {}

            if "tickets" in st.session_state.targets_sel:
                m_tix, rep_tix, meta_tix = _train_one_target(st.session_state.feat_df, "tickets_city", st.session_state.holdout_sel)
                st.session_state.models["tickets"] = m_tix
                st.session_state.model_meta["tickets"] = meta_tix
                st.session_state.reports["tickets"] = rep_tix

            if "revenue" in st.session_state.targets_sel:
                m_rev, rep_rev, meta_rev = _train_one_target(st.session_state.feat_df, "revenue_city", st.session_state.holdout_sel)
                st.session_state.models["revenue"] = m_rev
                st.session_state.model_meta["revenue"] = meta_rev
                st.session_state.reports["revenue"] = rep_rev

            st.success("Models trained and saved. Scroll down to ‘Predict for Current Titles’.")

# Show reports if present
if st.session_state.get("reports"):
    st.subheader("Validation Metrics")
    rep_rows = []
    for k, rep in st.session_state.reports.items():
        rep_rows.append({
            "Target": k,
            "R^2": round(rep["r2"], 3) if pd.notna(rep["r2"]) else "—",
            "MAE": round(rep["mae"], 1) if pd.notna(rep["mae"]) else "—",
            "Split": rep["split"],
            "Train rows": rep["n_train"],
            "Test rows": rep["n_test"],
        })
    st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

# ------------- Predict -------------
st.subheader("Predict for Current Titles (by City)")
with st.form("predict_form", clear_on_submit=False):
    city_sel = st.multiselect("Cities to predict", ["Calgary","Edmonton"], default=["Calgary","Edmonton"], key="city_sel_predict")
    predict_submit = st.form_submit_button("Predict Now")

if predict_submit:
    if not st.session_state.get("models"):
        st.error("Please train the model(s) first (above).")
    else:
        tables = []
        if "tickets" in st.session_state.models:
            preds_t = _predict_for_titles(st.session_state.models["tickets"], st.session_state.model_meta["tickets"], titles, city_sel, attrs_df)
            preds_t = preds_t.rename(columns={"prediction": "predicted_tickets"})
            tables.append(preds_t)
        if "revenue" in st.session_state.models:
            preds_r = _predict_for_titles(st.session_state.models["revenue"], st.session_state.model_meta["revenue"], titles, city_sel, attrs_df)
            preds_r = preds_r.rename(columns={"prediction": "predicted_revenue"})
            tables.append(preds_r)

        if tables:
            out = tables[0]
            for extra in tables[1:]:
                colname = [c for c in extra.columns if c.startswith("predicted_")][0]
                out = out.merge(extra[["title","city", colname]], on=["title","city"], how="outer")

            # Add attribute snapshot columns for reference
            bin_cols = ATTR_COLUMNS
            attr_snap = []
            for t in out["title"].astype(str):
                att = infer_title_attributes(t)
                attr_snap.append({"title": t, **{k:int(bool(att.get(k, False))) for k in bin_cols}})
            out = out.merge(pd.DataFrame(attr_snap), on="title", how="left")

            # Reorder columns safely
            cols_exist = [c for c in ["title","city","season"] if c in out.columns]
            pred_cols  = [c for c in out.columns if c.startswith("predicted_")]
            other_cols = [c for c in bin_cols if c in out.columns]
            out = out[cols_exist + pred_cols + other_cols]

            st.dataframe(out.sort_values(["city","title"]), use_container_width=True)

            buf = StringIO(); out.to_csv(buf, index=False)
            st.download_button("⬇️ Download Predictions (CSV)", data=buf.getvalue().encode("utf-8"),
                               file_name="predictions_by_city.csv", mime="text/csv")
