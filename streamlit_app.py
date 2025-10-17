# streamlit_app.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v8)
# Rule-based update: folds in historical validation without ML training.
# - CSV upload for historical actuals; auto‑excludes pandemic season rows
# - Likeness buckets + Gender tagging for priors
# - Priors computed per (Bucket × Gender × Market) using only seasons < cutoff
# - Forecast = Historical Prior × Score Adapter (from Familiarity/Motivation)
# - Confidence bands derived from cohort CV
# - No leakage: you choose a cutoff season; priors use strictly earlier rows
# - Everything else from v7 retained (Trends/Wiki/YouTube/Spotify optional)

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

WEIGHTS = {
    "fam_wiki": 0.55,
    "fam_trends": 0.30,
    "fam_spotify": 0.15,
    "mot_youtube": 0.45,
    "mot_trends": 0.25,
    "mot_spotify": 0.15,
    "mot_wiki": 0.15,
}

# Adapter: how strongly the score nudges the historical prior (0..1 sensible)
SCORE_ADAPTER_STRENGTH = 0.40

# Segment multipliers (existing v7 logic)
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
    if not vals:
        return []
    s = pd.Series(vals, dtype="float64")
    ranks = s.rank(method="average", pct=True)
    return (ranks * 100.0).tolist()

def clamp01(x):
    return max(0.0, min(1.0, x))

# -------------------------
# GOOGLE TRENDS
# -------------------------
def _pytrends():
    from pytrends.request import TrendReq
    return TrendReq(hl="en-US", tz=360, retries=5, backoff_factor=0.5)

def _trends_kw_variants(title: str) -> List[str]:
    return [title, f"{title} ballet", f"{title} story"]

def fetch_trends_province_or_global(title: str, region_geo: str) -> float:
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
                except Exception:
                    continue
            return float(np.mean(vals)) if vals else 0.0
        score = _get_score(region_geo)
        if score == 0.0 and region_geo != "CA":
            score = _get_score("CA")
        if score == 0.0:
            score = _get_score("")
        return score
    except Exception:
        return 0.0

def fetch_trends_city(title: str, city_name: str) -> float:
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
                vals.append(v)
        if vals:
            return float(np.mean(vals))
        return fetch_trends_province_or_global(title, "CA-AB")
    except Exception:
        return 0.0

def fetch_google_trends_score(title: str, market: str) -> float:
    if market == "Calgary":
        s = fetch_trends_city(title, "Calgary")
        return s if s > 0 else fetch_trends_province_or_global(title, "CA-AB")
    if market == "Edmonton":
        s = fetch_trends_city(title, "Edmonton")
        return s if s > 0 else fetch_trends_province_or_global(title, "CA-AB")
    return fetch_trends_province_or_global(title, "CA-AB")

# -------------------------
# WIKIPEDIA
# -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"

def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        params = {"action": "query","list": "search","srsearch": query,"format": "json","srlimit": 5}
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
# YOUTUBE / SPOTIFY (optional)
# -------------------------
def fetch_youtube_metrics(title: str, api_key: Optional[str]) -> Tuple[float, float]:
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
# VALIDATION-DRIVEN PRIORS (NO ML)
# -------------------------
FAMILY_SET = {"the nutcracker","nutcracker","cinderella","beauty and the beast","alice in wonderland","peter pan","pinocchio","frozen","wizard of oz"}
ROMANCE_SET = {"swan lake","sleeping beauty","giselle","romeo and juliet","the merry widow","merry widow"}

MALE_TITLES = {"pinocchio","peter pan","don quixote","hunchback","notre dame","notre dame de paris"}
FEMALE_TITLES = {"nutcracker","sleeping beauty","cinderella","beauty and the beast","alice in wonderland","giselle","swan lake","the merry widow","romeo and juliet"}


def likeness_bucket(title: str) -> str:
    t = (title or "").strip().lower()
    if t in FAMILY_SET:
        return "Family Classic"
    if t in ROMANCE_SET:
        return "Classic Romance"
    if any(k in t for k in ["notre dame","hunchback","contemporary","composers","mixed bill","forsythe","balanchine","jerome robbins","nijinsky","grimm","new works","don quixote"]):
        return "Spectacle/Contemporary"
    return "Spectacle/Contemporary"


def gender_focus(title: str) -> str:
    t = (title or "").strip().lower().replace("the ", "")
    if "romeo and juliet" in t:
        return "dual/ensemble"
    if any(k in t for k in FEMALE_TITLES):
        return "female_lead"
    if any(k in t for k in MALE_TITLES):
        return "male_lead"
    if any(k in t for k in ["contemporary","composers","mixed bill","forsythe","balanchine","jerome robbins","nijinsky","grimm","new works"]):
        return "dual/ensemble"
    return "dual/ensemble"


def parse_season_start_year(s: Optional[str]) -> Optional[int]:
    if pd.isna(s):
        return None
    import re
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else None


def safe_div(a, b):
    try:
        return float(a) / float(b) if (pd.notna(a) and pd.notna(b) and float(b) != 0.0) else np.nan
    except Exception:
        return np.nan


def build_tickets_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "pandemic_season" in df.columns:
        df = df[df["pandemic_season"] != 1].copy()
    df["tickets_est_from_rev"] = [safe_div(a,b) for a,b in zip(df.get("total_revenue"), df.get("avg_price_overall"))]
    df["tickets_final"] = df.get("total_tickets")
    mask_missing = df["tickets_final"].isna() | (df["tickets_final"]<=0)
    df.loc[mask_missing, "tickets_final"] = df.loc[mask_missing, "tickets_est_from_rev"]
    # City splits
    def estimate_city_tickets(row, city="yyc"):
        t = row.get(f"{city}_tickets")
        if pd.notna(t) and float(t) > 0:
            return float(t)
        rev = row.get(f"{city}_revenue"); price = row.get(f"avg_price_{city}")
        if pd.notna(rev) and pd.notna(price) and float(price) > 0:
            return float(rev) / float(price)
        share = row.get(f"{city}_share_tickets"); tot = row.get("tickets_final")
        if pd.notna(share) and pd.notna(tot):
            return float(tot) * float(share)
        return np.nan
    df["yyc_tickets_final"] = df.apply(lambda r: estimate_city_tickets(r, "yyc"), axis=1)
    df["yeg_tickets_final"] = df.apply(lambda r: estimate_city_tickets(r, "yeg"), axis=1)
    df["season_start_year"] = df.get("season").apply(parse_season_start_year)
    df["bucket"] = df.get("title").astype(str).apply(likeness_bucket)
    df["gender_focus"] = df.get("title").astype(str).apply(gender_focus)
    return df


def build_priors(df: pd.DataFrame, cutoff_year: int, market_key: str) -> pd.DataFrame:
    """Compute cohort priors per (bucket × gender) for given market.
       Uses only rows with season_start_year < cutoff_year. No leakage."""
    use_col = {
        "AB": "tickets_final",
        "Calgary": "yyc_tickets_final",
        "Edmonton": "yeg_tickets_final",
    }[market_key]
    hist = df[(df[use_col].notna()) & (df["season_start_year"].notna()) & (df["season_start_year"] < cutoff_year)].copy()
    if hist.empty:
        return pd.DataFrame(columns=["bucket","gender_focus","prior_mean","prior_std","prior_cv","n_obs"])    
    grp = hist.groupby(["bucket","gender_focus"], dropna=False)[use_col]
    stats = grp.agg(["mean","std","count"]).reset_index()
    stats.columns = ["bucket","gender_focus","prior_mean","prior_std","n_obs"]
    stats["prior_cv"] = stats.apply(lambda r: float(r["prior_std"])/float(r["prior_mean"]) if (pd.notna(r["prior_std"]) and pd.notna(r["prior_mean"]) and r["prior_mean"]>0) else np.nan, axis=1)
    stats["market"] = market_key
    return stats


def score_adapter(fam: float, mot: float) -> float:
    """Turn Familiarity/Motivation (0..100) into a multiplicative nudge around 1.0."""
    # Normalize to 0..1 then center at 0.5 baseline
    f = clamp01((fam or 0.0)/100.0)
    m = clamp01((mot or 0.0)/100.0)
    base = (0.6*f + 0.4*m)  # slightly more weight on familiarity signal
    delta = (base - 0.5)    # -0.5..+0.5
    return 1.0 + SCORE_ADAPTER_STRENGTH * delta  # e.g., ±20% at extremes if strength=0.4


# -------------------------
# SCORING (v8: now returns priors & rule-based forecasts)
# -------------------------

def score_titles(titles, yt_key, sp_id, sp_secret, use_trends, market, segment_name, attrs_df,
                 priors_df: Optional[pd.DataFrame]=None, use_adjusted=True) -> pd.DataFrame:
    # Map title -> attrs from editor
    user_attrs = {}
    for _, row in attrs_df.iterrows():
        nm = str(row["Title"]).strip()
        if not nm: continue
        user_attrs[nm.lower()] = {col: bool(row.get(col, False)) for col in ATTR_COLUMNS}

    trends_scores, wiki_titles, wiki_scores, yt_search_counts, yt_top_view_sqrts, sp_pops = [], [], [], [], [], []
    fam_boosts, mot_boosts = [], []
    buckets, genders = [], []

    for t in titles:
        st.write(f"Fetching metrics for **{t}** …")

        # Trends (optional by market)
        ts = fetch_google_trends_score(t, market=market) if use_trends else 0.0
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

        # Segment boost
        base_attrs = infer_title_attributes(t)
        over = user_attrs.get(t.lower(), {})
        base_attrs.update(over)
        fdelta, mdelta = segment_boosts(base_attrs, segment_name)
        fam_boosts.append(fdelta)
        mot_boosts.append(mdelta)

        # Likeness + Gender (for priors lookup)
        buckets.append(likeness_bucket(t))
        genders.append(gender_focus(t))

    # Familiarity from Wikipedia
    wiki_log = [math.log1p(v) for v in wiki_scores]
    wiki_fam = percentile_normalize(wiki_log)  # 0..100

    # Other sources normalized 0..100
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

        # Apply segment percentage boosts
        fam_adj = fam * (1.0 + fam_boosts[i])
        mot_adj = mot * (1.0 + mot_boosts[i])

        # ---- Historical prior lookup (optional, rule-based)
        prior_mean = prior_cv = prior_n = np.nan
        prior_bucket = buckets[i]
        prior_gender = genders[i]
        if isinstance(priors_df, pd.DataFrame) and not priors_df.empty:
            hit = priors_df[(priors_df["bucket"]==prior_bucket) & (priors_df["gender_focus"]==prior_gender)]
            if hit.empty:
                # fallback to bucket-only prior
                hit = priors_df[priors_df["bucket"]==prior_bucket]
            if not hit.empty:
                row0 = hit.sort_values("n_obs", ascending=False).iloc[0]
                prior_mean = float(row0["prior_mean"]) if pd.notna(row0["prior_mean"]) else np.nan
                prior_cv = float(row0["prior_cv"]) if pd.notna(row0["prior_cv"]) else np.nan
                prior_n = int(row0["n_obs"]) if pd.notna(row0["n_obs"]) else 0

        # Score adapter → multiplicative factor around 1.0
        adapter = score_adapter(fam_adj if use_adjusted else fam, mot_adj if use_adjusted else mot)

        # Forecast: if we have a prior mean, nudge by adapter; otherwise None
        forecast = np.nan
        band_lo = band_hi = np.nan
        if pd.notna(prior_mean):
            forecast = prior_mean * adapter
            # Confidence band from cohort CV; clamp CV to sensible range
            cv = prior_cv if pd.notna(prior_cv) else 0.35
            cv = float(np.clip(cv, 0.05, 0.80))
            band_lo = forecast * (1.0 - cv)
            band_hi = forecast * (1.0 + cv)

        rows.append({
            "Title": t,
            "ResolvedWikiPage": wiki_titles[i],
            "Market": market,
            "TargetSegment": segment_name,
            "LikenessBucket": prior_bucket,
            "GenderFocus": prior_gender,
            "Familiarity": round(fam, 1),
            "Motivation": round(mot, 1),
            "FamiliarityAdj": round(fam_adj, 1),
            "MotivationAdj": round(mot_adj, 1),
            "SegBoostF%": round(fam_boosts[i]*100.0, 1),
            "SegBoostM%": round(mot_boosts[i]*100.0, 1),
            "GoogleTrends": round(trends_n[i], 1),
            "WikipediaRawAvgDaily": round(wiki_scores[i], 2),
            "WikipediaFamiliarity": round(wiki_fam[i], 1),
            "YouTubeN": round(yt_combo[i], 1),
            "SpotifyN": round(sp_n[i], 1),
            # Priors & rule forecast
            "PriorMeanTickets": round(prior_mean, 1) if pd.notna(prior_mean) else np.nan,
            "PriorCV": round(prior_cv, 3) if pd.notna(prior_cv) else np.nan,
            "PriorN": prior_n,
            "ForecastTickets": round(forecast, 0) if pd.notna(forecast) else np.nan,
            "ForecastLo": round(band_lo, 0) if pd.notna(band_lo) else np.nan,
            "ForecastHi": round(band_hi, 0) if pd.notna(band_hi) else np.nan,
        })

    return pd.DataFrame(rows)


def apply_benchmark(df: pd.DataFrame, benchmark_title: str, use_adjusted: bool) -> pd.DataFrame:
    df = df.copy()
    if benchmark_title not in df["Title"].values:
        return df
    b_row = df[df["Title"] == benchmark_title].iloc[0]
    cols = ["GoogleTrends","WikipediaFamiliarity","YouTubeN","SpotifyN"]
    cols = (["FamiliarityAdj","MotivationAdj"] + cols) if use_adjusted else (["Familiarity","Motivation"] + cols)
    for col in cols:
        series = df[col].astype(float)
        if series.std(ddof=0) < 1e-6:
            continue
        bench = float(b_row[col])
        if bench and abs(bench) > 1e-9:
            df[col] = (series / bench) * 100.0
    return df

# -------------------------
# PLOTTING & PDF (unchanged)
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
        tab_cols = ["Title","Market","TargetSegment","LikenessBucket","GenderFocus",fx,fy,
                    "SegBoostF%","SegBoostM%","GoogleTrends","WikipediaFamiliarity","YouTubeN","SpotifyN",
                    "PriorMeanTickets","PriorCV","PriorN","ForecastTickets","ForecastLo","ForecastHi","ResolvedWikiPage"]
        present = [c for c in tab_cols if c in df.columns]
        tab = plt.table(cellText=df[present].round(1).values, colLabels=present, loc='center')
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
            "Rule-based forecast uses historical cohort priors by Likeness×Gender, "
            "nudged by Familiarity/Motivation. Confidence bands derive from cohort CV.\n"
            "Priors exclude pandemic season and use only seasons before the chosen cutoff (no leakage)."
        )
        plt.text(0.01, 0.99, memo, va='top')
        pdf.savefig(fig5, bbox_inches='tight'); plt.close(fig5)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Viability", layout="wide")
st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v8)")

st.markdown(
    "Market-aware Trends + robust Wikipedia. **Historical priors (no-ML)** add realism: "
    "cohort means by Likeness × Gender × Market, with confidence bands from cohort CV. "
    "All priors exclude pandemic seasons and respect your cutoff season."
)

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

with st.expander("📥 Historical Data (for priors)"):
    st.caption("Upload the latest cleaned dataset (with columns season, title, total_tickets/total_revenue, prices, pandemic_season, YYC/YEG splits). Pandemic seasons are dropped automatically.")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
    cutoff_help = "Seasons strictly **before** this year feed priors (prevents leakage). Example: if cutoff=2025, 2024/25 counts as 2024 and is included."
    cutoff_year = st.number_input("Cutoff start year for priors", min_value=2010, max_value=2100, value=2025, help=cutoff_help)

with st.expander("⚙️ Options"):
    use_trends = st.checkbox("Use Google Trends", value=True)
    market = st.selectbox("Market", ["Alberta (province)", "Calgary (city)", "Edmonton (city)"], index=0)
    market_key = {"Alberta (province)":"AB","Calgary (city)":"Calgary","Edmonton (city)":"Edmonton"}[market]
    segment_name = st.selectbox("Target Segment (applies demographic boosts)", list(SEGMENT_RULES.keys()), index=0)
    use_adjusted = st.checkbox("Use Segment-Adjusted scores in charts/exports", value=True)
    use_priors = st.checkbox("Use Historical Priors (rule-based forecast)", value=True)
    st.caption("If city data is sparse, Trends falls back to Alberta → Canada → Global automatically.")

# Default titles
default_titles = [
    "The Nutcracker","Sleeping Beauty","Cinderella","Pinocchio",
    "The Merry Widow","The Hunchback of Notre Dame","Frozen",
    "Beauty and the Beast","Alice in Wonderland","Peter Pan",
    "Romeo and Juliet","Swan Lake","Don Quixote","Contemporary Composers",
    "Nijinsky","Notre Dame de Paris","Wizard of Oz","Grimm"
]

titles_input = st.text_area("Enter titles (one per line):", value="\n".join(default_titles), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

# Title attributes editor
st.subheader("Title Attributes (edit to fit your production/angle)")
attr_rows = []
for t in titles:
    inferred = infer_title_attributes(t)
    row = {"Title": t}
    row.update(inferred)
    attr_rows.append(row)
attrs_df = pd.DataFrame(attr_rows, columns=["Title"]+ATTR_COLUMNS)
attrs_df = st.data_editor(attrs_df, use_container_width=True, hide_index=True)

colA, colB, colC, colD, colE = st.columns(5)
with colA:
    do_benchmark = st.checkbox("Normalize to a benchmark title?")
with colB:
    benchmark_title = st.selectbox("Benchmark title", options=titles, index=0 if "The Nutcracker" in titles else 0)
with colC:
    run_btn = st.button("Score Titles", type="primary")
with colD:
    gen_pdf = st.checkbox("Prepare PDF after scoring", value=False)
with colE:
    st.caption("v8 uses **priors × score adapter** for forecasts; download CSV for full columns.")

priors_df = None
if uploaded is not None:
    try:
        raw_df = pd.read_excel(uploaded) if uploaded.name.endswith((".xlsx",".xls")) else pd.read_csv(uploaded)
        hist_df = build_tickets_columns(raw_df)
        priors_df_AB = build_priors(hist_df, cutoff_year, "AB")
        priors_df_YYC = build_priors(hist_df, cutoff_year, "Calgary")
        priors_df_YEG = build_priors(hist_df, cutoff_year, "Edmonton")
        priors_df = pd.concat([priors_df_AB, priors_df_YYC, priors_df_YEG], ignore_index=True)
        # Keep only selected market's priors for lookup
        priors_df = priors_df[priors_df["market"]==market_key].copy()
        st.success(f"Priors ready for {market}: {len(priors_df)} cohorts.")
        st.dataframe(priors_df.sort_values(["bucket","gender_focus","n_obs"], ascending=[True,True,False]), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load or build priors: {e}")

if run_btn:
    with st.spinner("Scoring titles…"):
        df = score_titles(
            titles,
            yt_key.strip() or None,
            sp_id.strip() or None,
            sp_secret.strip() or None,
            use_trends,
            market_key,
            segment_name,
            attrs_df,
            priors_df if (use_priors and priors_df is not None) else None,
            use_adjusted=use_adjusted,
        )

        # Choose which columns to sort & visualize
        sort_x, sort_y = ("FamiliarityAdj","MotivationAdj") if use_adjusted else ("Familiarity","Motivation")
        df = df.sort_values(by=[sort_y, sort_x], ascending=False)

        if do_benchmark and benchmark_title:
            df = apply_benchmark(df, benchmark_title, use_adjusted)

        st.success("Done.")
        st.dataframe(df, use_container_width=True)

        # Diagnostics — dispersion
        st.subheader("Diagnostics")
        diag_cols = [sort_x, sort_y, "GoogleTrends","WikipediaRawAvgDaily","WikipediaFamiliarity","YouTubeN","SpotifyN"]
        if "ForecastTickets" in df.columns:
            diag_cols += ["PriorMeanTickets","PriorCV","PriorN","ForecastTickets"]
        present = [c for c in diag_cols if c in df.columns]
        diag = pd.DataFrame({
            "metric": present,
            "min":    [float(df[c].min()) for c in present],
            "max":    [float(df[c].max()) for c in present],
            "std":    [float(df[c].std(ddof=0)) for c in present],
        })
        st.dataframe(diag, use_container_width=True)

        # Charts
        st.subheader("Charts")
        figQ = quadrant_plot(df, sort_x, sort_y, f"{sort_x} vs {sort_y} (Quadrant Map)")
        st.pyplot(figQ)
        if "ForecastTickets" in df.columns and df["ForecastTickets"].notna().any():
            figF = bar_chart(df.fillna({"ForecastTickets":0}), "ForecastTickets", "Forecast Tickets (rule-based)")
            st.pyplot(figF)

        # CSV / PDF
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "title_scores_v8.csv", "text/csv")
        if gen_pdf:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = f"title_scores_brief_{ts}.pdf"
            generate_pdf_brief(df, use_adjusted, pdf_path)
            st.success("PDF created.")
            st.download_button("⬇️ Download PDF Brief", data=open(pdf_path, "rb").read(), file_name=pdf_path, mime="application/pdf")
