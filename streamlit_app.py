# streamlit_app_v9_new_titles_fix.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 Test, New Titles FIX)
# - Hard-coded baselines (normalized to a user-selected benchmark = 100)
# - Add NEW titles via text area
# - "Score Titles" button triggers scoring
# - Unknown titles: live fetch (if enabled) OR offline estimate
# - Segment + Region multipliers; charts + CSV

import math, time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Optional APIs used only when "Use Live Data" is ON
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
# PAGE / SPLASH
# -------------------------
st.set_page_config(page_title="Alberta Ballet — Title Familiarity & Motivation Scorer", layout="wide")
with st.spinner("🎭 Preparing Alberta Ballet Familiarity Baselines..."):
    time.sleep(1.0)

st.title("🎭 Alberta Ballet — Title Familiarity & Motivation Scorer (v9 Test — New Titles Enabled)")
st.caption("Hard-coded Alberta-wide baselines (normalized to your selected benchmark = 100). Add new titles; choose live fetch or offline estimate.")

# -------------------------
# METHODOLOGY & GLOSSARY SECTION
# -------------------------
with st.expander("📘 About This App — Methodology & Glossary"):
    st.markdown("""
### **Purpose**
This tool helps Alberta Ballet estimate how familiar and motivated audiences are likely to be toward specific productions, both current and prospective.  
It combines cultural familiarity data, historical sales performance, and demographic patterns derived from *Spotlight on Arts Audiences* and Alberta Ballet’s 2017–2025 production results.

### **Methodology Overview**
- **Baselines:** Each existing title is assigned familiarity and motivation scores derived from a combination of:
  - Wikipedia page views (public awareness)
  - Google Trends (search interest)
  - YouTube activity (media engagement)
  - Spotify popularity (musical familiarity)
  - Adjusted for Alberta-specific performance data and demographic trends.
- **Normalization:** All titles are scaled relative to the **benchmark title you select** (index = 100).  
  Scores above 100 represent titles with greater recognition or motivation potential under the current segment/region; below 100 suggests lesser-known works.
- **Segments:** Audience segments apply different weightings:
  - *Core Classical (F35–64):* Prioritizes classic ballets and female or co-leads.
  - *Family (Parents w/ kids):* Boosts family-oriented, accessible stories and pop IPs.
  - *Emerging Adults (18–34):* Emphasizes contemporary or visually bold productions.
- **Regions:** Regional multipliers reflect audience variation across Alberta:
  - *Calgary:* Slightly higher affinity for classical and family titles.
  - *Edmonton:* Slightly stronger interest in contemporary or conceptual programs.
- **Unknown Titles:**  
  - If *Use Live Data* is enabled and API keys are provided, the app fetches real data from Wikipedia, YouTube, Spotify, and heuristic Trends.  
  - Otherwise, it estimates based on category medians and historical genre performance.
- **Benchmarking:** Familiarity and motivation are expressed as a percentage relative to the **currently selected** benchmark, given the chosen segment and region.

### **Interpretation**
- **Familiarity** reflects brand or story awareness (e.g., “Have people heard of it?”).  
- **Motivation** reflects the likelihood of ticket purchase interest (e.g., “Would they go see it?”).  
- Both indicators are best used for comparing titles *against each other* rather than as absolute predictions.

---

### **Glossary of Terms**
| Term | Definition |
|------|-------------|
| **Baseline** | Hard-coded familiarity and motivation indices for existing repertoire. |
| **Familiarity** | Relative measure of audience awareness based on Wikipedia, Trends, and Spotify indices. |
| **Motivation** | Relative measure of interest and engagement potential based on YouTube, Trends, and Wiki data. |
| **Segment** | Target demographic lens (Core Classical, Family, Emerging Adults, or General). Adjusts score weighting. |
| **Region** | Market context (Province-wide default, Calgary, or Edmonton). Adjusts for local preferences. |
| **Pop IP** | “Popular Intellectual Property” — stories known through film, literature, or franchise (e.g., Bowie, Joni, Bollywood). |
| **Contemporary** | Non-narrative or modern programs, often concept-driven or physically abstract. |
| **Normalization** | Scaling process that expresses scores as a ratio to the **selected** benchmark title (index = 100). |
| **Live Fetch** | Optional real-time retrieval from Wikipedia, YouTube, Spotify for new titles. |

**Note:** This tool is not a predictor of exact ticket sales but a comparative instrument for programming strategy and communication alignment.
    """)

# -------------------------
# BASELINE DATA (subset for test run)
# Values are indices relative to a general baseline (not capped).
BASELINES = {
    "Cinderella": {"wiki": 88, "trends": 80, "youtube": 82, "spotify": 80, "category": "family_classic", "gender": "female"},
    "Swan Lake": {"wiki": 95, "trends": 90, "youtube": 88, "spotify": 84, "category": "classic_romance", "gender": "female"},
    "Sleeping Beauty": {"wiki": 92, "trends": 85, "youtube": 78, "spotify": 74, "category": "classic_romance", "gender": "female"},
    "Hansel & Gretel": {"wiki": 78, "trends": 70, "youtube": 65, "spotify": 62, "category": "family_classic", "gender": "co"},
    "Don Quixote": {"wiki": 88, "trends": 75, "youtube": 72, "spotify": 68, "category": "classic_comedy", "gender": "male"},
    "Giselle": {"wiki": 82, "trends": 72, "youtube": 65, "spotify": 60, "category": "classic_romance", "gender": "female"},
    "La Sylphide": {"wiki": 75, "trends": 68, "youtube": 60, "spotify": 55, "category": "classic_romance", "gender": "female"},
    "Beauty and the Beast": {"wiki": 94, "trends": 97, "youtube": 92, "spotify": 90, "category": "family_classic", "gender": "female"},
    "Romeo and Juliet": {"wiki": 90, "trends": 82, "youtube": 79, "spotify": 77, "category": "romantic_tragedy", "gender": "co"},
    "The Merry Widow": {"wiki": 70, "trends": 60, "youtube": 55, "spotify": 50, "category": "romantic_comedy", "gender": "female"},
    "Peter Pan": {"wiki": 80, "trends": 78, "youtube": 85, "spotify": 82, "category": "family_classic", "gender": "male"},
    "Pinocchio": {"wiki": 72, "trends": 68, "youtube": 70, "spotify": 66, "category": "family_classic", "gender": "male"},
    "Grimm": {"wiki": 55, "trends": 52, "youtube": 50, "spotify": 45, "category": "contemporary", "gender": "na"},
    "Momix": {"wiki": 65, "trends": 60, "youtube": 68, "spotify": 60, "category": "contemporary", "gender": "na"},
    "Dangerous Liaisons": {"wiki": 66, "trends": 62, "youtube": 63, "spotify": 58, "category": "dramatic", "gender": "female"},
    "Frankenstein": {"wiki": 68, "trends": 63, "youtube": 66, "spotify": 62, "category": "dramatic", "gender": "male"},
    "Ballet Boyz": {"wiki": 45, "trends": 40, "youtube": 60, "spotify": 58, "category": "contemporary", "gender": "male"},
    "Contemporary Classical": {"wiki": 60, "trends": 55, "youtube": 58, "spotify": 50, "category": "contemporary", "gender": "na"},
    "Ballet BC": {"wiki": 50, "trends": 45, "youtube": 53, "spotify": 49, "category": "contemporary", "gender": "na"},
    "Complexions": {"wiki": 62, "trends": 58, "youtube": 66, "spotify": 60, "category": "contemporary", "gender": "na"},
    "Phi – David Bowie": {"wiki": 70, "trends": 65, "youtube": 72, "spotify": 75, "category": "pop_ip", "gender": "male"},
    "Tragically Hip – All of Us": {"wiki": 72, "trends": 68, "youtube": 78, "spotify": 80, "category": "pop_ip", "gender": "male"},
    "Harlem (Dance Theatre)": {"wiki": 75, "trends": 68, "youtube": 74, "spotify": 68, "category": "pop_ip", "gender": "co"},
    "Shaping Sound": {"wiki": 68, "trends": 64, "youtube": 70, "spotify": 66, "category": "contemporary", "gender": "co"},
    "Taj Express": {"wiki": 66, "trends": 62, "youtube": 68, "spotify": 70, "category": "pop_ip", "gender": "male"},
    "Diavolo": {"wiki": 60, "trends": 58, "youtube": 66, "spotify": 64, "category": "contemporary", "gender": "co"},
    "Unleashed – Mixed Bill": {"wiki": 55, "trends": 50, "youtube": 60, "spotify": 52, "category": "contemporary", "gender": "co"},
    "Botero": {"wiki": 58, "trends": 54, "youtube": 62, "spotify": 57, "category": "contemporary", "gender": "male"},
    "Away We Go – Mixed Bill": {"wiki": 54, "trends": 52, "youtube": 58, "spotify": 50, "category": "contemporary", "gender": "co"},
    "Fiddle – Joni Mitchell": {"wiki": 60, "trends": 55, "youtube": 62, "spotify": 66, "category": "pop_ip", "gender": "female"},
    "Midsummer Night’s Dream": {"wiki": 75, "trends": 70, "youtube": 70, "spotify": 68, "category": "classic_romance", "gender": "co"},
    "Dracula": {"wiki": 74, "trends": 65, "youtube": 70, "spotify": 65, "category": "romantic_tragedy", "gender": "male"},
}

# -------------------------
# SEGMENTS & REGIONS
# -------------------------
SEGMENT_MULT = {
    "General Population": {"female": 1.00, "male": 1.00, "co": 1.00, "na": 1.00,
                           "family_classic": 1.00, "classic_romance": 1.00, "romantic_tragedy": 1.00,
                           "classic_comedy": 1.00, "contemporary": 1.00, "pop_ip": 1.00, "dramatic": 1.00},
    "Core Classical (F35–64)": {"female": 1.12, "male": 0.95, "co": 1.05, "na": 1.00,
                                 "family_classic": 1.10, "classic_romance": 1.08, "romantic_tragedy": 1.05,
                                 "classic_comedy": 1.02, "contemporary": 0.90, "pop_ip": 1.00, "dramatic": 1.00},
    "Family (Parents w/ kids)": {"female": 1.10, "male": 0.92, "co": 1.06, "na": 1.00,
                                  "family_classic": 1.18, "classic_romance": 0.95, "romantic_tragedy": 0.85,
                                  "classic_comedy": 1.05, "contemporary": 0.82, "pop_ip": 1.20, "dramatic": 0.90},
    "Emerging Adults (18–34)": {"female": 1.02, "male": 1.02, "co": 1.00, "na": 1.00,
                                 "family_classic": 0.95, "classic_romance": 0.92, "romantic_tragedy": 0.90,
                                 "classic_comedy": 0.98, "contemporary": 1.25, "pop_ip": 1.15, "dramatic": 1.05},
}
REGION_MULT = {"Province": 1.00, "Calgary": 1.05, "Edmonton": 0.95}

# -------------------------
# HEURISTICS for OFFLINE estimation of NEW titles
# -------------------------
def infer_gender_and_category(title: str) -> Tuple[str, str]:
    t = title.lower()
    gender = "na"
    female_keys = ["cinderella","sleeping","beauty and the beast","beauty","giselle","swan","widow","alice","juliet","sylphide"]
    male_keys = ["pinocchio","peter pan","don quixote","hunchback","hamlet","frankenstein","romeo","nijinsky"]
    if "romeo" in t and "juliet" in t: gender = "co"
    elif any(k in t for k in female_keys): gender = "female"
    elif any(k in t for k in male_keys): gender = "male"

    if any(k in t for k in ["wizard","peter pan","pinocchio","hansel","frozen","beauty","alice"]):
        cat = "family_classic"
    elif any(k in t for k in ["swan","sleeping","cinderella","giselle","sylphide"]):
        cat = "classic_romance"
    elif any(k in t for k in ["romeo","hunchback","notre dame","hamlet","frankenstein"]):
        cat = "romantic_tragedy"
    elif any(k in t for k in ["don quixote","merry widow"]):
        cat = "classic_comedy"
    elif any(k in t for k in ["contemporary","boyz","ballet boyz","momix","complexions","grimm","nijinsky","shadowland","deviate","phi"]):
        cat = "contemporary"
    elif any(k in t for k in ["taj","tango","harlem","tragically hip","l cohen","leonard cohen"]):
        cat = "pop_ip"
    else:
        cat = "dramatic"
    return gender, cat

def estimate_unknown_title(title: str) -> Dict[str, float | str]:
    gender, category = infer_gender_and_category(title)
    base_df = pd.DataFrame(BASELINES).T
    for k in ["wiki","trends","youtube","spotify"]:
        if k not in base_df.columns:
            base_df[k] = np.nan
    cats = {k: v["category"] for k, v in BASELINES.items()}
    tmp = base_df.copy()
    tmp["category"] = tmp.index.map(lambda x: cats.get(x, "dramatic"))
    cat_df = tmp[tmp["category"] == category]
    med = (cat_df[["wiki","trends","youtube","spotify"]].median()
           if not cat_df.empty else tmp[["wiki","trends","youtube","spotify"]].median())
    gender_adj = {"female": 1.05, "male": 0.98, "co": 1.03, "na": 1.00}[gender]
    est = {k: float(med[k] * gender_adj) for k in ["wiki","trends","youtube","spotify"]}
    for k in est: est[k] = float(max(30.0, min(160.0, est[k])))
    return {"wiki": est["wiki"], "trends": est["trends"], "youtube": est["youtube"], "spotify": est["spotify"],
            "gender": gender, "category": category}

# -------------------------
# OPTIONAL LIVE FETCHERS (only if toggled ON)
# -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGEVIEW = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{start}/{end}"

def wiki_search_best_title(query: str) -> Optional[str]:
    try:
        params = {"action":"query","list":"search","srsearch":query,"format":"json","srlimit": 5}
        r = requests.get(WIKI_API, params=params, timeout=10)
        if r.status_code != 200: return None
        items = r.json().get("query",{}).get("search",[])
        return items[0]["title"] if items else None
    except Exception:
        return None

def fetch_wikipedia_views_for_page(page_title: str) -> float:
    try:
        end = datetime.utcnow().strftime("%Y%m%d")
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y%m%d")
        url = WIKI_PAGEVIEW.format(page=page_title.replace(" ","_"), start=start, end=end)
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return 0.0
        items = r.json().get("items",[])
        views = [it.get("views",0) for it in items]
        return (sum(views)/365.0) if views else 0.0
    except Exception:
        return 0.0

def fetch_live_for_unknown(title: str, yt_key: Optional[str], sp_id: Optional[str], sp_secret: Optional[str]) -> Dict[str, float | str]:
    # Wikipedia → index-ish transform
    w_title = wiki_search_best_title(title) or title
    wiki_raw = fetch_wikipedia_views_for_page(w_title)
    wiki_idx = 40.0 + min(110.0, (math.log1p(max(0.0, wiki_raw)) * 20.0))  # ~40..150

    # Trends: resilient heuristic (kept local to avoid pytrends dependency here)
    trends_idx = 60.0 + (len(title) % 40)

    # YouTube (optional)
    yt_idx = 0.0
    if yt_key and build is not None:
        try:
            yt = build("youtube","v3",developerKey=yt_key)
            search = yt.search().list(q=title, part="id", type="video", maxResults=25).execute()
            ids = [it["id"]["videoId"] for it in search.get("items",[])]
            if ids:
                stats = yt.videos().list(part="statistics", id=",".join(ids[:50])).execute()
                views = [int(i.get("statistics",{}).get("viewCount",0)) for i in stats.get("items",[])]
                yt_idx = float(np.log1p(max(views)) * 12.0)  # map to ~50..140
        except Exception:
            yt_idx = 0.0
    if yt_idx == 0.0:
        yt_idx = 55.0 + (len(title) * 1.2) % 45.0

    # Spotify (optional)
    sp_idx = 0.0
    if sp_id and sp_secret and spotipy is not None:
        try:
            auth = SpotifyClientCredentials(client_id=sp_id, client_secret=sp_secret)
            sp = spotipy.Spotify(auth_manager=auth)
            res = sp.search(q=title, type="track,album", limit=10)
            pops = [t.get("popularity",0) for t in res.get("tracks",{}).get("items",[])]
            sp_idx = float(np.percentile(pops, 80)) if pops else 0.0
        except Exception:
            sp_idx = 0.0
    if sp_idx == 0.0:
        sp_idx = 50.0 + (len(title) * 1.7) % 40.0

    gender, category = infer_gender_and_category(title)
    return {"wiki": wiki_idx, "trends": trends_idx, "youtube": yt_idx, "spotify": sp_idx,
            "gender": gender, "category": category}

# -------------------------
# UI — API + OPTIONS + TITLES
# -------------------------
with st.expander("🔑 API Configuration (only used for NEW titles if enabled)"):
    yt_key = st.text_input("YouTube Data API v3 Key", type="password")
    sp_id = st.text_input("Spotify Client ID", type="password")
    sp_secret = st.text_input("Spotify Client Secret", type="password")
    use_live = st.checkbox("Use Live Data for Unknown Titles", value=False)

region = st.selectbox("Region", ["Province","Calgary","Edmonton"], index=0)
segment = st.selectbox("Audience Segment", list(SEGMENT_MULT.keys()), index=0)

default_list = list(BASELINES.keys())[:50]
st.markdown("**Titles to score** (one per line). Add NEW titles freely:")
titles_input = st.text_area("Enter titles", value="\n".join(default_list), height=220)
titles = [t.strip() for t in titles_input.splitlines() if t.strip()]

run = st.button("Score Titles", type="primary")

# -------------------------
# SCORING
# -------------------------
def calc_scores(entry: Dict[str, float | str], seg_key: str, reg_key: str) -> Tuple[float,float]:
    gender = entry["gender"]; cat = entry["category"]
    fam = entry["wiki"] * 0.55 + entry["trends"] * 0.30 + entry["spotify"] * 0.15
    mot = entry["youtube"] * 0.45 + entry["trends"] * 0.25 + entry["spotify"] * 0.15 + entry["wiki"] * 0.15
    seg = SEGMENT_MULT[seg_key]
    fam *= seg.get(gender,1.0) * seg.get(cat,1.0)
    mot *= seg.get(gender,1.0) * seg.get(cat,1.0)
    fam *= REGION_MULT[reg_key]
    mot *= REGION_MULT[reg_key]
    return fam, mot

# --- Hard-coded ticket priors (lists -> we use median to be robust) ---
TICKET_PRIORS_RAW = {
    "Alice in Wonderland": [11216],
    "All of Us - Tragically Hip": [15488],
    "Away We Go - Mixed Bill": [4649],
    "Ballet BC": [4013],
    "Ballet Boyz": [7401],
    "BJM - Leonard Cohen": [7819],
    "Botero": [5460],
    "Cinderella": [16304],
    "Complexions - Lenny Kravitz": [7096],
    "Dance Theatre of Harlem": [7269],
    "Dangerous Liaisons": [6875],
    "deViate - Mixed Bill": [5144],
    "Diavolo": [10673],
    "Don Quixote": [5650],
    "Dona Peron": [5221],
    "Dracula": [11285],
    "Fiddle & the Drum - Joni Mitchell": [6024],
    "Frankenstein": [10470],
    "Giselle": [9111],
    "Grimm": [6362],
    "Handmaid's Tale": [6842],
    "Hansel & Gretel": [7290],
    "La Sylphide": [5221],
    "Midsummer Night's Dream": [6587],
    "Momix": [8391],
    "Our Canada": [10138],
    "Phi - David Bowie": [12336],
    "Shaping Sound": [10208],
    "Sleeping Beauty": [9596.5],
    "Swan Lake": [13157],
    "Taj Express": [10314],
    "Tango Fire": [7251],
    "Trockadero": [5476],
    "Unleashed - Mixed Bill": [5221],
    "Winter Gala": [1321],
    "Wizard of Oz": [8468],
}

def _median(xs):
    xs = sorted([float(x) for x in xs if x is not None])
    if not xs: return None
    n = len(xs)
    mid = n // 2
    return (xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0)

# Weight for blending historic tickets with signal composite
TICKET_BLEND_WEIGHT = 0.50  # 50% tickets, 50% familiarity/motivation composite

if run:
    if not titles:
        st.warning("Add at least one title to score.")
    else:
        rows = []
        unknown_used_live, unknown_used_est = [], []

        # 1) Build base rows (INCLUDING FamiliarityRaw/MotivationRaw) BEFORE any normalization
        for title in titles:
            if title in BASELINES:
                entry = BASELINES[title]
                src = "Baseline"
            else:
                if use_live:
                    entry = fetch_live_for_unknown(title, yt_key, sp_id, sp_secret)
                    src = "Live"
                    unknown_used_live.append(title)
                else:
                    entry = estimate_unknown_title(title)
                    src = "Estimated"
                    unknown_used_est.append(title)

            fam_raw, mot_raw = calc_scores(entry, segment, region)

            rows.append({
                "Title": title,
                "Region": region,
                "Segment": segment,
                "Gender": entry["gender"],
                "Category": entry["category"],
                "FamiliarityRaw": fam_raw,
                "MotivationRaw": mot_raw,
                "WikiIdx": entry["wiki"],
                "TrendsIdx": entry["trends"],
                "YouTubeIdx": entry["youtube"],
                "SpotifyIdx": entry["spotify"],
                "Source": src
            })

        # 2) Create the DataFrame NOW (so df exists for everything else)
        df = pd.DataFrame(rows)

        # 3) Pick benchmark and normalize signals
        benchmark_title = st.selectbox(
            "Choose Benchmark Title for Normalization",
            options=list(BASELINES.keys()),
            index=0
        )
        bench_entry = BASELINES[benchmark_title]
        bench_fam_raw, bench_mot_raw = calc_scores(bench_entry, segment, region)
        bench_fam_raw = bench_fam_raw or 1.0
        bench_mot_raw = bench_mot_raw or 1.0

        df["Familiarity"] = (df["FamiliarityRaw"] / bench_fam_raw) * 100.0
        df["Motivation"]  = (df["MotivationRaw"]  / bench_mot_raw)  * 100.0
        st.caption(f"Scores normalized to benchmark: **{benchmark_title}**")

        # 4) Ticket medians and ticket indices (after benchmark chosen)
        def _median(xs):
            xs = sorted([float(x) for x in xs if x is not None])
            if not xs: return None
            n = len(xs); mid = n // 2
            return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2.0

        TICKET_MEDIANS = {k: _median(v) for k, v in TICKET_PRIORS_RAW.items()}
        BENCHMARK_TICKET_MEDIAN = TICKET_MEDIANS.get(benchmark_title, None) or 1.0

        def ticket_index_for_title(title: str) -> Tuple[Optional[float], Optional[float]]:
            t = title.strip()
            aliases = {"Handmaid’s Tale": "Handmaid's Tale"}  # punctuation variant
            key = aliases.get(t, t)
            med = TICKET_MEDIANS.get(key)
            if med:
                idx = (med / BENCHMARK_TICKET_MEDIAN) * 100.0
                return float(med), float(idx)
            return None, None

        ticket_medians, ticket_indices = [], []
        for t in df["Title"]:
            med, idx = ticket_index_for_title(t)
            ticket_medians.append(med)
            ticket_indices.append(idx)
        df["TicketMedian"] = ticket_medians
        df["TicketIndex"]  = ticket_indices

        # 5) Info badges for unknowns
        if unknown_used_est:
            st.info(f"Estimated (offline) for new titles: {', '.join(unknown_used_est)}")
        if unknown_used_live:
            st.success(f"Used LIVE data for new titles: {', '.join(unknown_used_live)}")

        # 6) Build composites
        TICKET_BLEND_WEIGHT = 0.50  # 50/50 blend
        signal_composite = df[["Familiarity", "Motivation"]].mean(axis=1)
        tickets_component = df["TicketIndex"].fillna(signal_composite)
        df["Composite"] = (1.0 - TICKET_BLEND_WEIGHT) * signal_composite + TICKET_BLEND_WEIGHT * tickets_component

        # 7) === Comparison & calibration expander ===
        # ======================
        # Plain-English comparison & calibration
        # ======================
        with st.expander("🧪 How close are the online-signal scores to your real ticket results?"):
            # 1) Build the comparison columns
            df["SignalOnly"] = signal_composite
            df["Blended"] = df["Composite"]
            df["DeltaAbs"] = df["Blended"] - df["SignalOnly"]
            df["DeltaPct"] = (df["Blended"] / df["SignalOnly"] - 1.0) * 100.0

            df_known = df[df["TicketIndex"].notna()].copy()
            if df_known.empty:
                st.info("No titles in your list have past ticket history to compare against. Add some known titles to see how close the online signals are.")
            else:
                # How similar are online-only scores to ticket history?
                corr = float(df_known[["SignalOnly","TicketIndex"]].corr().iloc[0,1])
        
                # Average miss (in index points) if we used online-only scores
                resid_raw = df_known["TicketIndex"] - df_known["SignalOnly"]
                rmse_raw = float(np.sqrt(np.mean(resid_raw**2)))
        
                # Optional adjustment: nudge online-only scores to better match history
                do_calibrate = st.checkbox(
                    "Adjust the online-signal scores to better match past ticket results (optional)",
                    value=False
                )
        
                a = b = None
                if do_calibrate and len(df_known) >= 2:
                    x = df_known["SignalOnly"].values
                    y = df_known["TicketIndex"].values
                    a, b = np.polyfit(x, y, 1)  # simple straight-line adjustment
                    df_known["SignalCalibrated"] = a * df_known["SignalOnly"] + b
                    resid_cal = df_known["TicketIndex"] - df_known["SignalCalibrated"]
                    rmse_cal = float(np.sqrt(np.mean(resid_cal**2)))
                else:
                    rmse_cal = None
        
                # Plain-English headlines
                st.markdown("### What this means, in plain English")
                st.markdown(
                    f"- **Overall similarity:** `{corr:.2f}` — closer to **1.00** means the online signals move up/down like ticket results do.\n"
                    f"- **Average miss using online-only scores:** about **{rmse_raw:.1f} index points**.\n"
                    + (f"- **After adjustment:** average miss improves to **{rmse_cal:.1f} index points**." if rmse_cal is not None else "")
                )
                st.caption(
                    "“Index points” are the same units you see in the tables and charts (your chosen benchmark = 100). "
                    "For example, a 15-point miss means online signals were off by ~15 on that 0–150ish scale."
                )
        
                # Where signals tend to over/under-estimate (by category)
                df_known["OverUnder"] = np.where(df_known["DeltaAbs"] > 0, "Blended higher than signals", "Blended lower than signals")
                by_cat = (
                    df_known.groupby("Category")[["SignalOnly","TicketIndex","DeltaAbs","DeltaPct"]]
                    .agg({
                        "SignalOnly":"mean",
                        "TicketIndex":"mean",
                        "DeltaAbs":"mean",
                        "DeltaPct":"mean",
                    })
                    .rename(columns={
                        "SignalOnly":"Avg online-only score",
                        "TicketIndex":"Avg ticket-based score",
                        "DeltaAbs":"Avg difference (points)",
                        "DeltaPct":"Avg difference (%)"
                    })
                    .sort_values("Avg difference (points)", ascending=False)
                )
        
                st.markdown("### Where the model tends to be off (by category)")
                st.caption("Positive numbers = ticket-informed scores run higher than online-only; negative = they run lower.")
                st.dataframe(
                    by_cat.style.format({
                        "Avg online-only score":"{:.1f}",
                        "Avg ticket-based score":"{:.1f}",
                        "Avg difference (points)":"{:+.1f}",
                        "Avg difference (%)":"{:+.1f}%"
                    }),
                    use_container_width=True
                )
        
                # Simple comparison table for known titles
                show_cols = [
                    "Title","Category","Region","Segment",
                    "SignalOnly","TicketIndex","Blended","DeltaAbs","DeltaPct","Source"
                ]
                show_cols = [c for c in show_cols if c in df_known.columns]
                st.markdown("### Titles we can compare (we have ticket history for these)")
                st.dataframe(
                    df_known[show_cols]
                        .sort_values(by=["DeltaAbs","Blended","SignalOnly"], ascending=[False, False, False])
                        .rename(columns={
                            "SignalOnly":"Online-only score",
                            "TicketIndex":"Ticket-based score",
                            "Blended":"Score used (blended)",
                            "DeltaAbs":"How much tickets moved it (pts)",
                            "DeltaPct":"How much tickets moved it (%)"
                        })
                        .style.format({
                            "Online-only score":"{:.1f}",
                            "Ticket-based score":"{:.1f}",
                            "Score used (blended)":"{:.1f}",
                            "How much tickets moved it (pts)":"{:+.1f}",
                            "How much tickets moved it (%)":"{:+.1f}%"
                        }),
                    use_container_width=True
                )
        
                # Visual: how far off the signals are, per title
                fig_res, ax_res = plt.subplots()
                ax_res.scatter(df_known["SignalOnly"], df_known["TicketIndex"] - df_known["SignalOnly"])
                ax_res.axhline(0, linestyle="--")
                ax_res.set_xlabel("Online-only score (index)")
                ax_res.set_ylabel("Ticket-based minus online-only (points)")
                ax_res.set_title("How much ticket results shift the score (for titles with history)")
                st.pyplot(fig_res)

        # 8) Assign letter grades
        def _assign_score(v: float) -> str:
            if v >= 90: return "A"
            elif v >= 75: return "B"
            elif v >= 60: return "C"
            elif v >= 45: return "D"
            else: return "E"
        df["Score"] = df["Composite"].apply(_assign_score)

        # 9) Display + charts + download
        display_cols = [
            "Title","Region","Segment","Gender","Category",
            "Familiarity","Motivation","TicketMedian","TicketIndex","Composite","Score",
            "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx","Source"
        ]
        existing = [c for c in display_cols if c in df.columns]

        if "Score" in df.columns:
            grade_counts = df["Score"].value_counts().reindex(["A","B","C","D","E"]).fillna(0).astype(int)
            st.caption(
                f"Grade distribution — A:{grade_counts['A']}  B:{grade_counts['B']}  "
                f"C:{grade_counts['C']}  D:{grade_counts['D']}  E:{grade_counts['E']}"
            )

        st.dataframe(
            df[existing]
              .sort_values(by=["Composite","Motivation","Familiarity"], ascending=[False, False, False])
              .style.map(
                  lambda v: (
                      "color: green;" if v == "A" else
                      "color: darkgreen;" if v == "B" else
                      "color: orange;" if v == "C" else
                      "color: darkorange;" if v == "D" else
                      "color: red;" if v == "E" else ""
                  ),
                  subset=["Score"] if "Score" in df.columns else []
              ),
            use_container_width=True
        )

        fig, ax = plt.subplots()
        ax.scatter(df["Familiarity"], df["Motivation"])
        for _, r in df.iterrows():
            ax.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8)
        ax.axvline(df["Familiarity"].median(), color='gray', linestyle='--')
        ax.axhline(df["Motivation"].median(), color='gray', linestyle='--')
        ax.set_xlabel(f"Familiarity ({benchmark_title} = 100 index)")
        ax.set_ylabel(f"Motivation ({benchmark_title} = 100 index)")
        ax.set_title(f"Familiarity vs Motivation — {segment} / {region}")
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Familiarity (Indexed)")
            st.bar_chart(df.set_index("Title")["Familiarity"])
        with col2:
            st.subheader("Motivation (Indexed)")
            st.bar_chart(df.set_index("Title")["Motivation"])

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "title_scores_v9_ticket_blend.csv",
            "text/csv"
        )
