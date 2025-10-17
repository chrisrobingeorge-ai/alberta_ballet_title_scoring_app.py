# streamlit_app_v9_new_titles_fix.py
# Alberta Ballet — Title Familiarity & Motivation Scorer (v9 Test, New Titles FIX)
# - Hard-coded baselines (Nutcracker = 100 index)
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
st.caption("Hard-coded Alberta-wide baselines (Nutcracker = 100). Add new titles; choose live fetch or offline estimate.")

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
- **Normalization:** All titles are scaled relative to *The Nutcracker* (index = 100).  
  Scores above 100 represent titles with greater recognition or motivation potential; below 100 suggests lesser-known works.
- **Segments:** Audience segments apply different weightings:
  - *Core Classical (F35–64):* Prioritizes classic ballets and female or co-leads.
  - *Family (Parents w/ kids):* Boosts family-oriented, accessible stories and pop IPs.
  - *Emerging Adults (18–34):* Emphasizes contemporary or visually bold productions.
- **Regions:** Regional multipliers reflect audience variation across Alberta:
  - *Calgary:* Slightly higher affinity for classical and family titles.
  - *Edmonton:* Slightly stronger interest in contemporary or conceptual programs.
- **Unknown Titles:**  
  - If *Use Live Data* is enabled and API keys are provided, the app fetches real data from Wikipedia, YouTube, Spotify, and Google Trends.  
  - Otherwise, it estimates based on category medians and historical genre performance.
- **Benchmarking:** All familiarity and motivation scores are expressed as a percentage relative to *The Nutcracker* under the current segment and region selection.

### **Interpretation**
- **Familiarity** reflects brand or story awareness (e.g., “Have people heard of it?”).  
- **Motivation** reflects the likelihood of ticket purchase interest (e.g., “Would they go see it?”).  
- Both indicators are best used for comparing titles *against each other* rather than as absolute predictions.

---

### **Glossary of Terms**
| Term | Definition |
|------|-------------|
| **Baseline** | Hard-coded familiarity and motivation indices for existing repertoire, normalized to *The Nutcracker* (100). |
| **Familiarity** | Relative measure of audience awareness based on Wikipedia, Trends, and Spotify indices. |
| **Motivation** | Relative measure of interest and engagement potential based on YouTube, Trends, and Wiki data. |
| **Segment** | Target demographic lens (Core Classical, Family, Emerging Adults, or General). Adjusts score weighting. |
| **Region** | Market context (Province-wide default, Calgary, or Edmonton). Adjusts for local preferences. |
| **Pop IP** | “Popular Intellectual Property” — stories known through film, literature, or franchise (e.g., *Frozen*, *Beauty and the Beast*). |
| **Contemporary** | Non-narrative or modern programs, often concept-driven or physically abstract. |
| **Classic Romance** | Canonical 19th-century works with romantic or tragic themes (e.g., *Swan Lake*, *Giselle*). |
| **Family Classic** | Accessible narrative works suitable for all ages, often fairy-tale or children’s stories. |
| **Normalization** | Scaling process that expresses all familiarity and motivation scores as a ratio to *The Nutcracker*. |
| **Live Fetch** | Optional real-time retrieval of data from Wikipedia, Google Trends, YouTube, and Spotify for new titles. |

---

**Note:** This tool is not a predictor of exact ticket sales but a comparative instrument for programming strategy and communication alignment.
    """)

# -------------------------
# BASELINE DATA (subset for test run)
# Values are indices relative to Nutcracker=100 (not capped).
BASELINES: Dict[str, Dict[str, float | str]] = {
    "The Nutcracker": {"wiki": 100, "trends": 100, "youtube": 100, "spotify": 100, "gender": "female", "category": "family_classic"},
    "Sleeping Beauty": {"wiki": 92, "trends": 85, "youtube": 78, "spotify": 74, "gender": "female", "category": "classic_romance"},
    "Cinderella": {"wiki": 88, "trends": 80, "youtube": 82, "spotify": 80, "gender": "female", "category": "family_classic"},
    "Swan Lake": {"wiki": 95, "trends": 90, "youtube": 88, "spotify": 84, "gender": "female", "category": "classic_romance"},
    "Romeo and Juliet": {"wiki": 90, "trends": 82, "youtube": 79, "spotify": 77, "gender": "co", "category": "romantic_tragedy"},
    "The Merry Widow": {"wiki": 70, "trends": 60, "youtube": 55, "spotify": 50, "gender": "female", "category": "romantic_comedy"},
    "Beauty and the Beast": {"wiki": 94, "trends": 97, "youtube": 92, "spotify": 90, "gender": "female", "category": "family_classic"},
    "Frozen": {"wiki": 100, "trends": 110, "youtube": 120, "spotify": 115, "gender": "female", "category": "pop_ip"},
    "Peter Pan": {"wiki": 80, "trends": 78, "youtube": 85, "spotify": 82, "gender": "male", "category": "family_classic"},
    "Pinocchio": {"wiki": 72, "trends": 68, "youtube": 70, "spotify": 66, "gender": "male", "category": "family_classic"},
    "Don Quixote": {"wiki": 88, "trends": 75, "youtube": 72, "spotify": 68, "gender": "male", "category": "classic_comedy"},
    "Giselle": {"wiki": 82, "trends": 72, "youtube": 65, "spotify": 60, "gender": "female", "category": "classic_romance"},
    "Nijinsky": {"wiki": 50, "trends": 45, "youtube": 48, "spotify": 40, "gender": "male", "category": "contemporary"},
    "Hansel and Gretel": {"wiki": 78, "trends": 70, "youtube": 65, "spotify": 62, "gender": "co", "category": "family_classic"},
    "Contemporary Composers": {"wiki": 60, "trends": 55, "youtube": 58, "spotify": 50, "gender": "na", "category": "contemporary"},
    "Notre Dame de Paris": {"wiki": 85, "trends": 88, "youtube": 80, "spotify": 76, "gender": "male", "category": "romantic_tragedy"},
    "Wizard of Oz": {"wiki": 97, "trends": 93, "youtube": 95, "spotify": 90, "gender": "female", "category": "family_classic"},
    "Grimm": {"wiki": 55, "trends": 52, "youtube": 50, "spotify": 45, "gender": "na", "category": "contemporary"},
    "Ballet Boyz": {"wiki": 45, "trends": 40, "youtube": 60, "spotify": 58, "gender": "male", "category": "contemporary"},
    "Frankenstein": {"wiki": 68, "trends": 63, "youtube": 66, "spotify": 62, "gender": "male", "category": "dramatic"},
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

    if any(k in t for k in ["nutcracker","wizard","peter pan","pinocchio","hansel","frozen","beauty","alice"]):
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

default_list = list(BASELINES.keys())[:10]
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

if run:
    if not titles:
        st.warning("Add at least one title to score.")
    else:
        rows = []
        unknown_used_live, unknown_used_est = [], []
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

        df = pd.DataFrame(rows)

        # --- Nutcracker normalization (index Nutcracker = 100) ---
        nut_entry = BASELINES["The Nutcracker"]
        nut_fam_raw, nut_mot_raw = calc_scores(nut_entry, segment, region)
        # Avoid divide-by-zero (very unlikely, but safe)
        nut_fam_raw = nut_fam_raw or 1.0
        nut_mot_raw = nut_mot_raw or 1.0
        df["Familiarity"] = (df["FamiliarityRaw"] / nut_fam_raw) * 100.0
        df["Motivation"]  = (df["MotivationRaw"]  / nut_mot_raw)  * 100.0

        # (Optional) show which titles used which data path
        if unknown_used_est:
            st.info(f"Estimated (offline) for new titles: {', '.join(unknown_used_est)}")
        if unknown_used_live:
            st.success(f"Used LIVE data for new titles: {', '.join(unknown_used_live)}")

        # --- Letter grade (A–E) from the indexed Familiarity & Motivation ---
        df["Composite"] = df[["Familiarity", "Motivation"]].mean(axis=1)

        def _assign_score(v: float) -> str:
            if v >= 90: return "A"
            elif v >= 75: return "B"
            elif v >= 60: return "C"
            elif v >= 45: return "D"
            else: return "E"

        df["Score"] = df["Composite"].apply(_assign_score)

        # --- (Optional) Apply Benchmark AFTER indexes exist ---
        if "do_benchmark" in locals() and do_benchmark and "benchmark_title" in locals() and benchmark_title:
            # apply_benchmark supports Familiarity/Motivation columns
            df = apply_benchmark(df, benchmark_title, use_adjusted=False)

        # ===== Display Table with Score =====
        display_cols = [
            "Title","Region","Segment","Gender","Category",
            "Familiarity","Motivation","Score",
            "WikiIdx","TrendsIdx","YouTubeIdx","SpotifyIdx","Source"
        ]
        existing = [c for c in display_cols if c in df.columns]

        # Quick grade summary
        grade_counts = df["Score"].value_counts().reindex(["A","B","C","D","E"]).fillna(0).astype(int)
        st.caption(f"Grade distribution — A:{grade_counts['A']}  B:{grade_counts['B']}  C:{grade_counts['C']}  D:{grade_counts['D']}  E:{grade_counts['E']}")

        st.dataframe(
            df[existing]
              .sort_values(by=["Motivation","Familiarity"], ascending=False)
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

        # Quadrant
        fig, ax = plt.subplots()
        ax.scatter(df["Familiarity"], df["Motivation"])
        for _, r in df.iterrows():
            ax.annotate(r["Title"], (r["Familiarity"], r["Motivation"]), fontsize=8)
        ax.axvline(df["Familiarity"].median(), color='gray', linestyle='--')
        ax.axhline(df["Motivation"].median(), color='gray', linestyle='--')
        ax.set_xlabel("Familiarity (Nutcracker = 100 index)")
        ax.set_ylabel("Motivation (Nutcracker = 100 index)")
        ax.set_title(f"Familiarity vs Motivation — {segment} / {region}")
        st.pyplot(fig)

        # Bars
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
            "title_scores_v9_new_titles.csv",
            "text/csv"
        )
