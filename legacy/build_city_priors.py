# ============================================================================
# ⚠️  DEPRECATED - DO NOT USE FOR PRODUCTION  ⚠️
# ============================================================================
#
# This script has been moved to the legacy/ directory and is DEPRECATED.
# It is kept only for historical reference and to understand how city priors
# were originally computed.
#
# The current application learns city priors dynamically from history_city_sales.csv
# at runtime. See the learn_priors_from_history() function in streamlit_app.py.
#
# ============================================================================

import pandas as pd
import numpy as np
import sys, re

# Usage: python legacy/build_city_priors.py path/to/history.csv

def infer_category(title: str) -> str:
    # Keep aligned with app's infer_gender_and_category() / BASELINES categories
    t = title.lower()
    if any(k in t for k in ["wizard","peter pan","pinocchio","hansel","frozen","beauty","alice"]):
        return "family_classic"
    if any(k in t for k in ["swan","sleeping","cinderella","giselle","sylphide"]):
        return "classic_romance"
    if any(k in t for k in ["romeo","hunchback","notre dame","hamlet","frankenstein","dracula"]):
        return "romantic_tragedy"
    if any(k in t for k in ["don quixote","merry widow"]):
        return "classic_comedy"
    if any(k in t for k in ["contemporary","boyz","ballet boyz","momix","grimm","nijinsky","shadowland","deviate","complexions"]):
        return "contemporary"
    if any(k in t for k in ["taj","tango","harlem","tragically hip","leonard cohen","gordon lightfoot","bowie","phi"]):
        return "pop_ip"
    return "dramatic"

def main(path):
    df = pd.read_csv(path)
    # normalize headers (robust to minor variations)
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "show" in lc and "title" in lc: ren[c] = "Title"
        elif "single" in lc and "calg" in lc: ren[c] = "Singles_Calgary"
        elif "single" in lc and "edmon" in lc: ren[c] = "Singles_Edmonton"
        elif ("subscr" in lc or "subs" in lc) and "calg" in lc: ren[c] = "Subs_Calgary"
        elif ("subscr" in lc or "subs" in lc) and "edmon" in lc: ren[c] = "Subs_Edmonton"
    df = df.rename(columns=ren)

    for col in ["Singles_Calgary","Singles_Edmonton","Subs_Calgary","Subs_Edmonton"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Title"] = df["Title"].astype(str).str.strip()
    df["Category"] = df["Title"].map(infer_category)

    # aggregate duplicates (sum across runs)
    agg = df.groupby(["Title","Category"], as_index=False).sum(numeric_only=True)

    agg["Calgary_Total"]  = agg["Singles_Calgary"]  + agg["Subs_Calgary"]
    agg["Edmonton_Total"] = agg["Singles_Edmonton"] + agg["Subs_Edmonton"]
    agg["Grand_Total"]    = agg["Calgary_Total"] + agg["Edmonton_Total"]

    # ---- Title priors (require minimum volume to avoid noise) ----
    MIN_VOL = 800
    tpri = {}
    good = agg[agg["Grand_Total"] >= MIN_VOL].copy()
    good = good[(good["Calgary_Total"] + good["Edmonton_Total"]) > 0]
    for _, r in good.iterrows():
        cgy, edm = float(r["Calgary_Total"]), float(r["Edmonton_Total"])
        den = max(1e-9, cgy + edm)
        tpri[r["Title"]] = {"Calgary": round(cgy/den, 4), "Edmonton": round(edm/den, 4)}

    # ---- Category priors (weighted by volume) ----
    cpri = {}
    cat = agg.groupby("Category", as_index=False)[["Calgary_Total","Edmonton_Total"]].sum()
    for _, r in cat.iterrows():
        cgy, edm = float(r["Calgary_Total"]), float(r["Edmonton_Total"])
        den = max(1e-9, cgy + edm)
        cpri[r["Category"]] = {"Calgary": round(cgy/den, 4), "Edmonton": round(edm/den, 4)}

    # ---- Subscriber share by city & category ----
    subs_city = {"Calgary": {}, "Edmonton": {}}
    cat2 = agg.groupby("Category", as_index=False).sum(numeric_only=True)
    for _, r in cat2.iterrows():
        # Calgary subs fraction: subs / (subs + singles) in that city
        def frac(num, den): return round((num/den), 4) if den > 0 else None
        subs_city["Calgary"][r["Category"]]  = frac(r["Subs_Calgary"],  r["Subs_Calgary"]  + r["Singles_Calgary"])
        subs_city["Edmonton"][r["Category"]] = frac(r["Subs_Edmonton"], r["Subs_Edmonton"] + r["Singles_Edmonton"])

    # print as Python literals ready to paste
    import pprint
    print("# ---- Paste into app as TITLE_CITY_PRIORS ----")
    pprint.pprint(tpri, sort_dicts=True)
    print("\n# ---- Paste into app as CATEGORY_CITY_PRIORS ----")
    pprint.pprint(cpri, sort_dicts=True)
    print("\n# ---- Paste into app as SUBS_SHARE_BY_CATEGORY_CITY ----")
    pprint.pprint(subs_city, sort_dicts=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/build_city_priors.py path/to/history.csv")
        sys.exit(1)
    main(sys.argv[1])
