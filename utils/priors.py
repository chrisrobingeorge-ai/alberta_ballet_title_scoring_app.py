# utils/priors.py
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime

# === learned priors (filled by learn_priors_from_history) ===
TITLE_CITY_PRIORS: dict[str, dict[str, float]] = {}
CATEGORY_CITY_PRIORS: dict[str, dict[str, float]] = {}

# ---- sensible fallbacks if history is missing or too thin ----
DEFAULT_BASE_CITY_SPLIT = {"Calgary": 0.60, "Edmonton": 0.40}  # only used as last-ditch

_CITY_BLEND_TO_PRIOR = 0.40         # how much to trust learned priors vs base
_CITY_CLIP_RANGE     = (0.15, 0.85) # guardrails

# ----------------- helpers (schema-flexible) -----------------
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None

def _find_city(df):       return _pick_col(df, ["city", "market", "region"])
def _find_title(df):      return _pick_col(df, ["title", "production", "show", "event"])
def _find_category(df):   return _pick_col(df, ["category", "program", "genre", "show_type"])
def _find_date(df):       return _pick_col(df, ["date", "performance_date", "start_date", "order_date", "season_start"])
def _find_total(df):      return _pick_col(df, ["total_tickets", "tickets", "qty", "units", "total"])
def _find_singles(df):    return _pick_col(df, ["singles", "single_tickets", "single_qty", "single_units"])

def _to_datetime_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT]*len(s)), errors="coerce", utc=True)

def _clip01(p: float, lo_hi=_CITY_CLIP_RANGE) -> float:
    lo, hi = lo_hi
    return float(min(hi, max(lo, p)))

def _normalize_pair(c: float, e: float) -> tuple[float, float]:
    s = (c + e) or 1.0
    return float(c / s), float(e / s)

def _default_base_split() -> tuple[float, float]:
    c = float(DEFAULT_BASE_CITY_SPLIT.get("Calgary", 0.5))
    e = float(DEFAULT_BASE_CITY_SPLIT.get("Edmonton", 0.5))
    return _normalize_pair(c, e)

def _is_sane_prior(p: dict | None) -> bool:
    if not isinstance(p, dict): return False
    try:
        c = float(p.get("Calgary", np.nan)); e = float(p.get("Edmonton", np.nan))
        if not np.isfinite(c) or not np.isfinite(e): return False
        return (c + e) > 0
    except Exception:
        return False

# ----------------- learn from history -----------------
def learn_priors_from_history(
    df: pd.DataFrame,
    *,
    calgary_labels = {"calgary", "yyc"},
    edmonton_labels = {"edmonton", "yeg"},
    recency_half_life_days: int | None = 730,  # 2-year half-life
    min_rows_title: int = 2,
    min_rows_category: int = 5,
) -> dict:
    """
    Populate TITLE_CITY_PRIORS, CATEGORY_CITY_PRIORS
    using an arbitrary history DataFrame (column names auto-detected).
    
    Note: This function focuses on single ticket data only.
    """
    global TITLE_CITY_PRIORS, CATEGORY_CITY_PRIORS
    TITLE_CITY_PRIORS = {}
    CATEGORY_CITY_PRIORS = {}

    if df is None or df.empty:
        return {"ok": False, "reason": "history_is_empty"}

    city_col     = _find_city(df)
    title_col    = _find_title(df)
    category_col = _find_category(df)
    date_col     = _find_date(df)
    total_col    = _find_total(df)
    singles_col  = _find_singles(df)

    if not city_col:
        return {"ok": False, "reason": "missing_city_column"}
    if not title_col and not category_col:
        return {"ok": False, "reason": "missing_title_and_category"}

    work = df.copy()

    # normalize city names
    def _norm_city(x: str) -> str | None:
        if not isinstance(x, str): return None
        xl = x.strip().lower()
        if xl in calgary_labels or "calg" in xl or "yyc" in xl:  return "Calgary"
        if xl in edmonton_labels or "edmon" in xl or "yeg" in xl: return "Edmonton"
        return None

    work["__city"] = work[city_col].map(_norm_city)
    work = work[work["__city"].isin(["Calgary", "Edmonton"])].copy()
    if work.empty:
        return {"ok": False, "reason": "no_valid_cities"}

    # totals
    if total_col:
        work["__tot"] = pd.to_numeric(work[total_col], errors="coerce").fillna(0.0)
    else:
        s = pd.to_numeric(work[singles_col], errors="coerce").fillna(0.0) if singles_col else 0.0
        work["__tot"] = s

    # recency weights
    if date_col and recency_half_life_days:
        dts = _to_datetime_safe(work[date_col])
        now = pd.Timestamp(datetime.utcnow(), tz="UTC")
        age_days = (now - dts).dt.days.clip(lower=0)
        lam = np.log(2.0) / float(recency_half_life_days)
        w = np.exp(-lam * age_days)
        work["__w"] = w.fillna(1.0).astype(float)
    else:
        work["__w"] = 1.0

    def _agg_weighted_sum(group, value_col):
        return float((group[value_col] * group["__w"]).sum())

    # title-level priors
    if title_col:
        g = work.groupby([work[title_col], "__city"], dropna=False)
        sums = g.apply(lambda x: _agg_weighted_sum(x, "__tot")).rename("wt_tot").reset_index()
        counts = work.groupby(work[title_col]).size().rename("n").reset_index()
        valid_titles = set(counts[counts["n"] >= min_rows_title][title_col])
        for t in valid_titles:
            sub = sums[sums[title_col] == t]
            c = float(sub.loc[sub["__city"] == "Calgary", "wt_tot"].sum())
            e = float(sub.loc[sub["__city"] == "Edmonton", "wt_tot"].sum())
            if c + e > 0:
                C, E = _normalize_pair(c, e)
                TITLE_CITY_PRIORS[str(t)] = {"Calgary": C, "Edmonton": E}

    # category-level priors
    if category_col:
        g = work.groupby([work[category_col], "__city"], dropna=False)
        sums = g.apply(lambda x: _agg_weighted_sum(x, "__tot")).rename("wt_tot").reset_index()
        counts = work.groupby(work[category_col]).size().rename("n").reset_index()
        valid_cats = set(counts[counts["n"] >= min_rows_category][category_col])
        for cat in valid_cats:
            sub = sums[sums[work[category_col].name] == cat]
            c = float(sub.loc[sub["__city"] == "Calgary", "wt_tot"].sum())
            e = float(sub.loc[sub["__city"] == "Edmonton", "wt_tot"].sum())
            if c + e > 0:
                C, E = _normalize_pair(c, e)
                CATEGORY_CITY_PRIORS[str(cat)] = {"Calgary": C, "Edmonton": E}

    return {
        "ok": True,
        "titles_learned": len(TITLE_CITY_PRIORS),
        "categories_learned": len(CATEGORY_CITY_PRIORS),
    }

# ----------------- public API used by your app -----------------
def city_split_for(title: str, category: str) -> tuple[float, float]:
    base_c, base_e = _default_base_split()
    prior = TITLE_CITY_PRIORS.get(title)
    if not _is_sane_prior(prior) and category:
        prior = CATEGORY_CITY_PRIORS.get(category)

    if _is_sane_prior(prior):
        pc = float(prior["Calgary"]); pe = float(prior["Edmonton"])
        pc, pe = _normalize_pair(pc, pe)
        w = float(_CITY_BLEND_TO_PRIOR)
        c = (1.0 - w) * base_c + w * pc
        e = (1.0 - w) * base_e + w * pe
    else:
        c, e = base_c, base_e

    c = _clip01(c); e = _clip01(e)
    return _normalize_pair(c, e)
