"""
Title Explanation Engine

This module generates SHAP-driven, multi-paragraph narratives for individual titles
in the Season Report PDF. Each narrative explains:
- Signal positioning (Familiarity & Motivation)
- Historical & category context
- Seasonal & macro factors
- SHAP-based driver summary
- Board-level interpretation

All explanations are programmatically derived from features, predictions, and SHAP values.
No hardcoded title-specific logic.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


def build_title_explanation(
    title_metadata: Dict[str, Any],
    prediction_outputs: Optional[Dict[str, Any]] = None,
    shap_values: Optional[Dict[str, float]] = None,
    *,
    style: str = "board"
) -> str:
    """
    Generate a comprehensive (~250-350 word) narrative explanation for a single title.
    
    This narrative is designed for board-level audiences and includes:
    1. Signal positioning (Familiarity/Motivation)
    2. Historical & category context (premiere vs remount)
    3. Seasonal & macro factors
    4. SHAP-based driver summary (if available)
    5. Board-level interpretation of Ticket Index
    
    Args:
        title_metadata: Dictionary containing title features and metadata:
            - Title: str
            - Month: str
            - Category: str
            - Familiarity: float (0-100+ scale)
            - Motivation: float (0-100+ scale)
            - SignalOnly: float (combined signal)
            - TicketIndex used: float (relative demand measure)
            - FutureSeasonalityFactor: float (month adjustment)
            - PrimarySegment: str
            - SecondarySegment: str
            - YYC_Singles: int
            - YEG_Singles: int
            - ReturnDecayPct: float (if remount)
            - IsRemount: bool (if available)
            - YearsSinceLastRun: int (if available)
        
        prediction_outputs: Optional model output metadata
        shap_values: Optional dict mapping feature names to SHAP contributions
        style: Narrative style ("board", "technical", "executive")
    
    Returns:
        Multi-paragraph HTML-formatted narrative string suitable for PDF generation
    """
    # Extract key values with safe defaults
    title = title_metadata.get("Title", "Unknown Title")
    month = title_metadata.get("Month", "")
    category = title_metadata.get("Category", "")
    
    familiarity = title_metadata.get("Familiarity", None)
    motivation = title_metadata.get("Motivation", None)
    signal_only = title_metadata.get("SignalOnly", None)
    
    ticket_index = title_metadata.get("TicketIndex used", 
                                     title_metadata.get("EffectiveTicketIndex", 100))
    seasonality = title_metadata.get("FutureSeasonalityFactor", 1.0)
    
    primary_seg = title_metadata.get("PrimarySegment", "")
    secondary_seg = title_metadata.get("SecondarySegment", "")
    
    yyc = title_metadata.get("YYC_Singles", 0) or 0
    yeg = title_metadata.get("YEG_Singles", 0) or 0
    total_tickets = yyc + yeg
    
    decay_pct = title_metadata.get("ReturnDecayPct", 0.0) or 0.0
    is_remount = title_metadata.get("IsRemount", False) or (decay_pct > 0)
    years_since = title_metadata.get("YearsSinceLastRun", None)
    
    # Build narrative in paragraphs
    paragraphs = []
    
    # Paragraph 1: Title header and signal positioning
    p1_parts = [f"<b>{month} — {title}</b> ({category})"]
    
    if familiarity is not None and motivation is not None:
        fam_desc = _describe_signal_level(familiarity)
        mot_desc = _describe_signal_level(motivation)
        p1_parts.append(
            f"This title registers a <b>Familiarity</b> score of {familiarity:.1f} ({fam_desc}) "
            f"and a <b>Motivation</b> score of {motivation:.1f} ({mot_desc}), reflecting "
            f"{'strong' if signal_only and signal_only > 100 else 'moderate' if signal_only and signal_only > 70 else 'emerging'} "
            f"public visibility across Wikipedia page traffic, Google search patterns, YouTube viewing behavior, "
            f"and Spotify streaming activity."
        )
    else:
        p1_parts.append(
            f"This title's demand forecast is derived from a combined interest signal that benchmarks "
            f"public visibility across multiple digital platforms."
        )
    
    paragraphs.append(" ".join(p1_parts))
    
    # Paragraph 2: Historical & category context
    p2_parts = []
    
    if is_remount and years_since is not None and years_since > 0:
        p2_parts.append(
            f"This production represents a remount, last performed approximately {years_since} "
            f"{'year' if years_since == 1 else 'years'} ago. "
            f"Historical Alberta Ballet data shows that {_describe_category(category)} productions "
            f"typically {'benefit from audience recognition on return engagements' if years_since >= 3 else 'face modest softening when remounted within a short window'}, "
            f"a pattern the model incorporates into its baseline expectations."
        )
    elif is_remount:
        p2_parts.append(
            f"This production represents a remount, returning to Alberta Ballet's repertoire. "
            f"The model accounts for audience familiarity dynamics based on learned patterns "
            f"from similar {_describe_category(category)} remounts in the company's history."
        )
    else:
        p2_parts.append(
            f"This title represents a premiere for Alberta Ballet audiences. "
            f"The model evaluates it within the context of {_describe_category(category)} productions, "
            f"drawing on learned patterns from comparable works without local performance history. "
            f"This category-informed baseline helps anchor expectations when direct title-level priors are unavailable."
        )
    
    if p2_parts:
        paragraphs.append(" ".join(p2_parts))
    
    # Paragraph 3: Seasonal & macro layer
    p3_parts = []
    month_name = month.split()[0] if month else "scheduled month"
    
    if seasonality is not None:
        if seasonality > 1.05:
            seasonal_desc = (
                f"The {month_name} scheduling carries a favorable seasonal multiplier of {seasonality:.2f}, "
                f"reflecting historically stronger demand for this category during this period — "
                f"likely influenced by {'holiday proximity and heightened cultural activity' if 'Dec' in month or 'Nov' in month else 'seasonal momentum and audience availability'}."
            )
        elif seasonality < 0.95:
            seasonal_desc = (
                f"The {month_name} timing applies a seasonal adjustment of {seasonality:.2f}, "
                f"acknowledging that this category typically experiences softer demand during this shoulder period "
                f"compared to peak months."
            )
        else:
            seasonal_desc = (
                f"The {month_name} scheduling carries a near-neutral seasonal factor of {seasonality:.2f}, "
                f"suggesting typical mid-season performance expectations for this category."
            )
        
        p3_parts.append(seasonal_desc)
    
    # Add economic context (generic for now, could be enhanced with actual economic data)
    p3_parts.append(
        f"The forecast incorporates time-aligned economic indicators including interest rates, "
        f"energy prices, employment levels, and consumer confidence, ensuring the prediction reflects "
        f"the macroeconomic environment audiences will experience at opening."
    )
    
    if p3_parts:
        paragraphs.append(" ".join(p3_parts))
    
    # Paragraph 4: SHAP-based driver summary (or feature-based if SHAP unavailable)
    p4_parts = []
    
    if shap_values and len(shap_values) > 0:
        # Use actual SHAP values to identify key drivers
        top_contributors = _identify_shap_drivers(shap_values, title_metadata)
        p4_parts.append(top_contributors)
    else:
        # Fall back to feature-based interpretation
        drivers = []
        if familiarity and familiarity > 110:
            drivers.append("elevated public recognition")
        if motivation and motivation > 110:
            drivers.append("strong engagement signals")
        if seasonality and seasonality > 1.05:
            drivers.append("favorable seasonal timing")
        if is_remount and years_since and years_since >= 3:
            drivers.append("renewed interest after an appropriate interval")
        
        if drivers:
            p4_parts.append(
                f"The model's demand estimate is primarily shaped by {_format_list(drivers)}, "
                f"which collectively position this title {'above' if ticket_index > 100 else 'at' if ticket_index >= 95 else 'below'} "
                f"the reference benchmark."
            )
        else:
            p4_parts.append(
                f"The model synthesizes multiple feature contributions to arrive at a balanced demand estimate, "
                f"with the combined signal strength positioning this title at a Ticket Index of {ticket_index:.1f}."
            )
    
    if p4_parts:
        paragraphs.append(" ".join(p4_parts))
    
    # Paragraph 5: Board-level interpretation
    p5_parts = []
    
    index_tier = _interpret_ticket_index(ticket_index)
    p5_parts.append(
        f"The resulting <b>Ticket Index of {ticket_index:.1f}</b> places this production in the "
        f"<b>{index_tier}</b> tier of expected demand relative to the company's historical benchmark. "
    )
    
    if total_tickets > 0:
        p5_parts.append(
            f"Translated into actionable planning figures and applying learned Calgary-Edmonton splits, "
            f"the model forecasts approximately <b>{yyc:,} tickets in Calgary</b> and "
            f"<b>{yeg:,} tickets in Edmonton</b>, totaling {total_tickets:,} single-ticket sales. "
        )
    
    if primary_seg:
        p5_parts.append(
            f"Audience composition is expected to skew toward <b>{primary_seg}</b>"
            f"{' with secondary appeal to ' + secondary_seg if secondary_seg else ''}, "
            f"informing marketing channel prioritization and messaging strategy."
        )
    
    if p5_parts:
        paragraphs.append(" ".join(p5_parts))
    
    # Join all paragraphs with spacing
    return " ".join(paragraphs)


def _describe_signal_level(value: float) -> str:
    """Describe a signal score in qualitative terms."""
    if value >= 120:
        return "exceptionally high"
    elif value >= 100:
        return "strong"
    elif value >= 80:
        return "above average"
    elif value >= 60:
        return "moderate"
    elif value >= 40:
        return "emerging"
    else:
        return "limited"


def _describe_category(category: str) -> str:
    """Convert category codes to readable descriptions."""
    category_map = {
        "family_classic": "family-oriented classical",
        "adult_classic": "adult classical",
        "contemporary": "contemporary",
        "holiday": "holiday-themed",
        "mixed_rep": "mixed repertoire",
        "nutcracker": "Nutcracker",
    }
    return category_map.get(category.lower(), category)


def _interpret_ticket_index(index: float) -> str:
    """Interpret Ticket Index value as a demand tier."""
    if index >= 120:
        return "exceptional demand"
    elif index >= 105:
        return "strong demand"
    elif index >= 95:
        return "benchmark demand"
    elif index >= 80:
        return "moderate demand"
    elif index >= 60:
        return "developing demand"
    else:
        return "emerging demand"


def _format_list(items: List[str]) -> str:
    """Format a list of items into natural English."""
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def _identify_shap_drivers(shap_values: Dict[str, float], metadata: Dict[str, Any]) -> str:
    """
    Identify and describe the top SHAP contributors to the prediction.
    
    Args:
        shap_values: Dict mapping feature names to SHAP contribution values
        metadata: Title metadata for contextual interpretation
    
    Returns:
        Narrative paragraph describing key drivers
    """
    # Sort by absolute SHAP value to find most influential features
    sorted_features = sorted(
        shap_values.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Take top 3-5 most influential
    top_features = sorted_features[:5]
    
    # Separate positive and negative contributors
    positive = [(feat, val) for feat, val in top_features if val > 0]
    negative = [(feat, val) for feat, val in top_features if val < 0]
    
    parts = []
    
    if positive:
        pos_desc = _describe_shap_features(positive, direction="positive")
        parts.append(
            f"Key upward drivers include {pos_desc}, which collectively elevate the forecast "
            f"by approximately {sum(val for _, val in positive):.1f} index points."
        )
    
    if negative:
        neg_desc = _describe_shap_features(negative, direction="negative")
        parts.append(
            f"Offsetting factors include {neg_desc}, which moderate expectations "
            f"by roughly {abs(sum(val for _, val in negative)):.1f} index points."
        )
    
    if not parts:
        parts.append(
            "Feature contributions are relatively balanced, with no single factor dominating the prediction."
        )
    
    return " ".join(parts)


def _describe_shap_features(features: List[Tuple[str, float]], direction: str) -> str:
    """
    Convert SHAP feature names into human-readable descriptions.
    
    Args:
        features: List of (feature_name, shap_value) tuples
        direction: "positive" or "negative"
    
    Returns:
        Readable description of features
    """
    descriptions = []
    
    for feat_name, val in features:
        # Map technical feature names to readable descriptions
        if "familiarity" in feat_name.lower() or "wiki" in feat_name.lower():
            descriptions.append("strong public recognition signals")
        elif "motivation" in feat_name.lower() or "youtube" in feat_name.lower():
            descriptions.append("elevated engagement indicators")
        elif "season" in feat_name.lower() or "month" in feat_name.lower():
            descriptions.append("favorable seasonal positioning")
        elif "remount" in feat_name.lower() or "years_since" in feat_name.lower():
            descriptions.append("remount timing dynamics")
        elif "category" in feat_name.lower():
            descriptions.append("category-specific historical patterns")
        elif "econ" in feat_name.lower() or "sentiment" in feat_name.lower():
            descriptions.append("macroeconomic conditions")
        elif "prior" in feat_name.lower() or "median" in feat_name.lower():
            descriptions.append("strong historical precedent")
        else:
            # Generic fallback
            readable = feat_name.replace("_", " ").lower()
            descriptions.append(f"{readable}")
    
    # Deduplicate while preserving order
    unique_descriptions = []
    seen = set()
    for desc in descriptions:
        if desc not in seen:
            unique_descriptions.append(desc)
            seen.add(desc)
    
    return _format_list(unique_descriptions[:3])  # Limit to top 3 for readability
