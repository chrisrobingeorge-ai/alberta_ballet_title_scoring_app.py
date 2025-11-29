"""
Economic Factors Module - BoC Live Data Integration

This module provides functions to compute economic sentiment adjustments using
live data from the Bank of Canada Valet API. It integrates with the existing
historical economic adjustment pipeline as a SUPPLEMENTAL layer.

IMPORTANT DESIGN PRINCIPLE:
- The existing historical economic data (WCS oil prices, Alberta unemployment)
  remains fully intact and is NOT replaced.
- This BoC integration provides ADDITIONAL "live" or "latest" values for
  today's macro conditions.
- Historical analysis, backtests, and model training continue to rely on
  the existing historical datasets.
- When BoC data is unavailable, the system falls back to the historical-based
  economic sentiment factor.

Usage:
    from utils.economic_factors import compute_boc_economic_sentiment
    
    # Get BoC-based sentiment factor for today
    factor, details = compute_boc_economic_sentiment()
    
    # factor is typically in range [0.85, 1.15] where 1.0 = neutral
    # details contains the individual indicator values
"""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import yaml

# Import the BoC client
try:
    from utils.boc_client import (
        get_latest_boc_values,
        get_cache_info,
        BocApiError,
    )
    BOC_CLIENT_AVAILABLE = True
except ImportError:
    BOC_CLIENT_AVAILABLE = False
    get_latest_boc_values = None
    BocApiError = Exception

# Import historical economic sentiment (the existing function)
try:
    from data.loader import get_economic_sentiment_factor
    HISTORICAL_SENTIMENT_AVAILABLE = True
except ImportError:
    HISTORICAL_SENTIMENT_AVAILABLE = False
    get_economic_sentiment_factor = None

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

_boc_config: Optional[dict] = None


def _get_config_path() -> Path:
    """Get path to the BoC economic config file."""
    return Path(__file__).parent.parent / "config" / "economic_boc.yaml"


def load_boc_config(config_path: Optional[str] = None) -> dict:
    """
    Load BoC economic configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
        
    Returns:
        Dictionary with configuration settings
    """
    global _boc_config
    
    if config_path is None:
        path = _get_config_path()
    else:
        path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"BoC config file not found at {path}. Using defaults.")
        return _get_default_config()
    
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded BoC config from {path}")
        _boc_config = config
        return config
    except Exception as e:
        logger.warning(f"Error loading BoC config from {path}: {e}. Using defaults.")
        return _get_default_config()


def _get_default_config() -> dict:
    """Return default configuration when config file is unavailable."""
    return {
        "use_boc_live_data": False,  # Disabled by default when no config
        "fallback_mode": "historical",
        "boc_series": {},
        "sentiment_calculation": {
            "min_factor": 0.85,
            "max_factor": 1.15,
            "neutral": 1.0,
            "sensitivity": 0.10,
        },
    }


def get_boc_config() -> dict:
    """Get the loaded BoC config, loading it if necessary."""
    global _boc_config
    if _boc_config is None:
        _boc_config = load_boc_config()
    return _boc_config


def is_boc_live_enabled() -> bool:
    """Check if BoC live data integration is enabled."""
    config = get_boc_config()
    return config.get("use_boc_live_data", False) and BOC_CLIENT_AVAILABLE


# =============================================================================
# SENTIMENT CALCULATION
# =============================================================================


def _compute_z_score(
    value: float,
    baseline: float,
    std: float,
    direction: str = "positive"
) -> float:
    """
    Compute standardized score for a single indicator.
    
    Args:
        value: Current value of the indicator
        baseline: Historical mean/baseline value
        std: Historical standard deviation
        direction: "positive" if higher is better, "negative" if lower is better
        
    Returns:
        Z-score adjusted for direction (positive = favorable)
    """
    if std <= 0:
        std = 1.0  # Avoid division by zero
    
    z = (value - baseline) / std
    
    # For negative direction indicators (like interest rates),
    # flip the sign so positive z = favorable
    if direction == "negative":
        z = -z
    
    return z


def fetch_boc_indicators() -> Tuple[Dict[str, Optional[float]], bool]:
    """
    Fetch current values for all configured BoC indicators.
    
    Returns:
        Tuple of (values_dict, success_flag)
        - values_dict: Maps series keys to their current values (or None if failed)
        - success_flag: True if at least some values were fetched
    """
    if not BOC_CLIENT_AVAILABLE:
        logger.warning("BoC client not available")
        return {}, False
    
    config = get_boc_config()
    boc_series = config.get("boc_series", {})
    
    if not boc_series:
        logger.info("No BoC series configured")
        return {}, False
    
    # Collect series IDs to fetch
    series_ids = {}
    for key, series_config in boc_series.items():
        series_id = series_config.get("id")
        if series_id:
            series_ids[key] = series_id
    
    if not series_ids:
        return {}, False
    
    # Fetch all values
    try:
        raw_values = get_latest_boc_values(list(series_ids.values()))
    except Exception as e:
        logger.warning(f"Error fetching BoC values: {e}")
        return {}, False
    
    # Map back to config keys
    values = {}
    for key, series_id in series_ids.items():
        values[key] = raw_values.get(series_id)
    
    # Check if we got at least some values
    success = any(v is not None for v in values.values())
    
    if success:
        logger.info(f"Fetched {sum(1 for v in values.values() if v is not None)} BoC indicators")
    else:
        logger.warning("Failed to fetch any BoC indicators")
    
    return values, success


def compute_boc_economic_sentiment(
    run_date: Optional[date] = None,
    city: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute economic sentiment scalar using live BoC data.
    
    This function:
    1. Fetches latest values for configured BoC series
    2. Standardizes each value relative to historical norms
    3. Combines them into a weighted sentiment scalar
    4. Falls back to historical data if BoC is unavailable
    
    The result is a scalar typically in range [0.85, 1.15] where:
    - 1.0 = neutral economic conditions
    - > 1.0 = favorable conditions (may boost ticket sales)
    - < 1.0 = unfavorable conditions (may reduce ticket sales)
    
    Args:
        run_date: Date for the forecast (used for fallback; defaults to today)
        city: City for regional adjustments (Calgary/Edmonton; used for fallback)
        
    Returns:
        Tuple of (sentiment_factor, details_dict)
        - sentiment_factor: Float typically in [0.85, 1.15]
        - details_dict: Dictionary with indicator values, source info, etc.
    """
    config = get_boc_config()
    calc_config = config.get("sentiment_calculation", {})
    
    min_factor = calc_config.get("min_factor", 0.85)
    max_factor = calc_config.get("max_factor", 1.15)
    neutral = calc_config.get("neutral", 1.0)
    sensitivity = calc_config.get("sensitivity", 0.10)
    historical_stats = calc_config.get("historical_stats", {})
    
    details = {
        "source": None,
        "indicators": {},
        "factor": neutral,
        "boc_available": False,
        "fallback_used": False,
    }
    
    # Check if BoC live data is enabled
    if not is_boc_live_enabled():
        logger.debug("BoC live data is disabled, using fallback")
        return _get_fallback_sentiment(run_date, city, config, details)
    
    # Fetch BoC indicators
    values, success = fetch_boc_indicators()
    
    if not success:
        logger.info("BoC data unavailable, using fallback")
        return _get_fallback_sentiment(run_date, city, config, details)
    
    details["boc_available"] = True
    details["source"] = "boc_live"
    
    # Calculate weighted z-score
    boc_series = config.get("boc_series", {})
    weighted_z_sum = 0.0
    total_weight = 0.0
    
    for key, series_config in boc_series.items():
        value = values.get(key)
        
        if value is None:
            continue
        
        weight = series_config.get("weight", 0.0)
        direction = series_config.get("direction", "positive")
        baseline = series_config.get("baseline", 0.0)
        
        # Get historical stats for standardization
        stats = historical_stats.get(key, {})
        mean = stats.get("mean", baseline)
        std = stats.get("std", 1.0)
        
        # Compute z-score
        z = _compute_z_score(value, mean, std, direction)
        
        weighted_z_sum += weight * z
        total_weight += weight
        
        details["indicators"][key] = {
            "value": value,
            "baseline": baseline,
            "z_score": z,
            "weight": weight,
            "direction": direction,
        }
    
    if total_weight <= 0:
        logger.warning("No valid BoC indicators, using fallback")
        return _get_fallback_sentiment(run_date, city, config, details)
    
    # Normalize and convert to factor
    average_z = weighted_z_sum / total_weight
    
    # Apply sensitivity to convert z-score to factor
    # A 1-std-dev positive move increases factor by 'sensitivity'
    factor = neutral + (average_z * sensitivity)
    
    # Clip to allowed range
    factor = float(np.clip(factor, min_factor, max_factor))
    
    details["factor"] = factor
    details["weighted_z"] = average_z
    
    logger.info(f"BoC economic sentiment factor: {factor:.3f} (z={average_z:.2f})")
    
    return factor, details


def _get_fallback_sentiment(
    run_date: Optional[date],
    city: Optional[str],
    config: dict,
    details: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Get fallback economic sentiment when BoC data is unavailable.
    
    Fallback modes:
    - "historical": Use existing historical-based factor (WCS oil, unemployment)
    - "neutral": Return 1.0 (no adjustment)
    - "last_cached": Use last successful BoC values (if any cached)
    """
    fallback_mode = config.get("fallback_mode", "historical")
    calc_config = config.get("sentiment_calculation", {})
    neutral = calc_config.get("neutral", 1.0)
    
    details["fallback_used"] = True
    details["fallback_mode"] = fallback_mode
    
    if fallback_mode == "neutral":
        details["source"] = "neutral_fallback"
        details["factor"] = neutral
        return neutral, details
    
    if fallback_mode == "historical" and HISTORICAL_SENTIMENT_AVAILABLE:
        try:
            import pandas as pd
            ts = pd.Timestamp(run_date) if run_date else pd.Timestamp.now()
            historical_factor = get_economic_sentiment_factor(
                run_date=ts,
                city=city,
            )
            details["source"] = "historical_fallback"
            details["factor"] = historical_factor
            logger.info(f"Using historical economic sentiment: {historical_factor:.3f}")
            return historical_factor, details
        except Exception as e:
            logger.warning(f"Historical sentiment fallback failed: {e}")
    
    # Final fallback: neutral
    details["source"] = "neutral_fallback"
    details["factor"] = neutral
    return neutral, details


def get_boc_indicator_display() -> Dict[str, Any]:
    """
    Get BoC indicator values formatted for UI display.
    
    Returns:
        Dictionary with:
        - "available": bool - whether BoC data is available
        - "indicators": list of dicts with label, value, formatted_value
        - "sentiment_factor": float - the computed sentiment
        - "sentiment_label": str - human-readable label
    """
    config = get_boc_config()
    display_config = config.get("display", {})
    
    if not is_boc_live_enabled():
        return {
            "available": False,
            "message": "BoC live data is disabled",
        }
    
    # Fetch indicators and compute sentiment
    factor, details = compute_boc_economic_sentiment()
    
    if not details.get("boc_available", False):
        return {
            "available": False,
            "message": "BoC data temporarily unavailable",
            "fallback_used": details.get("fallback_used", True),
            "fallback_mode": details.get("fallback_mode"),
            "sentiment_factor": factor,
        }
    
    # Format indicators for display
    show_indicators = display_config.get("show_indicators", [])
    labels = display_config.get("labels", {})
    formats = display_config.get("formats", {})
    
    indicator_list = []
    for key in show_indicators:
        ind_data = details.get("indicators", {}).get(key, {})
        value = ind_data.get("value")
        
        if value is None:
            continue
        
        label = labels.get(key, key)
        fmt = formats.get(key, "{:.2f}")
        
        try:
            formatted = fmt.format(value)
        except Exception:
            formatted = str(value)
        
        indicator_list.append({
            "key": key,
            "label": label,
            "value": value,
            "formatted_value": formatted,
            "z_score": ind_data.get("z_score", 0),
        })
    
    # Generate sentiment label
    if factor > 1.05:
        sentiment_label = "Favorable"
    elif factor > 1.0:
        sentiment_label = "Slightly Favorable"
    elif factor > 0.95:
        sentiment_label = "Neutral"
    elif factor > 0.90:
        sentiment_label = "Slightly Unfavorable"
    else:
        sentiment_label = "Unfavorable"
    
    return {
        "available": True,
        "indicators": indicator_list,
        "sentiment_factor": factor,
        "sentiment_label": sentiment_label,
        "source": details.get("source", "boc_live"),
    }


# =============================================================================
# INTEGRATION WITH EXISTING SENTIMENT FUNCTION
# =============================================================================


def get_combined_economic_sentiment(
    run_date: Optional[date] = None,
    city: Optional[str] = None,
    boc_weight: float = 0.3,
) -> Tuple[float, Dict[str, Any]]:
    """
    Get combined economic sentiment blending BoC live and historical data.
    
    This provides a weighted average of:
    - BoC live data (current market conditions)
    - Historical data (oil prices, unemployment)
    
    When BoC is unavailable, uses 100% historical.
    When historical is unavailable, uses 100% BoC (or neutral).
    
    Args:
        run_date: Date for the forecast
        city: City for regional adjustments
        boc_weight: Weight for BoC data (0-1); remainder is historical
        
    Returns:
        Tuple of (combined_factor, details_dict)
    """
    details = {
        "boc_factor": None,
        "historical_factor": None,
        "combined_factor": 1.0,
        "boc_weight_used": 0.0,
        "historical_weight_used": 0.0,
    }
    
    boc_factor = None
    boc_available = False
    
    # Try BoC
    if is_boc_live_enabled():
        boc_factor, boc_details = compute_boc_economic_sentiment(run_date, city)
        boc_available = boc_details.get("boc_available", False)
        if boc_available:
            details["boc_factor"] = boc_factor
            details["boc_details"] = boc_details
    
    # Try historical
    historical_factor = None
    historical_available = False
    
    if HISTORICAL_SENTIMENT_AVAILABLE:
        try:
            import pandas as pd
            ts = pd.Timestamp(run_date) if run_date else pd.Timestamp.now()
            historical_factor = get_economic_sentiment_factor(
                run_date=ts,
                city=city,
            )
            historical_available = True
            details["historical_factor"] = historical_factor
        except Exception as e:
            logger.debug(f"Historical sentiment unavailable: {e}")
    
    # Combine based on availability
    if boc_available and historical_available:
        combined = (boc_weight * boc_factor) + ((1 - boc_weight) * historical_factor)
        details["boc_weight_used"] = boc_weight
        details["historical_weight_used"] = 1 - boc_weight
    elif boc_available:
        combined = boc_factor
        details["boc_weight_used"] = 1.0
    elif historical_available:
        combined = historical_factor
        details["historical_weight_used"] = 1.0
    else:
        combined = 1.0  # Neutral fallback
    
    details["combined_factor"] = combined
    
    return combined, details
