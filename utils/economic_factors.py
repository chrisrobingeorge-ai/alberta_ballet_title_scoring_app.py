"""
Economic Factors Module - BoC and Alberta Live Data Integration

This module provides functions to compute economic sentiment adjustments using
live data from:
- Bank of Canada Valet API (https://www.bankofcanada.ca/valet/)
- Alberta Economic Dashboard API (https://economicdashboard.alberta.ca/)

It integrates with the existing historical economic adjustment pipeline as a
SUPPLEMENTAL layer.

IMPORTANT DESIGN PRINCIPLE:
- The existing historical economic data (WCS oil prices, Alberta unemployment)
  remains fully intact and is NOT replaced.
- This BoC/Alberta integration provides ADDITIONAL "live" or "latest" values for
  today's macro conditions.
- Historical analysis, backtests, and model training continue to rely on
  the existing historical datasets.
- When live data is unavailable, the system falls back to the historical-based
  economic sentiment factor.

Usage:
    from utils.economic_factors import (
        compute_boc_economic_sentiment,
        get_alberta_economic_indicators,
        get_current_economic_context,
    )
    
    # Get BoC-based sentiment factor for today
    factor, details = compute_boc_economic_sentiment()
    
    # Get Alberta indicators
    ab_indicators = get_alberta_economic_indicators()
    
    # Get combined economic context (all sources)
    context = get_current_economic_context()
    
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
        get_latest_group_observation,
        get_group_metadata,
        BocApiError,
    )
    BOC_CLIENT_AVAILABLE = True
    BOC_GROUPS_AVAILABLE = True
except ImportError:
    BOC_CLIENT_AVAILABLE = False
    BOC_GROUPS_AVAILABLE = False
    get_latest_boc_values = None
    get_latest_group_observation = None
    get_group_metadata = None
    BocApiError = Exception

# Import the Alberta client
try:
    from utils.alberta_client import (
        get_alberta_economic_indicators as _get_alberta_indicators,
        get_alberta_indicator,
        get_indicator_metadata as get_alberta_indicator_metadata,
        get_all_indicator_keys as get_all_alberta_indicator_keys,
        get_indicators_by_category_grouped as get_alberta_indicators_by_category,
        clear_cache as clear_alberta_cache,
        get_cache_info as get_alberta_cache_info,
        AlbertaApiError,
        ALBERTA_INDICATORS,
    )
    ALBERTA_CLIENT_AVAILABLE = True
except ImportError:
    ALBERTA_CLIENT_AVAILABLE = False
    _get_alberta_indicators = None
    get_alberta_indicator = None
    get_alberta_indicator_metadata = None
    get_all_alberta_indicator_keys = None
    get_alberta_indicators_by_category = None
    clear_alberta_cache = None
    get_alberta_cache_info = None
    AlbertaApiError = Exception
    ALBERTA_INDICATORS = {}

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
    except yaml.YAMLError as e:
        logger.warning(f"YAML parsing error in BoC config {path}: {e}. Using defaults.")
        return _get_default_config()
    except PermissionError as e:
        logger.warning(f"Permission denied reading BoC config {path}: {e}. Using defaults.")
        return _get_default_config()
    except Exception as e:
        logger.warning(f"Unexpected error loading BoC config from {path}: {e}. Using defaults.")
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


def is_boc_group_context_enabled() -> bool:
    """Check if BoC group-based context fetching is enabled."""
    config = get_boc_config()
    return (
        config.get("enable_boc_group_context", False) 
        and BOC_GROUPS_AVAILABLE
        and is_boc_live_enabled()
    )


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
    # Default std if not provided or invalid
    DEFAULT_STD = 1.0
    
    if std <= 0:
        logger.warning(
            f"Invalid std ({std}) for z-score calculation, using default={DEFAULT_STD}. "
            "Check historical_stats configuration."
        )
        std = DEFAULT_STD
    
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
    
    if total_weight == 0:
        logger.warning("No valid BoC indicators (total_weight=0), using fallback")
        return _get_fallback_sentiment(run_date, city, config, details)
    
    if total_weight < 0:
        logger.error(f"Negative total_weight ({total_weight}) - check weight configuration")
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


# =============================================================================
# MACRO CONTEXT SNAPSHOT (GROUP-BASED DATA)
# =============================================================================


def get_macro_context_snapshot() -> Dict[str, Any]:
    """
    Get a snapshot of macro economic context from BoC groups.
    
    This function fetches data from configured BoC groups (FX rates, BCPI, CEER)
    to provide additional context for diagnostics and display. This data is NOT
    used in the main economic sentiment calculation - it's purely supplemental.
    
    When group calls fail, the function returns a partial result with available
    data and does NOT impact the main scoring pipeline.
    
    Returns:
        Dictionary with:
        - "available": bool - whether any group data was fetched
        - "groups": dict - mapping of group keys to their data
        - "highlighted": list - key series values formatted for display
        - "fetched_at": str - ISO timestamp of the fetch
        
    Example:
        >>> snapshot = get_macro_context_snapshot()
        >>> if snapshot["available"]:
        ...     for item in snapshot["highlighted"]:
        ...         print(f"{item['label']}: {item['formatted_value']}")
    """
    result = {
        "available": False,
        "groups": {},
        "highlighted": [],
        "fetched_at": None,
        "errors": [],
    }
    
    # Check if group context is enabled
    if not is_boc_group_context_enabled():
        logger.debug("BoC group context is disabled")
        result["message"] = "BoC group context is disabled"
        return result
    
    config = get_boc_config()
    boc_groups_config = config.get("boc_groups", {})
    macro_context_groups = config.get("macro_context_groups", [])
    group_display = config.get("group_display", {})
    highlight_series = group_display.get("highlight_series", {})
    formats = group_display.get("formats", {})
    
    if not macro_context_groups:
        logger.debug("No macro context groups configured")
        result["message"] = "No macro context groups configured"
        return result
    
    from datetime import datetime, timezone
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    
    # Fetch each configured group
    for group_key in macro_context_groups:
        group_id = boc_groups_config.get(group_key)
        if not group_id:
            logger.debug(f"Group key '{group_key}' not found in config")
            continue
        
        try:
            # Fetch group data
            values = get_latest_group_observation(group_id)
            
            if values:
                # Get metadata for the group
                metadata = get_group_metadata(group_id)
                
                result["groups"][group_key] = {
                    "group_id": group_id,
                    "label": metadata.get("label", group_id) if metadata else group_id,
                    "description": metadata.get("description", "") if metadata else "",
                    "series": values,
                    "series_count": len(values),
                }
                
                # Extract highlighted series for display
                for series_id, value in values.items():
                    if series_id in highlight_series and value is not None:
                        fmt = formats.get(series_id, "{:.4f}")
                        try:
                            formatted = fmt.format(value)
                        except (ValueError, KeyError):
                            formatted = str(value)
                        
                        result["highlighted"].append({
                            "series_id": series_id,
                            "label": highlight_series[series_id],
                            "value": value,
                            "formatted_value": formatted,
                            "group_key": group_key,
                            "group_id": group_id,
                        })
                
                result["available"] = True
                logger.debug(f"Fetched macro context from group {group_id}: {len(values)} series")
                
        except Exception as e:
            # Log warning but continue - group failures should not impact scoring
            error_msg = f"Failed to fetch group {group_id}: {str(e)}"
            logger.warning(error_msg)
            result["errors"].append(error_msg)
    
    if not result["available"]:
        result["message"] = "No group data available"
    
    return result


def get_macro_context_display() -> Dict[str, Any]:
    """
    Get macro context formatted for UI display.
    
    Returns a structure optimized for the Streamlit UI, with grouped
    indicators and formatted values.
    
    Returns:
        Dictionary with:
        - "available": bool
        - "sections": list of sections, each with label and items
        - "last_updated": str - human-readable timestamp
    """
    snapshot = get_macro_context_snapshot()
    
    display = {
        "available": snapshot.get("available", False),
        "sections": [],
        "last_updated": None,
        "message": snapshot.get("message"),
    }
    
    if not display["available"]:
        return display
    
    # Format timestamp for display
    fetched_at = snapshot.get("fetched_at")
    if fetched_at:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(fetched_at.replace('Z', '+00:00'))
            display["last_updated"] = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            display["last_updated"] = fetched_at
    
    # Group highlighted items by group_key for display
    highlighted = snapshot.get("highlighted", [])
    groups = snapshot.get("groups", {})
    
    # Create sections based on groups that have data
    for group_key, group_data in groups.items():
        section_items = [
            item for item in highlighted
            if item.get("group_key") == group_key
        ]
        
        if section_items:
            display["sections"].append({
                "group_key": group_key,
                "group_id": group_data.get("group_id"),
                "label": group_data.get("label", group_key),
                "items": section_items,
            })
    
    return display


# =============================================================================
# ALBERTA ECONOMIC INDICATORS INTEGRATION
# =============================================================================

_alberta_config: Optional[dict] = None


def _get_alberta_config_path() -> Path:
    """Get path to the Alberta economic config file."""
    return Path(__file__).parent.parent / "config" / "economic_alberta.yaml"


def load_alberta_config(config_path: Optional[str] = None) -> dict:
    """
    Load Alberta economic configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
        
    Returns:
        Dictionary with configuration settings
    """
    global _alberta_config
    
    if config_path is None:
        path = _get_alberta_config_path()
    else:
        path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Alberta config file not found at {path}. Using defaults.")
        return _get_default_alberta_config()
    
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded Alberta config from {path}")
        _alberta_config = config
        return config
    except yaml.YAMLError as e:
        logger.warning(f"YAML parsing error in Alberta config {path}: {e}. Using defaults.")
        return _get_default_alberta_config()
    except PermissionError as e:
        logger.warning(f"Permission denied reading Alberta config {path}: {e}. Using defaults.")
        return _get_default_alberta_config()
    except Exception as e:
        logger.warning(f"Unexpected error loading Alberta config from {path}: {e}. Using defaults.")
        return _get_default_alberta_config()


def _get_default_alberta_config() -> dict:
    """Return default configuration when config file is unavailable."""
    return {
        "use_alberta_live_data": False,  # Disabled by default when no config
        "fallback_mode": "neutral",
        "alberta_indicators": {},
        "sentiment_calculation": {
            "min_factor": 0.85,
            "max_factor": 1.15,
            "neutral": 1.0,
            "sensitivity": 0.08,
        },
    }


def get_alberta_config() -> dict:
    """Get the loaded Alberta config, loading it if necessary."""
    global _alberta_config
    if _alberta_config is None:
        _alberta_config = load_alberta_config()
    return _alberta_config


def is_alberta_live_enabled() -> bool:
    """Check if Alberta live data integration is enabled."""
    config = get_alberta_config()
    return config.get("use_alberta_live_data", False) and ALBERTA_CLIENT_AVAILABLE


def get_alberta_economic_indicators(
    use_cache: bool = True
) -> Dict[str, Optional[float]]:
    """
    Get the latest values for all Alberta economic indicators.
    
    This is a wrapper around the Alberta client that provides the same
    interface as the BoC indicator fetching functions.
    
    Args:
        use_cache: Whether to use cached values (default True)
        
    Returns:
        Dictionary mapping indicator keys (ab_*) to their values (or None if unavailable)
        
    Example:
        >>> indicators = get_alberta_economic_indicators()
        >>> unemployment = indicators.get("ab_unemployment_rate")
        >>> wcs_price = indicators.get("ab_wcs_oil_price")
    """
    if not ALBERTA_CLIENT_AVAILABLE:
        logger.warning("Alberta client not available")
        return {}
    
    if not is_alberta_live_enabled():
        logger.debug("Alberta live data is disabled")
        return {}
    
    return _get_alberta_indicators(use_cache=use_cache)


def fetch_alberta_indicators() -> Tuple[Dict[str, Optional[float]], bool]:
    """
    Fetch current values for all configured Alberta indicators.
    
    This function mirrors the interface of fetch_boc_indicators() for
    consistency with the existing codebase.
    
    Returns:
        Tuple of (values_dict, success_flag)
        - values_dict: Maps indicator keys to their current values (or None if failed)
        - success_flag: True if at least some values were fetched
    """
    if not ALBERTA_CLIENT_AVAILABLE:
        logger.warning("Alberta client not available")
        return {}, False
    
    if not is_alberta_live_enabled():
        logger.info("Alberta live data is disabled")
        return {}, False
    
    try:
        values = _get_alberta_indicators(use_cache=True)
    except Exception as e:
        logger.warning(f"Error fetching Alberta values: {e}")
        return {}, False
    
    # Check if we got at least some values
    success = any(v is not None for v in values.values())
    
    if success:
        logger.info(f"Fetched {sum(1 for v in values.values() if v is not None)} Alberta indicators")
    else:
        logger.warning("Failed to fetch any Alberta indicators")
    
    return values, success


def compute_alberta_economic_sentiment(
    run_date: Optional[date] = None,
    city: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute economic sentiment scalar using live Alberta data.
    
    This function:
    1. Fetches latest values for configured Alberta indicators
    2. Standardizes each value relative to historical norms
    3. Combines them into a weighted sentiment scalar
    4. Falls back to neutral if Alberta data is unavailable
    
    The result is a scalar typically in range [0.85, 1.15] where:
    - 1.0 = neutral economic conditions
    - > 1.0 = favorable conditions (may boost ticket sales)
    - < 1.0 = unfavorable conditions (may reduce ticket sales)
    
    Args:
        run_date: Date for the forecast (currently unused, for API consistency)
        city: City for regional adjustments (currently unused)
        
    Returns:
        Tuple of (sentiment_factor, details_dict)
        - sentiment_factor: Float typically in [0.85, 1.15]
        - details_dict: Dictionary with indicator values, source info, etc.
    """
    config = get_alberta_config()
    calc_config = config.get("sentiment_calculation", {})
    
    min_factor = calc_config.get("min_factor", 0.85)
    max_factor = calc_config.get("max_factor", 1.15)
    neutral = calc_config.get("neutral", 1.0)
    sensitivity = calc_config.get("sensitivity", 0.08)
    historical_stats = calc_config.get("historical_stats", {})
    
    details = {
        "source": None,
        "indicators": {},
        "factor": neutral,
        "alberta_available": False,
        "fallback_used": False,
    }
    
    # Check if Alberta live data is enabled
    if not is_alberta_live_enabled():
        logger.debug("Alberta live data is disabled, using neutral fallback")
        details["fallback_used"] = True
        details["source"] = "neutral_fallback"
        return neutral, details
    
    # Fetch Alberta indicators
    values, success = fetch_alberta_indicators()
    
    if not success:
        logger.info("Alberta data unavailable, using neutral fallback")
        details["fallback_used"] = True
        details["source"] = "neutral_fallback"
        return neutral, details
    
    details["alberta_available"] = True
    details["source"] = "alberta_live"
    
    # Calculate weighted z-score
    alberta_indicators = config.get("alberta_indicators", {})
    weighted_z_sum = 0.0
    total_weight = 0.0
    
    for key, indicator_config in alberta_indicators.items():
        value = values.get(key)
        
        if value is None:
            continue
        
        weight = indicator_config.get("weight", 0.0)
        direction = indicator_config.get("direction", "positive")
        baseline = indicator_config.get("baseline", 0.0)
        
        # Get historical stats for standardization
        stats = historical_stats.get(key, {})
        mean = stats.get("mean", baseline)
        std = stats.get("std", 1.0)
        
        # Compute z-score (reusing the existing function)
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
    
    if total_weight == 0:
        logger.warning("No valid Alberta indicators (total_weight=0), using neutral")
        details["fallback_used"] = True
        details["source"] = "neutral_fallback"
        return neutral, details
    
    if total_weight < 0:
        logger.error(f"Negative total_weight ({total_weight}) - check weight configuration")
        details["fallback_used"] = True
        details["source"] = "neutral_fallback"
        return neutral, details
    
    # Normalize and convert to factor
    average_z = weighted_z_sum / total_weight
    
    # Apply sensitivity to convert z-score to factor
    factor = neutral + (average_z * sensitivity)
    
    # Clip to allowed range
    factor = float(np.clip(factor, min_factor, max_factor))
    
    details["factor"] = factor
    details["weighted_z"] = average_z
    
    logger.info(f"Alberta economic sentiment factor: {factor:.3f} (z={average_z:.2f})")
    
    return factor, details


def get_alberta_indicator_display() -> Dict[str, Any]:
    """
    Get Alberta indicator values formatted for UI display.
    
    Returns:
        Dictionary with:
        - "available": bool - whether Alberta data is available
        - "indicators": list of dicts with label, value, formatted_value
        - "sentiment_factor": float - the computed sentiment
        - "sentiment_label": str - human-readable label
    """
    config = get_alberta_config()
    display_config = config.get("display", {})
    
    if not is_alberta_live_enabled():
        return {
            "available": False,
            "message": "Alberta live data is disabled",
        }
    
    # Fetch indicators and compute sentiment
    factor, details = compute_alberta_economic_sentiment()
    
    if not details.get("alberta_available", False):
        return {
            "available": False,
            "message": "Alberta data temporarily unavailable",
            "fallback_used": details.get("fallback_used", True),
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
        "source": details.get("source", "alberta_live"),
    }


# =============================================================================
# COMBINED ECONOMIC CONTEXT (BOC + ALBERTA)
# =============================================================================


def get_current_economic_context(
    include_boc: bool = True,
    include_alberta: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Get a comprehensive snapshot of current economic context from all sources.
    
    This function combines indicators from:
    - Bank of Canada (BOC): Interest rates, bond yields, commodity indices
    - Alberta Economic Dashboard: Employment, oil prices, consumer spending
    
    The result provides a unified view of economic conditions that can be
    used by the title scoring application.
    
    Args:
        include_boc: Whether to include BOC indicators (default True)
        include_alberta: Whether to include Alberta indicators (default True)
        use_cache: Whether to use cached values (default True)
        
    Returns:
        Dictionary with:
        - "boc": Dict of BOC indicator values (or None if disabled/unavailable)
        - "alberta": Dict of Alberta indicator values (or None if disabled/unavailable)
        - "combined_sentiment": Float - weighted combination of sentiments
        - "fetched_at": ISO timestamp of the fetch
        - "sources_available": List of sources that provided data
    """
    from datetime import datetime, timezone
    
    result = {
        "boc": None,
        "alberta": None,
        "boc_sentiment": None,
        "alberta_sentiment": None,
        "combined_sentiment": 1.0,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources_available": [],
    }
    
    # Fetch BOC data
    if include_boc and is_boc_live_enabled():
        boc_factor, boc_details = compute_boc_economic_sentiment()
        if boc_details.get("boc_available", False):
            result["boc"] = {
                key: ind.get("value") 
                for key, ind in boc_details.get("indicators", {}).items()
            }
            result["boc_sentiment"] = boc_factor
            result["sources_available"].append("boc")
    
    # Fetch Alberta data
    if include_alberta and is_alberta_live_enabled():
        ab_factor, ab_details = compute_alberta_economic_sentiment()
        if ab_details.get("alberta_available", False):
            result["alberta"] = {
                key: ind.get("value")
                for key, ind in ab_details.get("indicators", {}).items()
            }
            result["alberta_sentiment"] = ab_factor
            result["sources_available"].append("alberta")
    
    # Compute combined sentiment
    sentiments = []
    weights = []
    
    if result["boc_sentiment"] is not None:
        sentiments.append(result["boc_sentiment"])
        weights.append(0.4)  # 40% weight for BOC
    
    if result["alberta_sentiment"] is not None:
        sentiments.append(result["alberta_sentiment"])
        weights.append(0.6)  # 60% weight for Alberta (more regional relevance)
    
    if sentiments:
        total_weight = sum(weights)
        result["combined_sentiment"] = sum(
            s * w for s, w in zip(sentiments, weights)
        ) / total_weight
    
    return result


def get_all_economic_indicators(use_cache: bool = True) -> Dict[str, Optional[float]]:
    """
    Get all available economic indicators from all sources.
    
    This function fetches indicators from both BOC and Alberta sources
    and returns them in a single dictionary with their respective keys.
    
    BOC indicators use their series IDs or config keys.
    Alberta indicators use their ab_* keys.
    
    Args:
        use_cache: Whether to use cached values (default True)
        
    Returns:
        Dictionary mapping indicator keys to values (or None if unavailable)
    """
    all_indicators: Dict[str, Optional[float]] = {}
    
    # Fetch BOC indicators
    if is_boc_live_enabled():
        boc_values, _ = fetch_boc_indicators()
        for key, value in boc_values.items():
            all_indicators[f"boc_{key}"] = value
    
    # Fetch Alberta indicators
    if is_alberta_live_enabled():
        alberta_values = get_alberta_economic_indicators(use_cache=use_cache)
        all_indicators.update(alberta_values)
    
    return all_indicators
