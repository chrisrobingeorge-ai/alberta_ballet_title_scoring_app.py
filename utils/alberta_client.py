"""
Alberta Economic Dashboard API Client

This module provides a client for fetching live economic data from the Alberta
Economic Dashboard APIs (https://economicdashboard.alberta.ca/). No API key required.

The client fetches the latest values for economic series like:
- Unemployment rate in Alberta
- Employment rate and levels
- Average weekly earnings
- Consumer Price Index (CPI)
- WCS Oil Price
- Retail trade and restaurant sales
- Population and migration

These values are used as supplemental "live" context for the economic sentiment
adjustment in the title scoring application, alongside the existing Bank of Canada
indicators.

Usage:
    from utils.alberta_client import get_alberta_economic_indicators
    
    # Fetch all Alberta indicators
    indicators = get_alberta_economic_indicators()
    
    # Access individual values
    unemployment = indicators.get("ab_unemployment_rate")
    wcs_price = indicators.get("ab_wcs_oil_price")
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import threading

import requests

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

ALBERTA_API_BASE_URL = "https://api.economicdata.alberta.ca/api/data"

# Default timeout for API requests (seconds)
DEFAULT_TIMEOUT = 15

# Cache TTL - values are cached for the current day (refreshed after midnight UTC)
CACHE_TTL_HOURS = 24

# =============================================================================
# ALBERTA INDICATOR DEFINITIONS
# =============================================================================

# Each indicator has:
#   - api_code: The unique code for the Alberta Economic Dashboard API
#   - key: The feature key to expose (ab_*)
#   - description: Human-readable description
#   - category: Category for grouping

ALBERTA_INDICATORS = {
    # Labour & Income (core drivers)
    "ab_unemployment_rate": {
        "api_code": "c1fe936a-324a-4a37-bfde-eeb3bb3d7c8c",
        "description": "Unemployment rate in Alberta",
        "category": "labour",
    },
    "ab_employment_rate": {
        "api_code": "cc63aded-d078-46be-a033-f3b5b81b97ab",
        "description": "Employment rate in Alberta",
        "category": "labour",
    },
    "ab_employment_level": {
        "api_code": "fb0ee090-7984-48cb-838d-57201ab7ae8f",
        "description": "Employment in Alberta (level)",
        "category": "labour",
    },
    "ab_participation_rate": {
        "api_code": "cd92ac03-ef03-414f-8530-b5db01f288bc",
        "description": "Participation rate in Alberta",
        "category": "labour",
    },
    "ab_avg_weekly_earnings": {
        "api_code": "9814f9e0-6a0d-45a5-b2df-a29af0262fb7",
        "description": "Average Weekly Earnings in Alberta",
        "category": "labour",
    },
    # Prices / Real Income
    "ab_cpi": {
        "api_code": "38524b3e-56ce-4d65-bea6-723497919ac2",
        "description": "Consumer Price Index for Alberta",
        "category": "prices",
    },
    # Energy Sector Activity
    "ab_wcs_oil_price": {
        "api_code": "1da37895-ed56-405e-81de-26231ffc6472",
        "description": "WCS Oil Price",
        "category": "energy",
    },
    # Consumer Spending / Behaviour
    "ab_retail_trade": {
        "api_code": "e2940527-8d0e-4279-b0d9-84e2e8be53b9",
        "description": "Retail Trade in Alberta",
        "category": "consumer",
    },
    "ab_restaurant_sales": {
        "api_code": "b5992517-5ef3-410a-8c69-7adee7ddfe58",
        "description": "Restaurant Sales in Alberta",
        "category": "consumer",
    },
    "ab_air_passengers": {
        "api_code": "1c22431c-ad90-4614-a973-9c39f309d9c7",
        "description": "Air Passengers (YEG + YYC total)",
        "category": "consumer",
    },
    # Population & Migration
    "ab_net_migration": {
        "api_code": "1216c72b-887b-47cc-a143-77c7dcac8948",
        "description": "Net Migration into Alberta",
        "category": "population",
    },
    "ab_population_quarterly": {
        "api_code": "04bcf4ae-9975-49bf-9993-8243edd7ef67",
        "description": "Population (Quarterly) in Alberta",
        "category": "population",
    },
}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class AlbertaApiError(Exception):
    """Exception raised when Alberta Economic Dashboard API calls fail."""
    pass


class AlbertaDataUnavailableError(AlbertaApiError):
    """Exception raised when requested series data is not available."""
    pass


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================


class AlbertaCache:
    """
    Simple thread-safe in-memory cache for Alberta API responses.
    
    Values are cached with a daily TTL - they expire at midnight UTC or
    after CACHE_TTL_HOURS hours, whichever comes first.
    """
    
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[float]:
        """
        Get cached value if still valid.
        
        Args:
            key: The indicator key
            
        Returns:
            Cached value if valid, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, cached_at = self._cache[key]
            
            # Check if cache has expired
            now = datetime.now(timezone.utc)
            cache_date = cached_at.date()
            current_date = now.date()
            
            # Expire if different day OR if TTL exceeded
            if cache_date != current_date:
                del self._cache[key]
                return None
            
            hours_elapsed = (now - cached_at).total_seconds() / 3600
            if hours_elapsed > CACHE_TTL_HOURS:
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: float) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The indicator key
            value: The numeric value to cache
        """
        with self._lock:
            self._cache[key] = (value, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        with self._lock:
            return {
                "size": len(self._cache),
                "keys": list(self._cache.keys()),
            }


# Global cache instance
_cache = AlbertaCache()


# =============================================================================
# API FUNCTIONS
# =============================================================================


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.
    
    Tries multiple date formats commonly used by the Alberta API.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # Common date formats used by Alberta Economic Dashboard API
    date_formats = [
        "%Y-%m-%d",       # 2024-01-15
        "%Y-%m-%dT%H:%M:%S",  # 2024-01-15T00:00:00
        "%Y-%m-%dT%H:%M:%SZ", # 2024-01-15T00:00:00Z
        "%Y/%m/%d",       # 2024/01/15
        "%d/%m/%Y",       # 15/01/2024
        "%Y-%m",          # 2024-01 (monthly data)
        "%Y",             # 2024 (annual data)
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.debug(f"Could not parse date '{date_str}' with any known format")
    return None


def _parse_alberta_response(data: List[dict]) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse the Alberta Economic Dashboard API response and extract the latest observation.
    
    The API returns a list of observations, each with date and value fields.
    We need to find the most recent observation.
    
    Args:
        data: List of observation dictionaries from API response
        
    Returns:
        Tuple of (latest_date, latest_value) or (None, None) if parsing fails
    """
    if not data:
        return None, None
    
    latest_date_str = None
    latest_date_parsed = None
    latest_value = None
    
    # Find the observation with the most recent date
    for obs in data:
        # The API typically uses 'Date' or 'date' for the date field
        # and 'Value' or 'value' for the value field
        date_str = obs.get("Date") or obs.get("date")
        value = obs.get("Value") or obs.get("value")
        
        if date_str is None or value is None:
            continue
        
        # Parse the date for proper comparison
        parsed_date = _parse_date(str(date_str))
        if parsed_date is None:
            # Fallback to string comparison if parsing fails
            if latest_date_str is None or str(date_str) > latest_date_str:
                latest_date_str = str(date_str)
                latest_date_parsed = None
                # Convert value to float
                try:
                    if isinstance(value, (int, float)):
                        latest_value = float(value)
                    elif isinstance(value, str):
                        cleaned = value.strip().replace(",", "")
                        if cleaned:
                            latest_value = float(cleaned)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse value '{value}': {e}")
                    continue
            continue
        
        # Compare dates to find the latest
        if latest_date_parsed is None or parsed_date > latest_date_parsed:
            latest_date_str = str(date_str)
            latest_date_parsed = parsed_date
            # Convert value to float
            try:
                if isinstance(value, (int, float)):
                    latest_value = float(value)
                elif isinstance(value, str):
                    cleaned = value.strip().replace(",", "")
                    if cleaned:
                        latest_value = float(cleaned)
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse value '{value}': {e}")
                continue
    
    return latest_date_str, latest_value


def _fetch_indicator(
    api_code: str,
    timeout: int = DEFAULT_TIMEOUT
) -> Tuple[Optional[str], Optional[float]]:
    """
    Fetch the latest observation for an indicator from the Alberta Economic Dashboard API.
    
    Args:
        api_code: The unique API code for the indicator
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (latest_date, latest_value) or (None, None) if fetch failed
        
    Raises:
        AlbertaApiError: If API request fails
        AlbertaDataUnavailableError: If indicator has no data
    """
    url = f"{ALBERTA_API_BASE_URL}"
    params = {"code": api_code}
    
    logger.debug(f"Fetching Alberta indicator {api_code} from {url}")
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching Alberta indicator {api_code}")
        raise AlbertaApiError(f"Timeout fetching indicator {api_code}")
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code == 404:
            logger.warning(f"Alberta indicator {api_code} not found (404)")
            raise AlbertaDataUnavailableError(f"Indicator {api_code} not found")
        logger.warning(f"HTTP error fetching Alberta indicator {api_code}: {e}")
        raise AlbertaApiError(f"HTTP error for indicator {api_code}: {e}")
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error fetching Alberta indicator {api_code}: {e}")
        raise AlbertaApiError(f"Request error for indicator {api_code}: {e}")
    
    try:
        data = response.json()
    except ValueError as e:
        logger.warning(f"Invalid JSON response for Alberta indicator {api_code}: {e}")
        raise AlbertaApiError(f"Invalid JSON response for indicator {api_code}")
    
    # The API returns a list of observations
    if not isinstance(data, list):
        logger.info(f"Unexpected response format for Alberta indicator {api_code}")
        return None, None
    
    if not data:
        logger.info(f"No observations available for Alberta indicator {api_code}")
        return None, None
    
    latest_date, latest_value = _parse_alberta_response(data)
    
    if latest_value is not None:
        logger.debug(f"Fetched Alberta indicator {api_code}: {latest_value} (date: {latest_date})")
    else:
        logger.info(f"Could not parse value for Alberta indicator {api_code}")
    
    return latest_date, latest_value


def get_alberta_indicator(
    key: str,
    use_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[float]:
    """
    Get the latest value for a specific Alberta economic indicator.
    
    This function fetches the most recent observation for an Alberta indicator,
    using an in-memory cache to avoid excessive API calls. Values are
    cached for the current day.
    
    Args:
        key: The indicator key (e.g., "ab_unemployment_rate")
        use_cache: Whether to use cached values (default True)
        timeout: Request timeout in seconds (default 15)
        
    Returns:
        Float value if available, None if data unavailable or fetch failed
        
    Example:
        >>> rate = get_alberta_indicator("ab_unemployment_rate")
        >>> if rate is not None:
        ...     print(f"Alberta unemployment rate: {rate}%")
    """
    # Validate key
    if key not in ALBERTA_INDICATORS:
        logger.warning(f"Unknown Alberta indicator key: {key}")
        return None
    
    # Check cache first
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            logger.debug(f"Using cached value for Alberta indicator {key}: {cached}")
            return cached
    
    # Fetch from API
    indicator_config = ALBERTA_INDICATORS[key]
    api_code = indicator_config["api_code"]
    
    try:
        _, value = _fetch_indicator(api_code, timeout=timeout)
        
        if value is not None and use_cache:
            _cache.set(key, value)
        
        return value
        
    except AlbertaApiError as e:
        logger.warning(f"Failed to fetch Alberta indicator {key}: {e}")
        return None


def get_alberta_economic_indicators(
    use_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Optional[float]]:
    """
    Get the latest values for all Alberta economic indicators.
    
    This function fetches the most recent observations for all 12 Alberta
    indicators, using cached values where available.
    
    Args:
        use_cache: Whether to use cached values (default True)
        timeout: Request timeout in seconds (default 15)
        
    Returns:
        Dictionary mapping indicator keys to their values (or None if unavailable)
        
    Example:
        >>> indicators = get_alberta_economic_indicators()
        >>> for key, value in indicators.items():
        ...     if value is not None:
        ...         print(f"{key}: {value}")
    """
    results: Dict[str, Optional[float]] = {}
    
    for key in ALBERTA_INDICATORS:
        results[key] = get_alberta_indicator(
            key,
            use_cache=use_cache,
            timeout=timeout
        )
    
    # Log summary
    fetched = sum(1 for v in results.values() if v is not None)
    logger.info(f"Fetched {fetched}/{len(ALBERTA_INDICATORS)} Alberta indicators")
    
    return results


def get_alberta_indicators_by_category(
    category: str,
    use_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Optional[float]]:
    """
    Get Alberta indicators filtered by category.
    
    Args:
        category: Category to filter by (labour, prices, energy, consumer, population)
        use_cache: Whether to use cached values (default True)
        timeout: Request timeout in seconds (default 15)
        
    Returns:
        Dictionary mapping indicator keys to values for the specified category
    """
    results: Dict[str, Optional[float]] = {}
    
    for key, config in ALBERTA_INDICATORS.items():
        if config["category"] == category:
            results[key] = get_alberta_indicator(key, use_cache=use_cache, timeout=timeout)
    
    return results


def clear_cache() -> None:
    """
    Clear the Alberta data cache.
    
    This forces fresh API calls on the next request.
    """
    _cache.clear()
    logger.info("Alberta cache cleared")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.
    
    Returns:
        Dictionary with cache size and cached indicator keys
    """
    return _cache.get_cache_info()


def get_indicator_metadata(key: str) -> Optional[Dict[str, str]]:
    """
    Get metadata for an Alberta indicator.
    
    Args:
        key: The indicator key
        
    Returns:
        Dictionary with api_code, description, category, or None if not found
    """
    if key not in ALBERTA_INDICATORS:
        return None
    return ALBERTA_INDICATORS[key].copy()


def get_all_indicator_keys() -> List[str]:
    """
    Get a list of all available Alberta indicator keys.
    
    Returns:
        List of indicator keys (ab_*)
    """
    return list(ALBERTA_INDICATORS.keys())


def get_indicators_by_category_grouped() -> Dict[str, List[str]]:
    """
    Get all indicator keys grouped by category.
    
    Returns:
        Dictionary mapping category names to lists of indicator keys
    """
    categories: Dict[str, List[str]] = {}
    
    for key, config in ALBERTA_INDICATORS.items():
        category = config["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(key)
    
    return categories
