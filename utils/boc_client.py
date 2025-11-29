"""
Bank of Canada Valet API Client

This module provides a client for fetching live economic data from the Bank of Canada
Valet Web API (https://www.bankofcanada.ca/valet/). No API key is required.

The client is used to fetch the latest values for economic series like:
- Policy rates (overnight rate)
- Government bond yields
- Commodity price indices
- Inflation indicators (CPI)

These values are used as supplemental "live" context for the economic sentiment
adjustment in the title scoring application. Historical analysis and model training
continue to rely on the existing historical economic datasets.

Usage:
    from utils.boc_client import get_latest_boc_value, get_latest_boc_values

    # Fetch single series
    policy_rate = get_latest_boc_value("B114039")

    # Fetch multiple series at once
    values = get_latest_boc_values(["B114039", "BD.CDN.5YR.DQ.YLD"])
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from functools import lru_cache
import threading

import requests

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

BOC_VALET_BASE_URL = "https://www.bankofcanada.ca/valet"
BOC_OBSERVATIONS_URL = f"{BOC_VALET_BASE_URL}/observations/{{series_name}}/json"

# Default timeout for API requests (seconds)
DEFAULT_TIMEOUT = 10

# Cache TTL - values are cached for the current day (refreshed after midnight UTC)
# This prevents excessive API calls during a single session
CACHE_TTL_HOURS = 24

# Values considered as null/missing in BoC API responses
NULL_VALUE_STRINGS = frozenset({'', 'na', 'n/a', 'null', 'none'})

# Valid characters for BoC series names (alphanumeric, dots, underscores, dashes)
VALID_SERIES_NAME_PATTERN = r'^[A-Za-z0-9._-]+$'


# =============================================================================
# EXCEPTIONS
# =============================================================================


class BocApiError(Exception):
    """Exception raised when Bank of Canada API calls fail."""
    pass


class BocDataUnavailableError(BocApiError):
    """Exception raised when requested series data is not available."""
    pass


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================


class BocCache:
    """
    Simple thread-safe in-memory cache for BoC API responses.
    
    Values are cached with a daily TTL - they expire at midnight UTC or
    after CACHE_TTL_HOURS hours, whichever comes first.
    """
    
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {series_name: (value, timestamp)}
        self._lock = threading.Lock()
    
    def get(self, series_name: str) -> Optional[float]:
        """
        Get cached value if still valid.
        
        Args:
            series_name: The BoC series identifier
            
        Returns:
            Cached value if valid, None otherwise
        """
        with self._lock:
            if series_name not in self._cache:
                return None
            
            value, cached_at = self._cache[series_name]
            
            # Check if cache has expired
            now = datetime.now(timezone.utc)
            cache_date = cached_at.date()
            current_date = now.date()
            
            # Expire if different day OR if TTL exceeded
            if cache_date != current_date:
                del self._cache[series_name]
                return None
            
            hours_elapsed = (now - cached_at).total_seconds() / 3600
            if hours_elapsed > CACHE_TTL_HOURS:
                del self._cache[series_name]
                return None
            
            return value
    
    def set(self, series_name: str, value: float) -> None:
        """
        Store a value in the cache.
        
        Args:
            series_name: The BoC series identifier
            value: The numeric value to cache
        """
        with self._lock:
            self._cache[series_name] = (value, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        with self._lock:
            return {
                "size": len(self._cache),
                "series": list(self._cache.keys()),
            }


# Global cache instance
_cache = BocCache()


# =============================================================================
# API FUNCTIONS
# =============================================================================


def _parse_observation_value(observations: List[dict], series_name: str) -> Optional[float]:
    """
    Parse the numeric value from BoC observations response.
    
    Args:
        observations: List of observation dictionaries from API response
        series_name: The series name to extract value for
        
    Returns:
        Float value if found and parseable, None otherwise
    """
    if not observations:
        return None
    
    # Get the most recent observation (last in list, or first if sorted desc)
    # The BoC API returns observations sorted by date ascending
    latest = observations[-1] if observations else None
    
    if not latest:
        return None
    
    # The value is stored under the series name key
    raw_value = latest.get(series_name)
    
    if raw_value is None:
        return None
    
    # Handle various value formats
    try:
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            # Remove any whitespace and try to parse
            cleaned = raw_value.strip()
            if not cleaned or cleaned.lower() in NULL_VALUE_STRINGS:
                return None
            return float(cleaned)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse value '{raw_value}' for series {series_name}: {e}")
        return None
    
    return None


def _validate_series_name(series_name: str) -> bool:
    """
    Validate that a series name contains only safe characters.
    
    Args:
        series_name: The BoC series identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not series_name or not isinstance(series_name, str):
        return False
    return bool(re.match(VALID_SERIES_NAME_PATTERN, series_name))


def _fetch_from_api(series_name: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[float]:
    """
    Fetch the latest value for a series directly from the BoC Valet API.
    
    Args:
        series_name: The BoC series identifier (e.g., "B114039")
        timeout: Request timeout in seconds
        
    Returns:
        Float value if successful, None otherwise
        
    Raises:
        BocApiError: If API request fails
        BocDataUnavailableError: If series has no data
    """
    # Validate series name to prevent injection attacks
    if not _validate_series_name(series_name):
        logger.warning(f"Invalid series name: {series_name}")
        raise BocApiError(f"Invalid series name: {series_name}")
    
    url = BOC_OBSERVATIONS_URL.format(series_name=series_name)
    params = {"recent": "1"}  # Get only the most recent observation
    
    logger.debug(f"Fetching BoC series {series_name} from {url}")
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching BoC series {series_name}")
        raise BocApiError(f"Timeout fetching series {series_name}")
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code == 404:
            logger.warning(f"BoC series {series_name} not found (404)")
            raise BocDataUnavailableError(f"Series {series_name} not found")
        logger.warning(f"HTTP error fetching BoC series {series_name}: {e}")
        raise BocApiError(f"HTTP error for series {series_name}: {e}")
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error fetching BoC series {series_name}: {e}")
        raise BocApiError(f"Request error for series {series_name}: {e}")
    
    try:
        data = response.json()
    except ValueError as e:
        logger.warning(f"Invalid JSON response for BoC series {series_name}: {e}")
        raise BocApiError(f"Invalid JSON response for series {series_name}")
    
    # Extract observations from response
    observations = data.get("observations", [])
    
    if not observations:
        logger.info(f"No observations available for BoC series {series_name}")
        return None
    
    value = _parse_observation_value(observations, series_name)
    
    if value is not None:
        logger.debug(f"Fetched BoC series {series_name}: {value}")
    else:
        logger.info(f"Could not parse value for BoC series {series_name}")
    
    return value


def get_latest_boc_value(
    series_name: str,
    use_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[float]:
    """
    Get the latest value for a Bank of Canada series.
    
    This function fetches the most recent observation for a BoC series,
    using an in-memory cache to avoid excessive API calls. Values are
    cached for the current day.
    
    Args:
        series_name: The BoC series identifier (e.g., "B114039" for policy rate)
        use_cache: Whether to use cached values (default True)
        timeout: Request timeout in seconds (default 10)
        
    Returns:
        Float value if available, None if data unavailable or fetch failed
        
    Example:
        >>> rate = get_latest_boc_value("B114039")
        >>> if rate is not None:
        ...     print(f"Current policy rate: {rate}%")
    """
    # Check cache first
    if use_cache:
        cached = _cache.get(series_name)
        if cached is not None:
            logger.debug(f"Using cached value for BoC series {series_name}: {cached}")
            return cached
    
    # Fetch from API
    try:
        value = _fetch_from_api(series_name, timeout=timeout)
        
        if value is not None and use_cache:
            _cache.set(series_name, value)
        
        return value
        
    except BocApiError as e:
        logger.warning(f"Failed to fetch BoC series {series_name}: {e}")
        return None


def get_latest_boc_values(
    series_names: List[str],
    use_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Optional[float]]:
    """
    Get the latest values for multiple Bank of Canada series.
    
    This function fetches values for multiple series efficiently,
    using cached values where available.
    
    Args:
        series_names: List of BoC series identifiers
        use_cache: Whether to use cached values (default True)
        timeout: Request timeout in seconds (default 10)
        
    Returns:
        Dictionary mapping series names to their values (or None if unavailable)
        
    Example:
        >>> series = ["B114039", "BD.CDN.5YR.DQ.YLD", "A.BCPI"]
        >>> values = get_latest_boc_values(series)
        >>> for name, value in values.items():
        ...     print(f"{name}: {value}")
    """
    results: Dict[str, Optional[float]] = {}
    
    for series_name in series_names:
        results[series_name] = get_latest_boc_value(
            series_name,
            use_cache=use_cache,
            timeout=timeout
        )
    
    return results


def clear_cache() -> None:
    """
    Clear the BoC data cache.
    
    This forces fresh API calls on the next request.
    """
    _cache.clear()
    logger.info("BoC cache cleared")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.
    
    Returns:
        Dictionary with cache size and cached series names
    """
    return _cache.get_cache_info()


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON SERIES
# =============================================================================

# Common series identifiers for easy reference
SERIES_POLICY_RATE = "B114039"  # Target Rate
SERIES_CORRA = "AVG.INTWO"  # Canadian Overnight Repo Rate Average
SERIES_GOV_BOND_2YR = "BD.CDN.2YR.DQ.YLD"  # 2-year bond yield
SERIES_GOV_BOND_5YR = "BD.CDN.5YR.DQ.YLD"  # 5-year bond yield
SERIES_GOV_BOND_10YR = "BD.CDN.10YR.DQ.YLD"  # 10-year bond yield
SERIES_BCPI_TOTAL = "A.BCPI"  # Annual BCPI Total
SERIES_BCPI_ENERGY = "A.ENER"  # Annual BCPI Energy
SERIES_BCPI_EX_ENERGY = "A.BCNE"  # Annual BCPI Excluding Energy
SERIES_CPI_CORE = "ATOM_V41693242"  # CPIX core inflation


def get_policy_rate(use_cache: bool = True) -> Optional[float]:
    """Get the current Bank of Canada policy (target) rate."""
    return get_latest_boc_value(SERIES_POLICY_RATE, use_cache=use_cache)


def get_bond_yields(use_cache: bool = True) -> Dict[str, Optional[float]]:
    """
    Get current Government of Canada bond yields.
    
    Returns:
        Dictionary with keys '2yr', '5yr', '10yr' mapped to yield values
    """
    values = get_latest_boc_values(
        [SERIES_GOV_BOND_2YR, SERIES_GOV_BOND_5YR, SERIES_GOV_BOND_10YR],
        use_cache=use_cache
    )
    return {
        "2yr": values.get(SERIES_GOV_BOND_2YR),
        "5yr": values.get(SERIES_GOV_BOND_5YR),
        "10yr": values.get(SERIES_GOV_BOND_10YR),
    }


def get_commodity_indices(use_cache: bool = True) -> Dict[str, Optional[float]]:
    """
    Get current Bank of Canada commodity price indices.
    
    Returns:
        Dictionary with keys 'total', 'energy', 'ex_energy' mapped to index values
    """
    values = get_latest_boc_values(
        [SERIES_BCPI_TOTAL, SERIES_BCPI_ENERGY, SERIES_BCPI_EX_ENERGY],
        use_cache=use_cache
    )
    return {
        "total": values.get(SERIES_BCPI_TOTAL),
        "energy": values.get(SERIES_BCPI_ENERGY),
        "ex_energy": values.get(SERIES_BCPI_EX_ENERGY),
    }
