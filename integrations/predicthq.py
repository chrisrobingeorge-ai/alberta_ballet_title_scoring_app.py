"""
PredictHQ API Client

This module provides a read-only client for the PredictHQ Events API to fetch
event data, predicted attendance, and demand intelligence features for
demand forecasting in Alberta (Calgary and Edmonton).

API Documentation:
    - Events API: https://docs.predicthq.com/api/events
    - Features API: https://docs.predicthq.com/api/features
    - Beam (Relevancy): https://docs.predicthq.com/api/beam

Environment Variables:
    PREDICTHQ_API_KEY: PredictHQ API access token (required)

Rate Limits:
    - Varies by subscription tier. The client includes built-in retry logic.

Key ML-Ready Features:
    - phq_attendance_sum: Total predicted attendance for events in a time window
    - phq_attendance_sports: Attendance for sports events
    - phq_attendance_concerts: Attendance for concerts/performing arts
    - phq_event_count: Count of significant events (rank >= threshold)
    - phq_rank_max: Maximum event rank in the time window
    - phq_rank_avg: Average event rank in the time window
    - phq_holidays_flag: Whether holidays overlap the run window
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# API Constants
PHQ_BASE_URL = "https://api.predicthq.com/v1"
PHQ_DEFAULT_RATE_LIMIT_DELAY = 0.2  # seconds between requests
PHQ_DEFAULT_PAGE_SIZE = 100
PHQ_MAX_RETRIES = 3
PHQ_RETRY_BACKOFF = 2.0

# Alberta city locations (lat, lon) for event searches
ALBERTA_LOCATIONS = {
    "Calgary": {"lat": 51.0447, "lon": -114.0719, "radius": "50km"},
    "Edmonton": {"lat": 53.5461, "lon": -113.4938, "radius": "50km"},
}

# PredictHQ event categories relevant for demand analysis
PHQ_RELEVANT_CATEGORIES = [
    "concerts",
    "sports",
    "performing-arts",
    "community",
    "festivals",
    "expos",
    "conferences",
]

PHQ_HOLIDAY_CATEGORIES = [
    "public-holidays",
    "school-holidays",
    "observances",
]

PHQ_WEATHER_CATEGORIES = [
    "severe-weather",
]


class PredictHQError(Exception):
    """Base exception for PredictHQ API errors."""
    pass


class PredictHQAuthError(PredictHQError):
    """Raised when authentication fails (401/403)."""
    pass


class PredictHQRateLimitError(PredictHQError):
    """Raised when rate limit is exceeded (429)."""
    pass


class PredictHQNotFoundError(PredictHQError):
    """Raised when a resource is not found (404)."""
    pass


@dataclass
class PredictHQEvent:
    """Represents a PredictHQ event."""
    event_id: str
    title: str
    category: str
    labels: List[str] = field(default_factory=list)
    rank: int = 0
    local_rank: Optional[int] = None
    phq_attendance: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    duration: Optional[int] = None
    timezone: Optional[str] = None
    location: Optional[List[float]] = None
    place_hierarchies: List[str] = field(default_factory=list)
    scope: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictHQFeatures:
    """Aggregated PredictHQ features for a date range and location."""
    city: str
    start_date: str
    end_date: str
    phq_attendance_sum: int = 0
    phq_attendance_sports: int = 0
    phq_attendance_concerts: int = 0
    phq_attendance_performing_arts: int = 0
    phq_event_count: int = 0
    phq_rank_max: int = 0
    phq_rank_avg: float = 0.0
    phq_holidays_flag: bool = False
    phq_severe_weather_flag: bool = False
    phq_event_spend: float = 0.0
    phq_demand_impact_score: float = 0.0
    events: List[PredictHQEvent] = field(default_factory=list)


class PredictHQClient:
    """
    Client for PredictHQ Events API.
    
    This client provides read-only access to event and demand intelligence data.
    It handles pagination, rate limiting, and HTTP error retries.
    
    Example:
        >>> client = PredictHQClient(api_key="your_access_token")
        >>> features = client.get_features_for_run(
        ...     city="Calgary",
        ...     start_date="2024-12-15",
        ...     end_date="2024-12-22"
        ... )
        >>> print(f"Total attendance: {features.phq_attendance_sum}")
        >>> print(f"Event count: {features.phq_event_count}")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = PHQ_BASE_URL,
        rate_limit_delay: float = PHQ_DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = PHQ_MAX_RETRIES,
    ):
        """
        Initialize the PredictHQ client.
        
        Args:
            api_key: PredictHQ API access token. If not provided, reads from 
                     PREDICTHQ_API_KEY env var.
            base_url: Base URL for the API (default: v1 API).
            rate_limit_delay: Seconds to wait between requests.
            max_retries: Maximum number of retry attempts for failed requests.
            
        Raises:
            PredictHQAuthError: If no API key is provided or found.
        """
        self.api_key = api_key or os.getenv("PREDICTHQ_API_KEY")
        if not self.api_key:
            raise PredictHQAuthError(
                "PredictHQ API key not provided. "
                "Set PREDICTHQ_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time: float = 0
        self._request_count: int = 0
        
        logger.info(f"Initialized PredictHQClient with base_url={self.base_url}")
    
    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the PredictHQ API.
        
        Args:
            endpoint: API endpoint (e.g., "/events")
            params: Query parameters
            method: HTTP method (default: GET)
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            PredictHQAuthError: On 401/403 responses
            PredictHQRateLimitError: On 429 responses
            PredictHQNotFoundError: On 404 responses
            PredictHQError: On other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            self._last_request_time = time.time()
            self._request_count += 1
            
            try:
                logger.debug(f"Request #{self._request_count}: {method} {url}")
                response = requests.request(
                    method,
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise PredictHQAuthError(
                        "Authentication failed (401). Check your PREDICTHQ_API_KEY."
                    )
                elif response.status_code == 403:
                    raise PredictHQAuthError(
                        "Access forbidden (403). Your API key may lack required permissions."
                    )
                elif response.status_code == 404:
                    raise PredictHQNotFoundError(
                        f"Resource not found (404): {endpoint}"
                    )
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = PHQ_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise PredictHQRateLimitError(
                        "Rate limit exceeded (429). Try again later."
                    )
                else:
                    if attempt < self.max_retries - 1:
                        wait_time = PHQ_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(
                            f"HTTP {response.status_code}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    raise PredictHQError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = PHQ_RETRY_BACKOFF ** (attempt + 1)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise PredictHQError(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise PredictHQError("Max retries exceeded")
    
    def _paginate(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle pagination for list endpoints.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_results: Maximum number of results to fetch (None for all)
            
        Returns:
            List of all items from all pages
        """
        all_items: List[Dict[str, Any]] = []
        params["limit"] = min(params.get("limit", PHQ_DEFAULT_PAGE_SIZE), PHQ_DEFAULT_PAGE_SIZE)
        params["offset"] = 0
        
        while True:
            response = self._make_request(endpoint, params)
            
            items = response.get("results", [])
            if not items:
                break
            
            all_items.extend(items)
            
            # Check if we've reached the limit
            if max_results and len(all_items) >= max_results:
                all_items = all_items[:max_results]
                break
            
            # Check pagination
            next_url = response.get("next")
            count = response.get("count", 0)
            
            logger.debug(
                f"Fetched {len(items)} items, total: {len(all_items)}/{count}"
            )
            
            if not next_url or len(all_items) >= count:
                break
            
            params["offset"] = params["offset"] + params["limit"]
        
        logger.info(f"Pagination complete: {len(all_items)} items")
        return all_items
    
    def search_events(
        self,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius: str = "50km",
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        categories: Optional[List[str]] = None,
        min_rank: Optional[int] = None,
        state: str = "AB",
        country: str = "CA",
        max_results: Optional[int] = None,
    ) -> List[PredictHQEvent]:
        """
        Search for events.
        
        Args:
            city: City name (Calgary or Edmonton). If provided, uses predefined coordinates.
            lat: Latitude for location-based search (used if city not provided).
            lon: Longitude for location-based search (used if city not provided).
            radius: Search radius (e.g., "50km", "30mi").
            start_date: Start date for event search (ISO 8601 or date object).
            end_date: End date for event search (ISO 8601 or date object).
            categories: List of event categories to filter by.
            min_rank: Minimum event rank (0-100).
            state: State/province code (default: "AB" for Alberta).
            country: Country code (default: "CA" for Canada).
            max_results: Maximum number of results to return.
            
        Returns:
            List of PredictHQEvent objects.
        """
        params: Dict[str, Any] = {}
        
        # Location filtering
        if city and city in ALBERTA_LOCATIONS:
            loc = ALBERTA_LOCATIONS[city]
            params["within"] = f"{loc['radius']}@{loc['lat']},{loc['lon']}"
        elif lat is not None and lon is not None:
            params["within"] = f"{radius}@{lat},{lon}"
        
        # State and country filtering
        if state:
            params["state"] = state
        if country:
            params["country"] = country
        
        # Date filtering
        if start_date:
            if isinstance(start_date, (date, datetime)):
                start_date = start_date.isoformat()[:10]
            params["active.gte"] = start_date
        
        if end_date:
            if isinstance(end_date, (date, datetime)):
                end_date = end_date.isoformat()[:10]
            params["active.lte"] = end_date
        
        # Category filtering
        if categories:
            params["category"] = ",".join(categories)
        
        # Rank filtering
        if min_rank is not None:
            params["rank.gte"] = min_rank
        
        # Sort by rank descending (most impactful events first)
        params["sort"] = "-rank"
        
        logger.info(f"Searching events with params: {params}")
        raw_events = self._paginate("/events", params, max_results)
        
        events = []
        for raw in raw_events:
            event = self._parse_event(raw)
            events.append(event)
        
        logger.info(f"Found {len(events)} events")
        return events
    
    def get_features_for_run(
        self,
        city: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        min_rank: int = 30,
    ) -> PredictHQFeatures:
        """
        Get aggregated ML-ready features for a production run window.
        
        This is the main method for extracting PredictHQ features for demand forecasting.
        It fetches events in the specified date range and city, then aggregates them
        into features suitable for ML models.
        
        Args:
            city: City name ("Calgary" or "Edmonton").
            start_date: Run start date.
            end_date: Run end date.
            min_rank: Minimum rank for events to include in count (default: 30).
            
        Returns:
            PredictHQFeatures object with aggregated features.
        """
        # Convert dates to string format
        if isinstance(start_date, (date, datetime)):
            start_date_str = start_date.isoformat()[:10]
        else:
            start_date_str = start_date
        
        if isinstance(end_date, (date, datetime)):
            end_date_str = end_date.isoformat()[:10]
        else:
            end_date_str = end_date
        
        # Initialize features object
        features = PredictHQFeatures(
            city=city,
            start_date=start_date_str,
            end_date=end_date_str,
        )
        
        # Fetch all relevant events
        all_events = self.search_events(
            city=city,
            start_date=start_date_str,
            end_date=end_date_str,
            categories=PHQ_RELEVANT_CATEGORIES,
        )
        
        # Fetch holiday events
        holiday_events = self.search_events(
            city=city,
            start_date=start_date_str,
            end_date=end_date_str,
            categories=PHQ_HOLIDAY_CATEGORIES,
        )
        
        # Fetch severe weather events
        weather_events = self.search_events(
            city=city,
            start_date=start_date_str,
            end_date=end_date_str,
            categories=PHQ_WEATHER_CATEGORIES,
        )
        
        # Aggregate features from regular events
        ranks = []
        for event in all_events:
            attendance = event.phq_attendance or 0
            features.phq_attendance_sum += attendance
            
            # Category-specific attendance
            if event.category == "sports":
                features.phq_attendance_sports += attendance
            elif event.category == "concerts":
                features.phq_attendance_concerts += attendance
            elif event.category == "performing-arts":
                features.phq_attendance_performing_arts += attendance
            
            # Track ranks for significant events
            if event.rank >= min_rank:
                features.phq_event_count += 1
                ranks.append(event.rank)
            
            # Track max rank
            if event.rank > features.phq_rank_max:
                features.phq_rank_max = event.rank
        
        # Calculate average rank
        if ranks:
            features.phq_rank_avg = sum(ranks) / len(ranks)
        
        # Check for holidays
        features.phq_holidays_flag = len(holiday_events) > 0
        
        # Check for severe weather
        features.phq_severe_weather_flag = len(weather_events) > 0
        
        # Calculate composite demand impact score
        # This is a custom weighted score combining attendance and event significance
        # Higher score = more competing events = potentially lower demand for ballet
        if features.phq_attendance_sum > 0 or features.phq_event_count > 0:
            # Normalize attendance to a 0-100 scale (assuming max ~500k for a major event period)
            attendance_score = min(100, features.phq_attendance_sum / 5000)
            # Weight: 60% attendance impact, 40% event rank impact
            features.phq_demand_impact_score = (
                0.6 * attendance_score + 
                0.4 * features.phq_rank_avg
            )
        
        # Store all events for reference
        features.events = all_events + holiday_events + weather_events
        
        logger.info(
            f"Features for {city} ({start_date_str} to {end_date_str}): "
            f"attendance={features.phq_attendance_sum}, "
            f"events={features.phq_event_count}, "
            f"max_rank={features.phq_rank_max}"
        )
        
        return features
    
    def _parse_event(self, raw: Dict[str, Any]) -> PredictHQEvent:
        """Parse raw event data into a PredictHQEvent object."""
        # Extract location
        geo = raw.get("geo", {})
        geometry = geo.get("geometry", {})
        coordinates = geometry.get("coordinates")  # [lon, lat]
        
        return PredictHQEvent(
            event_id=raw.get("id", ""),
            title=raw.get("title", ""),
            category=raw.get("category", ""),
            labels=raw.get("labels", []),
            rank=raw.get("rank", 0),
            local_rank=raw.get("local_rank"),
            phq_attendance=raw.get("phq_attendance"),
            start=raw.get("start"),
            end=raw.get("end"),
            duration=raw.get("duration"),
            timezone=raw.get("timezone"),
            location=coordinates,
            place_hierarchies=raw.get("place_hierarchies", []),
            scope=raw.get("scope"),
            country=raw.get("country"),
            state=raw.get("state"),
            raw_data=raw,
        )
    
    def get_request_count(self) -> int:
        """Return the total number of API requests made."""
        return self._request_count


def get_predicthq_features_dict(
    features: PredictHQFeatures,
) -> Dict[str, Any]:
    """
    Convert PredictHQFeatures to a dictionary suitable for DataFrame construction.
    
    This is a helper function to convert the features dataclass to a flat dictionary
    that can be easily merged with other DataFrames for ML training.
    
    Args:
        features: PredictHQFeatures object
        
    Returns:
        Dictionary with feature names as keys
    """
    return {
        "city": features.city,
        "phq_start_date": features.start_date,
        "phq_end_date": features.end_date,
        "phq_attendance_sum": features.phq_attendance_sum,
        "phq_attendance_sports": features.phq_attendance_sports,
        "phq_attendance_concerts": features.phq_attendance_concerts,
        "phq_attendance_performing_arts": features.phq_attendance_performing_arts,
        "phq_event_count": features.phq_event_count,
        "phq_rank_max": features.phq_rank_max,
        "phq_rank_avg": features.phq_rank_avg,
        "phq_holidays_flag": int(features.phq_holidays_flag),
        "phq_severe_weather_flag": int(features.phq_severe_weather_flag),
        "phq_event_spend": features.phq_event_spend,
        "phq_demand_impact_score": features.phq_demand_impact_score,
    }
