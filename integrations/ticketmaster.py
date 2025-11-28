"""
Ticketmaster API Client

This module provides a read-only client for the Ticketmaster Discovery and
Partner APIs to fetch event, venue, and performance schedule data.

API Documentation:
    - Discovery API: https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
    - Partner API: Contact Ticketmaster for access

Environment Variables:
    TM_API_KEY: Ticketmaster API key (required)
    TM_API_SECRET: Ticketmaster API secret (optional, for OAuth)

Rate Limits:
    - Discovery API: 5 requests/second, 5000 requests/day (free tier)
    - Partner API: Varies by contract
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# API Constants
TM_DISCOVERY_BASE_URL = "https://app.ticketmaster.com/discovery/v2"
TM_DEFAULT_RATE_LIMIT_DELAY = 0.25  # seconds between requests
TM_DEFAULT_PAGE_SIZE = 100
TM_MAX_RETRIES = 3
TM_RETRY_BACKOFF = 2.0


class TicketmasterError(Exception):
    """Base exception for Ticketmaster API errors."""
    pass


class TicketmasterAuthError(TicketmasterError):
    """Raised when authentication fails (401/403)."""
    pass


class TicketmasterRateLimitError(TicketmasterError):
    """Raised when rate limit is exceeded (429)."""
    pass


class TicketmasterNotFoundError(TicketmasterError):
    """Raised when a resource is not found (404)."""
    pass


@dataclass
class TicketmasterEvent:
    """Represents a Ticketmaster event."""
    event_id: str
    name: str
    url: Optional[str] = None
    start_date: Optional[str] = None
    start_time: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    venue_id: Optional[str] = None
    venue_capacity: Optional[int] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    country: Optional[str] = None
    price_ranges: List[Dict[str, Any]] = field(default_factory=list)
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TicketmasterVenue:
    """Represents a Ticketmaster venue."""
    venue_id: str
    name: str
    city: Optional[str] = None
    state_province: Optional[str] = None
    country: Optional[str] = None
    capacity: Optional[int] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


class TicketmasterClient:
    """
    Client for Ticketmaster Discovery API.
    
    This client provides read-only access to event and venue data.
    It handles pagination, rate limiting, and HTTP error retries.
    
    Example:
        >>> client = TicketmasterClient(api_key="your_key")
        >>> events = client.search_events(keyword="Nutcracker", city="Calgary")
        >>> for event in events:
        ...     print(f"{event.name} at {event.venue_name}")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = TM_DISCOVERY_BASE_URL,
        rate_limit_delay: float = TM_DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = TM_MAX_RETRIES,
    ):
        """
        Initialize the Ticketmaster client.
        
        Args:
            api_key: Ticketmaster API key. If not provided, reads from TM_API_KEY env var.
            base_url: Base URL for the API (default: Discovery API v2).
            rate_limit_delay: Seconds to wait between requests.
            max_retries: Maximum number of retry attempts for failed requests.
            
        Raises:
            TicketmasterAuthError: If no API key is provided or found.
        """
        self.api_key = api_key or os.getenv("TM_API_KEY")
        if not self.api_key:
            raise TicketmasterAuthError(
                "Ticketmaster API key not provided. "
                "Set TM_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time: float = 0
        self._request_count: int = 0
        
        logger.info(f"Initialized TicketmasterClient with base_url={self.base_url}")
    
    def _rate_limit_wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Ticketmaster API.
        
        Args:
            endpoint: API endpoint (e.g., "/events")
            params: Query parameters
            method: HTTP method (default: GET)
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            TicketmasterAuthError: On 401/403 responses
            TicketmasterRateLimitError: On 429 responses
            TicketmasterNotFoundError: On 404 responses
            TicketmasterError: On other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key
        
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            self._last_request_time = time.time()
            self._request_count += 1
            
            try:
                logger.debug(f"Request #{self._request_count}: {method} {url}")
                response = requests.request(method, url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise TicketmasterAuthError(
                        "Authentication failed (401). Check your TM_API_KEY."
                    )
                elif response.status_code == 403:
                    raise TicketmasterAuthError(
                        "Access forbidden (403). Your API key may lack required permissions."
                    )
                elif response.status_code == 404:
                    raise TicketmasterNotFoundError(
                        f"Resource not found (404): {endpoint}"
                    )
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = TM_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise TicketmasterRateLimitError(
                        "Rate limit exceeded (429). Try again later."
                    )
                else:
                    if attempt < self.max_retries - 1:
                        wait_time = TM_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(
                            f"HTTP {response.status_code}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    raise TicketmasterError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = TM_RETRY_BACKOFF ** (attempt + 1)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise TicketmasterError(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise TicketmasterError("Max retries exceeded")
    
    def _paginate(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle pagination for list endpoints.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_pages: Maximum number of pages to fetch (None for all)
            
        Returns:
            List of all items from all pages
        """
        all_items: List[Dict[str, Any]] = []
        params["size"] = params.get("size", TM_DEFAULT_PAGE_SIZE)
        params["page"] = 0
        page_count = 0
        
        while True:
            response = self._make_request(endpoint, params)
            
            # Extract items from _embedded
            embedded = response.get("_embedded", {})
            items = []
            for key in embedded:
                if isinstance(embedded[key], list):
                    items = embedded[key]
                    break
            
            if not items:
                break
            
            all_items.extend(items)
            page_count += 1
            
            # Check pagination
            page_info = response.get("page", {})
            total_pages = page_info.get("totalPages", 1)
            current_page = page_info.get("number", 0)
            
            logger.debug(
                f"Fetched page {current_page + 1}/{total_pages}, "
                f"items: {len(items)}, total: {len(all_items)}"
            )
            
            if current_page + 1 >= total_pages:
                break
            
            if max_pages and page_count >= max_pages:
                logger.info(f"Reached max_pages limit ({max_pages})")
                break
            
            params["page"] = current_page + 1
        
        logger.info(f"Pagination complete: {len(all_items)} items from {page_count} pages")
        return all_items
    
    def search_events(
        self,
        keyword: Optional[str] = None,
        event_id: Optional[str] = None,
        venue_id: Optional[str] = None,
        city: Optional[str] = None,
        state_code: Optional[str] = None,
        country_code: str = "CA",
        classification_name: str = "Ballet",
        start_date_time: Optional[str] = None,
        end_date_time: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[TicketmasterEvent]:
        """
        Search for events.
        
        Args:
            keyword: Keyword to search for (e.g., show title)
            event_id: Specific event ID to fetch
            venue_id: Filter by venue ID
            city: Filter by city name (e.g., "Calgary")
            state_code: State/province code (e.g., "AB" for Alberta)
            country_code: Country code (default: "CA" for Canada)
            classification_name: Event classification (default: "Ballet")
            start_date_time: Start date filter (ISO 8601 format)
            end_date_time: End date filter (ISO 8601 format)
            max_pages: Maximum pages to fetch
            
        Returns:
            List of TicketmasterEvent objects
        """
        params: Dict[str, Any] = {
            "countryCode": country_code,
        }
        
        if keyword:
            params["keyword"] = keyword
        if event_id:
            params["id"] = event_id
        if venue_id:
            params["venueId"] = venue_id
        if city:
            params["city"] = city
        if state_code:
            params["stateCode"] = state_code
        if classification_name:
            params["classificationName"] = classification_name
        if start_date_time:
            params["startDateTime"] = start_date_time
        if end_date_time:
            params["endDateTime"] = end_date_time
        
        logger.info(f"Searching events with params: {params}")
        raw_events = self._paginate("/events", params, max_pages)
        
        events = []
        for raw in raw_events:
            event = self._parse_event(raw)
            events.append(event)
        
        logger.info(f"Found {len(events)} events")
        return events
    
    def get_event(self, event_id: str) -> TicketmasterEvent:
        """
        Get a specific event by ID.
        
        Args:
            event_id: Ticketmaster event ID
            
        Returns:
            TicketmasterEvent object
        """
        logger.info(f"Fetching event: {event_id}")
        response = self._make_request(f"/events/{event_id}")
        return self._parse_event(response)
    
    def get_venue(self, venue_id: str) -> TicketmasterVenue:
        """
        Get a specific venue by ID.
        
        Args:
            venue_id: Ticketmaster venue ID
            
        Returns:
            TicketmasterVenue object
        """
        logger.info(f"Fetching venue: {venue_id}")
        response = self._make_request(f"/venues/{venue_id}")
        return self._parse_venue(response)
    
    def search_venues(
        self,
        keyword: Optional[str] = None,
        city: Optional[str] = None,
        state_code: Optional[str] = None,
        country_code: str = "CA",
        max_pages: Optional[int] = None,
    ) -> List[TicketmasterVenue]:
        """
        Search for venues.
        
        Args:
            keyword: Keyword to search for
            city: Filter by city name
            state_code: State/province code
            country_code: Country code (default: "CA")
            max_pages: Maximum pages to fetch
            
        Returns:
            List of TicketmasterVenue objects
        """
        params: Dict[str, Any] = {
            "countryCode": country_code,
        }
        
        if keyword:
            params["keyword"] = keyword
        if city:
            params["city"] = city
        if state_code:
            params["stateCode"] = state_code
        
        logger.info(f"Searching venues with params: {params}")
        raw_venues = self._paginate("/venues", params, max_pages)
        
        venues = []
        for raw in raw_venues:
            venue = self._parse_venue(raw)
            venues.append(venue)
        
        logger.info(f"Found {len(venues)} venues")
        return venues
    
    def _parse_event(self, raw: Dict[str, Any]) -> TicketmasterEvent:
        """Parse raw event data into a TicketmasterEvent object."""
        # Extract venue info
        venues = raw.get("_embedded", {}).get("venues", [])
        venue = venues[0] if venues else {}
        
        # Extract dates
        dates = raw.get("dates", {})
        start = dates.get("start", {})
        end = dates.get("end", {})
        
        # Extract price ranges
        price_ranges = raw.get("priceRanges", [])
        
        # Extract city info
        city_obj = venue.get("city", {})
        state_obj = venue.get("state", {})
        country_obj = venue.get("country", {})
        
        return TicketmasterEvent(
            event_id=raw.get("id", ""),
            name=raw.get("name", ""),
            url=raw.get("url"),
            start_date=start.get("localDate"),
            start_time=start.get("localTime"),
            end_date=end.get("localDate"),
            venue_name=venue.get("name"),
            venue_id=venue.get("id"),
            venue_capacity=venue.get("upcomingEvents", {}).get("_total"),
            city=city_obj.get("name"),
            state_province=state_obj.get("stateCode") or state_obj.get("name"),
            country=country_obj.get("countryCode") or country_obj.get("name"),
            price_ranges=price_ranges,
            classifications=raw.get("classifications", []),
            raw_data=raw,
        )
    
    def _parse_venue(self, raw: Dict[str, Any]) -> TicketmasterVenue:
        """Parse raw venue data into a TicketmasterVenue object."""
        city_obj = raw.get("city", {})
        state_obj = raw.get("state", {})
        country_obj = raw.get("country", {})
        location = raw.get("location", {})
        
        return TicketmasterVenue(
            venue_id=raw.get("id", ""),
            name=raw.get("name", ""),
            city=city_obj.get("name"),
            state_province=state_obj.get("stateCode") or state_obj.get("name"),
            country=country_obj.get("countryCode") or country_obj.get("name"),
            capacity=raw.get("upcomingEvents", {}).get("_total"),
            address=raw.get("address", {}).get("line1"),
            postal_code=raw.get("postalCode"),
            latitude=float(location.get("latitude")) if location.get("latitude") else None,
            longitude=float(location.get("longitude")) if location.get("longitude") else None,
            raw_data=raw,
        )
    
    def get_request_count(self) -> int:
        """Return the total number of API requests made."""
        return self._request_count
