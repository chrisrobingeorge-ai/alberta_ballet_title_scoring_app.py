"""
Archtics API Client

This module provides a read-only client for the Archtics Reporting API to fetch
sales summaries, venue manifests, and transaction data for performances.

API Access:
    Contact your Archtics account representative for API credentials and documentation.
    API access may require a separate contract or service agreement.

Environment Variables:
    ARCHTICS_API_KEY: Archtics API key (required)
    ARCHTICS_CLIENT_ID: Alternative authentication method (optional)
    ARCHTICS_BASE_URL: Organization-specific API endpoint (required)

Rate Limits:
    Varies by contract. Default implementation includes conservative rate limiting.
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
ARCHTICS_DEFAULT_RATE_LIMIT_DELAY = 0.5  # seconds between requests
ARCHTICS_DEFAULT_PAGE_SIZE = 100
ARCHTICS_MAX_RETRIES = 3
ARCHTICS_RETRY_BACKOFF = 2.0


class ArchticsError(Exception):
    """Base exception for Archtics API errors."""
    pass


class ArchticsAuthError(ArchticsError):
    """Raised when authentication fails (401/403)."""
    pass


class ArchticsRateLimitError(ArchticsError):
    """Raised when rate limit is exceeded (429)."""
    pass


class ArchticsNotFoundError(ArchticsError):
    """Raised when a resource is not found (404)."""
    pass


@dataclass
class ArchticsPerformance:
    """Represents an Archtics performance/event."""
    performance_id: str
    event_id: str
    event_name: str
    performance_date: Optional[str] = None
    performance_time: Optional[str] = None
    venue_id: Optional[str] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[int] = None
    city: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchticsSalesSummary:
    """Represents sales summary data for a performance."""
    performance_id: str
    event_name: Optional[str] = None
    single_tickets_sold: int = 0
    subscription_tickets_sold: int = 0
    comp_tickets: int = 0
    refunds: int = 0
    cancellations: int = 0
    total_tickets_sold: int = 0
    gross_revenue: float = 0.0
    net_revenue: float = 0.0
    channel_mix: Dict[str, int] = field(default_factory=dict)
    price_tier_breakdown: Dict[str, int] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchticsVenue:
    """Represents an Archtics venue manifest."""
    venue_id: str
    name: str
    capacity: int = 0
    city: Optional[str] = None
    address: Optional[str] = None
    price_tiers: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


class ArchticsClient:
    """
    Client for Archtics Reporting API.
    
    This client provides read-only access to sales summaries, venue manifests,
    and transaction data. It handles pagination, rate limiting, and HTTP error retries.
    
    Note: Archtics API endpoints and authentication methods vary by organization.
    This implementation uses a generic approach that may need adjustment for your
    specific Archtics setup.
    
    Example:
        >>> client = ArchticsClient(
        ...     api_key="your_key",
        ...     base_url="https://your-org.archtics.com/api"
        ... )
        >>> summary = client.get_sales_summary(event_id="12345")
        >>> print(f"Single tickets: {summary.single_tickets_sold}")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit_delay: float = ARCHTICS_DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = ARCHTICS_MAX_RETRIES,
    ):
        """
        Initialize the Archtics client.
        
        Args:
            api_key: Archtics API key. If not provided, reads from ARCHTICS_API_KEY env var.
            client_id: Alternative auth method. Reads from ARCHTICS_CLIENT_ID if not provided.
            base_url: Organization API endpoint. Reads from ARCHTICS_BASE_URL if not provided.
            rate_limit_delay: Seconds to wait between requests.
            max_retries: Maximum number of retry attempts for failed requests.
            
        Raises:
            ArchticsAuthError: If no credentials or base URL are provided.
        """
        self.api_key = api_key or os.getenv("ARCHTICS_API_KEY")
        self.client_id = client_id or os.getenv("ARCHTICS_CLIENT_ID")
        self.base_url = base_url or os.getenv("ARCHTICS_BASE_URL")
        
        if not self.api_key and not self.client_id:
            raise ArchticsAuthError(
                "Archtics credentials not provided. "
                "Set ARCHTICS_API_KEY or ARCHTICS_CLIENT_ID environment variable."
            )
        
        if not self.base_url:
            raise ArchticsAuthError(
                "Archtics base URL not provided. "
                "Set ARCHTICS_BASE_URL environment variable."
            )
        
        self.base_url = self.base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time: float = 0
        self._request_count: int = 0
        
        logger.info(f"Initialized ArchticsClient with base_url={self.base_url}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            # Common API key header patterns
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        
        if self.client_id:
            headers["X-Client-Id"] = self.client_id
        
        return headers
    
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
        Make an HTTP request to the Archtics API.
        
        Args:
            endpoint: API endpoint (e.g., "/events")
            params: Query parameters
            method: HTTP method (default: GET)
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            ArchticsAuthError: On 401/403 responses
            ArchticsRateLimitError: On 429 responses
            ArchticsNotFoundError: On 404 responses
            ArchticsError: On other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = self._get_auth_headers()
        
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            self._last_request_time = time.time()
            self._request_count += 1
            
            try:
                logger.debug(f"Request #{self._request_count}: {method} {url}")
                response = requests.request(
                    method, url, params=params, headers=headers, timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise ArchticsAuthError(
                        "Authentication failed (401). Check your ARCHTICS_API_KEY."
                    )
                elif response.status_code == 403:
                    raise ArchticsAuthError(
                        "Access forbidden (403). Your credentials may lack required permissions."
                    )
                elif response.status_code == 404:
                    raise ArchticsNotFoundError(
                        f"Resource not found (404): {endpoint}"
                    )
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = ARCHTICS_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise ArchticsRateLimitError(
                        "Rate limit exceeded (429). Try again later."
                    )
                else:
                    if attempt < self.max_retries - 1:
                        wait_time = ARCHTICS_RETRY_BACKOFF ** (attempt + 1)
                        logger.warning(
                            f"HTTP {response.status_code}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    raise ArchticsError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = ARCHTICS_RETRY_BACKOFF ** (attempt + 1)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise ArchticsError(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise ArchticsError("Max retries exceeded")
    
    def _paginate(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle pagination for list endpoints.
        
        Note: Pagination behavior varies by Archtics implementation.
        This uses offset/limit pagination as a common pattern.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_pages: Maximum number of pages to fetch (None for all)
            
        Returns:
            List of all items from all pages
        """
        all_items: List[Dict[str, Any]] = []
        page_size = params.get("limit", ARCHTICS_DEFAULT_PAGE_SIZE)
        params["limit"] = page_size
        params["offset"] = params.get("offset", 0)
        page_count = 0
        
        while True:
            try:
                response = self._make_request(endpoint, params)
            except ArchticsNotFoundError:
                # Some endpoints return 404 when no more data
                break
            
            # Extract items - common patterns
            items = []
            if isinstance(response, list):
                items = response
            elif "data" in response:
                items = response.get("data", [])
            elif "items" in response:
                items = response.get("items", [])
            elif "results" in response:
                items = response.get("results", [])
            
            if not items:
                break
            
            all_items.extend(items)
            page_count += 1
            
            # Check if there are more pages
            total = response.get("total", response.get("totalCount", len(items)))
            fetched = params["offset"] + len(items)
            
            logger.debug(
                f"Fetched page {page_count}, items: {len(items)}, total fetched: {fetched}"
            )
            
            if fetched >= total:
                break
            
            if max_pages and page_count >= max_pages:
                logger.info(f"Reached max_pages limit ({max_pages})")
                break
            
            params["offset"] = fetched
        
        logger.info(f"Pagination complete: {len(all_items)} items from {page_count} pages")
        return all_items
    
    def get_events(
        self,
        season: Optional[str] = None,
        event_name: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[ArchticsPerformance]:
        """
        Get events/performances from Archtics.
        
        Args:
            season: Season filter (e.g., "2024-25")
            event_name: Filter by event name (partial match)
            max_pages: Maximum pages to fetch
            
        Returns:
            List of ArchticsPerformance objects
        """
        params: Dict[str, Any] = {}
        
        if season:
            params["season"] = season
        if event_name:
            params["eventName"] = event_name
        
        logger.info(f"Fetching events with params: {params}")
        
        try:
            raw_events = self._paginate("/events", params, max_pages)
        except ArchticsNotFoundError:
            logger.warning("Events endpoint not found. Returning empty list.")
            return []
        
        events = []
        for raw in raw_events:
            event = self._parse_performance(raw)
            events.append(event)
        
        logger.info(f"Found {len(events)} events")
        return events
    
    def get_performances(
        self,
        event_id: str,
        max_pages: Optional[int] = None,
    ) -> List[ArchticsPerformance]:
        """
        Get performances for a specific event.
        
        Args:
            event_id: Event ID
            max_pages: Maximum pages to fetch
            
        Returns:
            List of ArchticsPerformance objects
        """
        logger.info(f"Fetching performances for event: {event_id}")
        
        try:
            raw_performances = self._paginate(
                f"/events/{event_id}/performances", {}, max_pages
            )
        except ArchticsNotFoundError:
            logger.warning(f"Performances not found for event {event_id}")
            return []
        
        performances = []
        for raw in raw_performances:
            perf = self._parse_performance(raw)
            performances.append(perf)
        
        logger.info(f"Found {len(performances)} performances")
        return performances
    
    def get_sales_summary(
        self,
        event_id: Optional[str] = None,
        performance_id: Optional[str] = None,
    ) -> ArchticsSalesSummary:
        """
        Get sales summary for an event or performance.
        
        Args:
            event_id: Event ID (for aggregated summary)
            performance_id: Performance ID (for specific performance)
            
        Returns:
            ArchticsSalesSummary object
            
        Raises:
            ArchticsError: If neither event_id nor performance_id is provided
        """
        if not event_id and not performance_id:
            raise ArchticsError("Either event_id or performance_id must be provided")
        
        if performance_id:
            endpoint = f"/performances/{performance_id}/sales"
        else:
            endpoint = f"/events/{event_id}/sales"
        
        logger.info(f"Fetching sales summary: {endpoint}")
        
        try:
            response = self._make_request(endpoint)
        except ArchticsNotFoundError:
            logger.warning(f"Sales data not found. Returning empty summary.")
            return ArchticsSalesSummary(
                performance_id=performance_id or event_id or ""
            )
        
        return self._parse_sales_summary(response, performance_id or event_id or "")
    
    def get_sales_by_channel(
        self,
        event_id: str,
    ) -> Dict[str, int]:
        """
        Get sales breakdown by channel (web, phone, walkup, etc.).
        
        Args:
            event_id: Event ID
            
        Returns:
            Dictionary mapping channel names to ticket counts
        """
        logger.info(f"Fetching channel mix for event: {event_id}")
        
        try:
            response = self._make_request(f"/events/{event_id}/sales/channels")
        except ArchticsNotFoundError:
            logger.warning(f"Channel data not found for event {event_id}")
            return {}
        
        channels: Dict[str, int] = {}
        if isinstance(response, list):
            for item in response:
                channel = item.get("channel", item.get("name", "unknown"))
                count = item.get("tickets", item.get("count", 0))
                channels[channel] = count
        elif isinstance(response, dict):
            for key, value in response.items():
                if isinstance(value, (int, float)):
                    channels[key] = int(value)
        
        return channels
    
    def get_venue(self, venue_id: str) -> ArchticsVenue:
        """
        Get venue manifest including capacity and price tiers.
        
        Args:
            venue_id: Venue ID
            
        Returns:
            ArchticsVenue object
        """
        logger.info(f"Fetching venue: {venue_id}")
        response = self._make_request(f"/venues/{venue_id}")
        return self._parse_venue(response)
    
    def get_refunds_cancellations(
        self,
        event_id: str,
    ) -> Dict[str, int]:
        """
        Get refunds and cancellations for an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Dictionary with 'refunds' and 'cancellations' counts
        """
        logger.info(f"Fetching refunds/cancellations for event: {event_id}")
        
        try:
            response = self._make_request(f"/events/{event_id}/refunds")
        except ArchticsNotFoundError:
            logger.warning(f"Refund data not found for event {event_id}")
            return {"refunds": 0, "cancellations": 0}
        
        refunds = response.get("refunds", response.get("refundCount", 0))
        cancellations = response.get("cancellations", response.get("cancelCount", 0))
        
        return {
            "refunds": int(refunds) if refunds else 0,
            "cancellations": int(cancellations) if cancellations else 0,
        }
    
    def _parse_performance(self, raw: Dict[str, Any]) -> ArchticsPerformance:
        """Parse raw performance data into an ArchticsPerformance object."""
        return ArchticsPerformance(
            performance_id=str(raw.get("performanceId", raw.get("id", ""))),
            event_id=str(raw.get("eventId", "")),
            event_name=raw.get("eventName", raw.get("name", "")),
            performance_date=raw.get("date", raw.get("performanceDate")),
            performance_time=raw.get("time", raw.get("performanceTime")),
            venue_id=str(raw.get("venueId", "")) if raw.get("venueId") else None,
            venue_name=raw.get("venueName"),
            venue_capacity=raw.get("capacity", raw.get("venueCapacity")),
            city=raw.get("city"),
            raw_data=raw,
        )
    
    def _parse_sales_summary(
        self, raw: Dict[str, Any], performance_id: str
    ) -> ArchticsSalesSummary:
        """Parse raw sales data into an ArchticsSalesSummary object."""
        # Extract channel mix
        channel_mix: Dict[str, int] = {}
        if "channels" in raw:
            for ch in raw["channels"]:
                channel_mix[ch.get("name", "unknown")] = ch.get("count", 0)
        elif "channelMix" in raw:
            channel_mix = raw["channelMix"]
        
        # Extract price tier breakdown
        price_tiers: Dict[str, int] = {}
        if "priceTiers" in raw:
            for tier in raw["priceTiers"]:
                price_tiers[tier.get("name", "unknown")] = tier.get("count", 0)
        
        return ArchticsSalesSummary(
            performance_id=performance_id,
            event_name=raw.get("eventName"),
            single_tickets_sold=raw.get("singleTickets", raw.get("singles", 0)),
            subscription_tickets_sold=raw.get(
                "subscriptionTickets", raw.get("subscriptions", 0)
            ),
            comp_tickets=raw.get("comps", raw.get("compTickets", 0)),
            refunds=raw.get("refunds", 0),
            cancellations=raw.get("cancellations", 0),
            total_tickets_sold=raw.get("totalTickets", raw.get("total", 0)),
            gross_revenue=float(raw.get("grossRevenue", 0)),
            net_revenue=float(raw.get("netRevenue", 0)),
            channel_mix=channel_mix,
            price_tier_breakdown=price_tiers,
            raw_data=raw,
        )
    
    def _parse_venue(self, raw: Dict[str, Any]) -> ArchticsVenue:
        """Parse raw venue data into an ArchticsVenue object."""
        # Extract price tiers
        price_tiers: List[Dict[str, Any]] = []
        if "priceTiers" in raw:
            price_tiers = raw["priceTiers"]
        elif "pricing" in raw:
            price_tiers = raw["pricing"]
        
        # Extract sections
        sections: List[Dict[str, Any]] = []
        if "sections" in raw:
            sections = raw["sections"]
        elif "seatingSections" in raw:
            sections = raw["seatingSections"]
        
        return ArchticsVenue(
            venue_id=str(raw.get("venueId", raw.get("id", ""))),
            name=raw.get("name", ""),
            capacity=raw.get("capacity", 0),
            city=raw.get("city"),
            address=raw.get("address"),
            price_tiers=price_tiers,
            sections=sections,
            raw_data=raw,
        )
    
    def get_request_count(self) -> int:
        """Return the total number of API requests made."""
        return self._request_count
