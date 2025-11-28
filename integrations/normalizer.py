"""
Show Data Normalizer

This module normalizes data from Ticketmaster and Archtics APIs to a
consistent schema for analysis and CSV export.

The normalized schema includes:
- Show identification fields
- Venue and capacity information
- Performance counts
- Ticket sales by city and type
- Sales metrics (load factor, late sales, etc.)
- Pricing and channel distribution
- Date and timing information

Schema Target Columns:
    show_title, show_title_id, production_season, city,
    venue_name, venue_capacity, performance_count_city, performance_count_total,
    single_tickets_calgary, single_tickets_edmonton,
    subscription_tickets_calgary, subscription_tickets_edmonton,
    total_single_tickets, total_subscription_tickets, total_tickets_all,
    avg_tickets_per_performance, load_factor,
    weeks_to_80pct_sold, late_sales_share,
    channel_mix_distribution, group_sales_share, comp_ticket_share,
    refund_cancellation_rate, pricing_tier_structure, average_base_ticket_price,
    opening_date, closing_date, weekday_vs_weekend_mix
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Target schema column order
NORMALIZED_COLUMNS = [
    "show_title",
    "show_title_id",
    "production_season",
    "city",
    "venue_name",
    "venue_capacity",
    "performance_count_city",
    "performance_count_total",
    "single_tickets_calgary",
    "single_tickets_edmonton",
    "subscription_tickets_calgary",
    "subscription_tickets_edmonton",
    "total_single_tickets",
    "total_subscription_tickets",
    "total_tickets_all",
    "avg_tickets_per_performance",
    "load_factor",
    "weeks_to_80pct_sold",
    "late_sales_share",
    "channel_mix_distribution",
    "group_sales_share",
    "comp_ticket_share",
    "refund_cancellation_rate",
    "pricing_tier_structure",
    "average_base_ticket_price",
    "opening_date",
    "closing_date",
    "weekday_vs_weekend_mix",
]


@dataclass
class NormalizedShowData:
    """
    Normalized show data aligned to the target schema.
    
    All fields are normalized to a consistent format for export.
    Fields not available from APIs are set to None with documentation.
    """
    # Identification
    show_title: str
    show_title_id: str
    production_season: Optional[str] = None
    city: Optional[str] = None
    
    # Venue
    venue_name: Optional[str] = None
    venue_capacity: Optional[int] = None
    
    # Performance counts
    performance_count_city: int = 0
    performance_count_total: int = 0
    
    # Single tickets by city
    single_tickets_calgary: int = 0
    single_tickets_edmonton: int = 0
    
    # Subscription tickets by city
    subscription_tickets_calgary: int = 0
    subscription_tickets_edmonton: int = 0
    
    # Totals
    total_single_tickets: int = 0
    total_subscription_tickets: int = 0
    total_tickets_all: int = 0
    
    # Metrics
    avg_tickets_per_performance: Optional[float] = None
    load_factor: Optional[float] = None
    weeks_to_80pct_sold: Optional[float] = None
    late_sales_share: Optional[float] = None
    
    # Channel and group sales
    channel_mix_distribution: Optional[str] = None
    group_sales_share: Optional[float] = None
    comp_ticket_share: Optional[float] = None
    refund_cancellation_rate: Optional[float] = None
    
    # Pricing
    pricing_tier_structure: Optional[str] = None
    average_base_ticket_price: Optional[float] = None
    
    # Dates
    opening_date: Optional[str] = None
    closing_date: Optional[str] = None
    weekday_vs_weekend_mix: Optional[str] = None
    
    # Metadata
    source_tm: bool = False  # Data includes Ticketmaster
    source_archtics: bool = False  # Data includes Archtics
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "show_title": self.show_title,
            "show_title_id": self.show_title_id,
            "production_season": self.production_season,
            "city": self.city,
            "venue_name": self.venue_name,
            "venue_capacity": self.venue_capacity,
            "performance_count_city": self.performance_count_city,
            "performance_count_total": self.performance_count_total,
            "single_tickets_calgary": self.single_tickets_calgary,
            "single_tickets_edmonton": self.single_tickets_edmonton,
            "subscription_tickets_calgary": self.subscription_tickets_calgary,
            "subscription_tickets_edmonton": self.subscription_tickets_edmonton,
            "total_single_tickets": self.total_single_tickets,
            "total_subscription_tickets": self.total_subscription_tickets,
            "total_tickets_all": self.total_tickets_all,
            "avg_tickets_per_performance": self.avg_tickets_per_performance,
            "load_factor": self.load_factor,
            "weeks_to_80pct_sold": self.weeks_to_80pct_sold,
            "late_sales_share": self.late_sales_share,
            "channel_mix_distribution": self.channel_mix_distribution,
            "group_sales_share": self.group_sales_share,
            "comp_ticket_share": self.comp_ticket_share,
            "refund_cancellation_rate": self.refund_cancellation_rate,
            "pricing_tier_structure": self.pricing_tier_structure,
            "average_base_ticket_price": self.average_base_ticket_price,
            "opening_date": self.opening_date,
            "closing_date": self.closing_date,
            "weekday_vs_weekend_mix": self.weekday_vs_weekend_mix,
        }


class ShowDataNormalizer:
    """
    Normalizes data from Ticketmaster and Archtics to the target schema.
    
    This class handles:
    - Combining data from multiple sources
    - City detection (Calgary vs Edmonton)
    - Metric calculations (load factor, averages, etc.)
    - Serialization of complex fields
    
    Example:
        >>> normalizer = ShowDataNormalizer()
        >>> tm_events = ticketmaster_client.search_events(keyword="Nutcracker")
        >>> archtics_sales = archtics_client.get_sales_summary(event_id="123")
        >>> normalized = normalizer.normalize(
        ...     show_title="The Nutcracker",
        ...     show_id="nutcracker-2024",
        ...     tm_events=tm_events,
        ...     archtics_sales=archtics_sales,
        ... )
    """
    
    # City detection patterns
    CALGARY_PATTERNS = ["calgary", "yyc", "jubilee auditorium calgary"]
    EDMONTON_PATTERNS = ["edmonton", "yeg", "jubilee auditorium edmonton"]
    
    def __init__(self):
        """Initialize the normalizer."""
        logger.info("Initialized ShowDataNormalizer")
    
    def normalize(
        self,
        show_title: str,
        show_id: str,
        tm_events: Optional[List[Any]] = None,
        archtics_performances: Optional[List[Any]] = None,
        archtics_sales: Optional[Any] = None,
        archtics_venue: Optional[Any] = None,
        season: Optional[str] = None,
    ) -> NormalizedShowData:
        """
        Normalize show data from Ticketmaster and Archtics sources.
        
        Args:
            show_title: Show title
            show_id: Show identifier
            tm_events: List of TicketmasterEvent objects
            archtics_performances: List of ArchticsPerformance objects
            archtics_sales: ArchticsSalesSummary object
            archtics_venue: ArchticsVenue object
            season: Production season (e.g., "2024-25")
            
        Returns:
            NormalizedShowData object with all available data normalized
        """
        logger.info(f"Normalizing data for show: {show_title} ({show_id})")
        
        result = NormalizedShowData(
            show_title=show_title,
            show_title_id=show_id,
            production_season=season,
        )
        
        # Track data sources
        result.source_tm = tm_events is not None and len(tm_events) > 0
        result.source_archtics = archtics_sales is not None or archtics_performances is not None
        
        # Process Ticketmaster events
        if tm_events:
            self._process_tm_events(result, tm_events)
        
        # Process Archtics performances
        if archtics_performances:
            self._process_archtics_performances(result, archtics_performances)
        
        # Process Archtics sales
        if archtics_sales:
            self._process_archtics_sales(result, archtics_sales)
        
        # Process Archtics venue
        if archtics_venue:
            self._process_archtics_venue(result, archtics_venue)
        
        # Calculate derived metrics
        self._calculate_metrics(result)
        
        # Add warnings for missing data
        self._add_missing_data_warnings(result)
        
        logger.info(
            f"Normalized {show_title}: "
            f"{result.total_tickets_all} total tickets, "
            f"{result.performance_count_total} performances"
        )
        
        return result
    
    def _detect_city(self, text: str) -> Optional[str]:
        """
        Detect city from text (venue name, city field, etc.).
        
        Args:
            text: Text to search for city patterns
            
        Returns:
            "Calgary", "Edmonton", or None if not detected
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for pattern in self.CALGARY_PATTERNS:
            if pattern in text_lower:
                return "Calgary"
        
        for pattern in self.EDMONTON_PATTERNS:
            if pattern in text_lower:
                return "Edmonton"
        
        return None
    
    def _process_tm_events(
        self, result: NormalizedShowData, events: List[Any]
    ) -> None:
        """Process Ticketmaster events and update result."""
        calgary_count = 0
        edmonton_count = 0
        dates: List[str] = []
        venues: Dict[str, int] = {}
        
        for event in events:
            # Get city from event
            city = self._detect_city(event.city or "") or self._detect_city(
                event.venue_name or ""
            )
            
            if city == "Calgary":
                calgary_count += 1
            elif city == "Edmonton":
                edmonton_count += 1
            
            # Track dates
            if event.start_date:
                dates.append(event.start_date)
            
            # Track venues
            if event.venue_name:
                venues[event.venue_name] = venues.get(event.venue_name, 0) + 1
                
                # Update venue info if not set
                if not result.venue_name:
                    result.venue_name = event.venue_name
                if not result.venue_capacity and event.venue_capacity:
                    result.venue_capacity = event.venue_capacity
        
        # Update counts
        result.performance_count_total = len(events)
        
        # Determine primary city if all in one city
        if calgary_count > 0 and edmonton_count == 0:
            result.city = "Calgary"
            result.performance_count_city = calgary_count
        elif edmonton_count > 0 and calgary_count == 0:
            result.city = "Edmonton"
            result.performance_count_city = edmonton_count
        elif calgary_count > edmonton_count:
            result.city = "Calgary"
            result.performance_count_city = calgary_count
        elif edmonton_count > 0:
            result.city = "Edmonton"
            result.performance_count_city = edmonton_count
        
        # Set dates
        if dates:
            dates.sort()
            result.opening_date = dates[0]
            result.closing_date = dates[-1]
            
            # Calculate weekday/weekend mix
            result.weekday_vs_weekend_mix = self._calculate_weekday_weekend_mix(dates)
        
        # Process price ranges
        price_ranges: List[Dict[str, Any]] = []
        for event in events:
            if hasattr(event, 'price_ranges') and event.price_ranges:
                price_ranges.extend(event.price_ranges)
        
        if price_ranges:
            result.pricing_tier_structure = self._serialize_price_tiers(price_ranges)
            result.average_base_ticket_price = self._calculate_avg_price(price_ranges)
    
    def _process_archtics_performances(
        self, result: NormalizedShowData, performances: List[Any]
    ) -> None:
        """Process Archtics performances and update result."""
        if not performances:
            return
        
        calgary_count = 0
        edmonton_count = 0
        dates: List[str] = []
        
        for perf in performances:
            # Detect city
            city = self._detect_city(perf.city or "") or self._detect_city(
                perf.venue_name or ""
            )
            
            if city == "Calgary":
                calgary_count += 1
            elif city == "Edmonton":
                edmonton_count += 1
            
            # Track dates
            if perf.performance_date:
                dates.append(perf.performance_date)
            
            # Update venue info if not set
            if not result.venue_name and perf.venue_name:
                result.venue_name = perf.venue_name
            if not result.venue_capacity and perf.venue_capacity:
                result.venue_capacity = perf.venue_capacity
        
        # Update counts if not already set from TM
        if result.performance_count_total == 0:
            result.performance_count_total = len(performances)
        
        # Set dates if not already set
        if dates and not result.opening_date:
            dates.sort()
            result.opening_date = dates[0]
            result.closing_date = dates[-1]
            result.weekday_vs_weekend_mix = self._calculate_weekday_weekend_mix(dates)
    
    def _process_archtics_sales(
        self, result: NormalizedShowData, sales: Any
    ) -> None:
        """Process Archtics sales summary and update result."""
        if not sales:
            return
        
        # Update single tickets
        if hasattr(sales, 'single_tickets_sold'):
            total_singles = sales.single_tickets_sold or 0
            
            # Distribute between cities based on performance distribution
            if result.city == "Calgary":
                result.single_tickets_calgary = total_singles
            elif result.city == "Edmonton":
                result.single_tickets_edmonton = total_singles
            else:
                # Split 60/40 Calgary/Edmonton as default
                result.single_tickets_calgary = int(total_singles * 0.6)
                result.single_tickets_edmonton = total_singles - result.single_tickets_calgary
        
        # Update subscription tickets
        if hasattr(sales, 'subscription_tickets_sold'):
            total_subs = sales.subscription_tickets_sold or 0
            
            if result.city == "Calgary":
                result.subscription_tickets_calgary = total_subs
            elif result.city == "Edmonton":
                result.subscription_tickets_edmonton = total_subs
            else:
                result.subscription_tickets_calgary = int(total_subs * 0.6)
                result.subscription_tickets_edmonton = total_subs - result.subscription_tickets_calgary
        
        # Update totals
        result.total_single_tickets = (
            result.single_tickets_calgary + result.single_tickets_edmonton
        )
        result.total_subscription_tickets = (
            result.subscription_tickets_calgary + result.subscription_tickets_edmonton
        )
        result.total_tickets_all = (
            result.total_single_tickets + result.total_subscription_tickets
        )
        
        # Channel mix
        if hasattr(sales, 'channel_mix') and sales.channel_mix:
            result.channel_mix_distribution = self._serialize_dict(sales.channel_mix)
        
        # Comp tickets
        if hasattr(sales, 'comp_tickets') and result.total_tickets_all > 0:
            comp_tickets = sales.comp_tickets or 0
            total_incl_comps = result.total_tickets_all + comp_tickets
            if total_incl_comps > 0:
                result.comp_ticket_share = round(comp_tickets / total_incl_comps, 4)
        
        # Refund/cancellation rate
        if hasattr(sales, 'refunds') and hasattr(sales, 'cancellations'):
            refunds = sales.refunds or 0
            cancellations = sales.cancellations or 0
            total_returns = refunds + cancellations
            if result.total_tickets_all > 0:
                result.refund_cancellation_rate = round(
                    total_returns / result.total_tickets_all, 4
                )
        
        # Price tier breakdown
        if hasattr(sales, 'price_tier_breakdown') and sales.price_tier_breakdown:
            result.pricing_tier_structure = self._serialize_dict(
                sales.price_tier_breakdown
            )
    
    def _process_archtics_venue(
        self, result: NormalizedShowData, venue: Any
    ) -> None:
        """Process Archtics venue data and update result."""
        if not venue:
            return
        
        if not result.venue_name and hasattr(venue, 'name'):
            result.venue_name = venue.name
        
        if not result.venue_capacity and hasattr(venue, 'capacity'):
            result.venue_capacity = venue.capacity
        
        if not result.city and hasattr(venue, 'city'):
            result.city = venue.city
        
        # Price tiers
        if hasattr(venue, 'price_tiers') and venue.price_tiers:
            if not result.pricing_tier_structure:
                result.pricing_tier_structure = self._serialize_price_tiers(
                    venue.price_tiers
                )
    
    def _calculate_metrics(self, result: NormalizedShowData) -> None:
        """Calculate derived metrics."""
        # Average tickets per performance
        if result.performance_count_total > 0:
            result.avg_tickets_per_performance = round(
                result.total_tickets_all / result.performance_count_total, 2
            )
        
        # Load factor (tickets sold / capacity * performances)
        if result.venue_capacity and result.performance_count_total > 0:
            total_capacity = result.venue_capacity * result.performance_count_total
            if total_capacity > 0:
                result.load_factor = round(
                    result.total_tickets_all / total_capacity, 4
                )
    
    def _add_missing_data_warnings(self, result: NormalizedShowData) -> None:
        """Add warnings for fields that couldn't be populated."""
        if result.weeks_to_80pct_sold is None:
            result.warnings.append(
                "weeks_to_80pct_sold: Requires time-series sales data not available via API"
            )
        
        if result.late_sales_share is None:
            result.warnings.append(
                "late_sales_share: Requires time-series sales data not available via API"
            )
        
        if result.group_sales_share is None:
            result.warnings.append(
                "group_sales_share: Requires group sales breakdown not in standard API response"
            )
        
        if not result.source_tm and not result.source_archtics:
            result.warnings.append(
                "No data sources available. Check API credentials and identifiers."
            )
    
    def _calculate_weekday_weekend_mix(self, dates: List[str]) -> str:
        """
        Calculate weekday vs weekend mix from date strings.
        
        Args:
            dates: List of date strings (YYYY-MM-DD format)
            
        Returns:
            Serialized key:value string (e.g., "weekday:5,weekend:2")
        """
        weekday = 0
        weekend = 0
        
        for date_str in dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if dt.weekday() < 5:  # Monday=0, Sunday=6
                    weekday += 1
                else:
                    weekend += 1
            except (ValueError, TypeError):
                continue
        
        return f"weekday:{weekday},weekend:{weekend}"
    
    def _serialize_dict(self, data: Dict[str, Any]) -> str:
        """
        Serialize dictionary to key:value pairs string.
        
        Args:
            data: Dictionary to serialize
            
        Returns:
            Comma-separated key:value pairs (e.g., "web:100,phone:50")
        """
        pairs = []
        for key, value in sorted(data.items()):
            # Sanitize key
            safe_key = re.sub(r'[^\w\-]', '_', str(key))
            pairs.append(f"{safe_key}:{value}")
        return ",".join(pairs)
    
    def _serialize_price_tiers(self, tiers: List[Dict[str, Any]]) -> str:
        """
        Serialize price tier data to a string.
        
        Args:
            tiers: List of price tier dictionaries
            
        Returns:
            Serialized price tier string
        """
        tier_strs = []
        for tier in tiers:
            if isinstance(tier, dict):
                # Get tier name
                name = tier.get("type", tier.get("name", "unknown"))
                # Get price range
                min_price = tier.get("min", tier.get("minPrice", "?"))
                max_price = tier.get("max", tier.get("maxPrice", "?"))
                tier_strs.append(f"{name}:{min_price}-{max_price}")
            else:
                tier_strs.append(str(tier))
        
        return ",".join(tier_strs)
    
    def _calculate_avg_price(self, price_ranges: List[Dict[str, Any]]) -> Optional[float]:
        """
        Calculate average base ticket price from price ranges.
        
        Args:
            price_ranges: List of price range dictionaries
            
        Returns:
            Average price or None if can't be calculated
        """
        prices = []
        for pr in price_ranges:
            if isinstance(pr, dict):
                min_price = pr.get("min")
                max_price = pr.get("max")
                if min_price is not None and max_price is not None:
                    try:
                        avg = (float(min_price) + float(max_price)) / 2
                        prices.append(avg)
                    except (ValueError, TypeError):
                        continue
        
        if prices:
            return round(sum(prices) / len(prices), 2)
        return None


def get_normalized_columns() -> List[str]:
    """
    Get the list of normalized column names in the correct order.
    
    Returns:
        List of column names
    """
    return NORMALIZED_COLUMNS.copy()
