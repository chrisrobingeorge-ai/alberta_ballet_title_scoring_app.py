#!/usr/bin/env python3
"""
Pull Show Data CLI

This script pulls per-show/performance data from Ticketmaster and Archtics
APIs and exports a normalized CSV aligned to the analysis schema.

Usage:
    python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25
    python scripts/pull_show_data.py --show_id "nutcracker-2024" --city Calgary

Environment Variables Required:
    TM_API_KEY: Ticketmaster API key
    ARCHTICS_API_KEY: Archtics API key
    ARCHTICS_BASE_URL: Archtics organization endpoint

Output:
    Creates `data/<show_id>_archtics_ticketmaster.csv` with normalized data.

For setup instructions, see README.md section "Archtics + Ticketmaster Integration".
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations import (
    TicketmasterClient,
    ArchticsClient,
    ShowDataNormalizer,
    export_show_csv,
)
from integrations.ticketmaster import TicketmasterError, TicketmasterAuthError
from integrations.archtics import ArchticsError, ArchticsAuthError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_env_from_dotenv() -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = value
            logger.info("Loaded environment from .env file")
        except Exception as e:
            logger.warning(f"Error loading .env file: {e}")


def check_credentials() -> tuple[bool, bool]:
    """
    Check if required API credentials are available.
    
    Returns:
        Tuple of (tm_available, archtics_available)
    """
    tm_available = bool(os.getenv("TM_API_KEY"))
    archtics_available = bool(
        (os.getenv("ARCHTICS_API_KEY") or os.getenv("ARCHTICS_CLIENT_ID"))
        and os.getenv("ARCHTICS_BASE_URL")
    )
    
    return tm_available, archtics_available


def validate_args(args: argparse.Namespace) -> tuple[str, str]:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Tuple of (show_title, show_id)
        
    Raises:
        ValueError: If required arguments are missing
    """
    if not args.show_title and not args.show_id:
        raise ValueError(
            "Either --show_title or --show_id must be provided.\n"
            "Examples:\n"
            "  python scripts/pull_show_data.py --show_title 'The Nutcracker'\n"
            "  python scripts/pull_show_data.py --show_id 'nutcracker-2024'"
        )
    
    show_title = args.show_title or args.show_id
    show_id = args.show_id or (args.show_title.lower().replace(" ", "_") if args.show_title else "")
    
    return show_title, show_id


def pull_ticketmaster_data(
    client: TicketmasterClient,
    show_title: str,
    city: Optional[str] = None,
    season: Optional[str] = None,
) -> list:
    """
    Pull event data from Ticketmaster.
    
    Args:
        client: Ticketmaster client
        show_title: Show title to search for
        city: Optional city filter
        season: Optional season for date filtering
        
    Returns:
        List of TicketmasterEvent objects
    """
    logger.info(f"Fetching Ticketmaster data for: {show_title}")
    
    # Build date filters from season
    start_date = None
    end_date = None
    if season:
        try:
            # Parse season like "2024-25"
            start_year = int(season.split("-")[0])
            start_date = f"{start_year}-09-01T00:00:00Z"
            end_date = f"{start_year + 1}-08-31T23:59:59Z"
        except (ValueError, IndexError):
            logger.warning(f"Could not parse season '{season}' for date filtering")
    
    events = client.search_events(
        keyword=show_title,
        city=city,
        state_code="AB",  # Alberta
        start_date_time=start_date,
        end_date_time=end_date,
    )
    
    logger.info(f"Found {len(events)} events from Ticketmaster")
    return events


def pull_archtics_data(
    client: ArchticsClient,
    show_title: str,
    season: Optional[str] = None,
) -> tuple:
    """
    Pull sales data from Archtics.
    
    Args:
        client: Archtics client
        show_title: Show title to search for
        season: Optional season filter
        
    Returns:
        Tuple of (performances, sales_summary, venue)
    """
    logger.info(f"Fetching Archtics data for: {show_title}")
    
    # Get events matching the show
    events = client.get_events(
        event_name=show_title,
        season=season,
    )
    
    if not events:
        logger.warning("No matching events found in Archtics")
        return [], None, None
    
    logger.info(f"Found {len(events)} events in Archtics")
    
    # Get the first matching event
    event = events[0]
    
    # Get performances
    performances = client.get_performances(event.event_id)
    
    # Get sales summary
    sales_summary = client.get_sales_summary(event_id=event.event_id)
    
    # Get venue if available
    venue = None
    if event.venue_id:
        try:
            venue = client.get_venue(event.venue_id)
        except ArchticsError as e:
            logger.warning(f"Could not fetch venue: {e}")
    
    return performances, sales_summary, venue


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pull show data from Ticketmaster and Archtics APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25
  python scripts/pull_show_data.py --show_id nutcracker-2024 --city Calgary
  python scripts/pull_show_data.py --show_title "Swan Lake" --output custom_output.csv

Environment Variables:
  TM_API_KEY           Ticketmaster API key
  ARCHTICS_API_KEY     Archtics API key
  ARCHTICS_BASE_URL    Archtics organization endpoint

For full documentation, see README.md section "Archtics + Ticketmaster Integration".
        """,
    )
    
    parser.add_argument(
        "--show_title",
        help="Show title to search for (e.g., 'The Nutcracker'). Required if --show_id is not provided.",
    )
    parser.add_argument(
        "--show_id",
        help="Show identifier for the output file (e.g., 'nutcracker-2024'). Required if --show_title is not provided.",
    )
    parser.add_argument(
        "--season",
        help="Production season filter (e.g., '2024-25')",
    )
    parser.add_argument(
        "--city",
        choices=["Calgary", "Edmonton"],
        help="Filter by city",
    )
    parser.add_argument(
        "--output",
        help="Custom output file path (default: data/<show_id>_archtics_ticketmaster.csv)",
    )
    parser.add_argument(
        "--tm-only",
        action="store_true",
        help="Only fetch from Ticketmaster (skip Archtics)",
    )
    parser.add_argument(
        "--archtics-only",
        action="store_true",
        help="Only fetch from Archtics (skip Ticketmaster)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls",
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load .env file
    setup_env_from_dotenv()
    
    # Validate arguments
    try:
        show_title, show_id = validate_args(args)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    logger.info("=" * 60)
    logger.info("Pull Show Data")
    logger.info("=" * 60)
    logger.info(f"Show Title: {show_title}")
    logger.info(f"Show ID: {show_id}")
    if args.season:
        logger.info(f"Season: {args.season}")
    if args.city:
        logger.info(f"City: {args.city}")
    
    # Check credentials
    tm_available, archtics_available = check_credentials()
    
    if not tm_available and not args.archtics_only:
        logger.warning(
            "Ticketmaster credentials not found. Set TM_API_KEY environment variable."
        )
    
    if not archtics_available and not args.tm_only:
        logger.warning(
            "Archtics credentials not found. Set ARCHTICS_API_KEY and ARCHTICS_BASE_URL."
        )
    
    if not tm_available and not archtics_available:
        logger.error(
            "No API credentials available. Please set environment variables.\n"
            "See .env.example for required variables."
        )
        return 1
    
    # Dry run mode
    if args.dry_run:
        logger.info("\n[DRY RUN] Would fetch data from:")
        if tm_available and not args.archtics_only:
            logger.info("  - Ticketmaster Discovery API")
        if archtics_available and not args.tm_only:
            logger.info("  - Archtics Reporting API")
        
        output_path = args.output or f"data/{show_id}_archtics_ticketmaster.csv"
        logger.info(f"\n[DRY RUN] Would save to: {output_path}")
        return 0
    
    # Initialize clients and fetch data
    tm_events = []
    archtics_performances = []
    archtics_sales = None
    archtics_venue = None
    
    # Ticketmaster
    if tm_available and not args.archtics_only:
        try:
            tm_client = TicketmasterClient()
            tm_events = pull_ticketmaster_data(
                tm_client,
                show_title,
                city=args.city,
                season=args.season,
            )
            logger.info(f"Ticketmaster requests: {tm_client.get_request_count()}")
        except TicketmasterAuthError as e:
            logger.error(f"Ticketmaster authentication failed: {e}")
            logger.info("Check your TM_API_KEY environment variable.")
        except TicketmasterError as e:
            logger.error(f"Ticketmaster error: {e}")
    
    # Archtics
    if archtics_available and not args.tm_only:
        try:
            archtics_client = ArchticsClient()
            archtics_performances, archtics_sales, archtics_venue = pull_archtics_data(
                archtics_client,
                show_title,
                season=args.season,
            )
            logger.info(f"Archtics requests: {archtics_client.get_request_count()}")
        except ArchticsAuthError as e:
            logger.error(f"Archtics authentication failed: {e}")
            logger.info("Check ARCHTICS_API_KEY and ARCHTICS_BASE_URL environment variables.")
        except ArchticsError as e:
            logger.error(f"Archtics error: {e}")
    
    # Check if we got any data
    if not tm_events and not archtics_sales and not archtics_performances:
        logger.error(
            "No data retrieved from APIs.\n"
            "Please check:\n"
            "  - Show title/ID spelling\n"
            "  - Season format (e.g., '2024-25')\n"
            "  - City name (Calgary or Edmonton)\n"
            "  - API credentials"
        )
        return 1
    
    # Normalize data
    logger.info("\nNormalizing data...")
    normalizer = ShowDataNormalizer()
    normalized = normalizer.normalize(
        show_title=show_title,
        show_id=show_id,
        tm_events=tm_events,
        archtics_performances=archtics_performances,
        archtics_sales=archtics_sales,
        archtics_venue=archtics_venue,
        season=args.season,
    )
    
    # Export CSV
    logger.info("\nExporting CSV...")
    output_path = export_show_csv(
        normalized,
        output_path=args.output,
        show_id=show_id,
    )
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total tickets: {normalized.total_tickets_all:,}")
    logger.info(f"Performances: {normalized.performance_count_total}")
    if normalized.load_factor:
        logger.info(f"Load factor: {normalized.load_factor:.1%}")
    
    if normalized.warnings:
        logger.info("\nNotes:")
        for warning in normalized.warnings[:3]:
            logger.info(f"  - {warning}")
    
    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
