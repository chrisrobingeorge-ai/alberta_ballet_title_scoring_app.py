#!/usr/bin/env python3
"""
Pull Show Data CLI

This script pulls per-show/performance data from Ticketmaster and Archtics
APIs and exports a normalized CSV aligned to the analysis schema.

Usage:
    python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25
    python scripts/pull_show_data.py --show_id "nutcracker-2024" --city Calgary

Batch Mode:
    python scripts/pull_show_data.py --from_csv data/productions/history_city_sales.csv --season 2024-25

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
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    Validate command line arguments for single show mode.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Tuple of (show_title, show_id)
        
    Raises:
        ValueError: If required arguments are missing
    """
    # In batch mode, --from_csv is used instead of --show_title/--show_id
    if hasattr(args, 'from_csv') and args.from_csv:
        raise ValueError("validate_args should not be called in batch mode")
    
    if not args.show_title and not args.show_id:
        raise ValueError(
            "Either --show_title, --show_id, or --from_csv must be provided.\n"
            "Examples:\n"
            "  python scripts/pull_show_data.py --show_title 'The Nutcracker'\n"
            "  python scripts/pull_show_data.py --show_id 'nutcracker-2024'\n"
            "  python scripts/pull_show_data.py --from_csv data/productions/history_city_sales.csv"
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


@dataclass
class BatchResult:
    """Result of processing a single show in batch mode."""
    show_title: str
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    total_tickets: int = 0
    performances: int = 0


@dataclass
class BatchSummary:
    """Summary of a batch processing run."""
    total_shows: int = 0
    successful: int = 0
    failed: int = 0
    skipped_duplicates: int = 0
    results: List[BatchResult] = field(default_factory=list)
    
    def add_result(self, result: BatchResult) -> None:
        """Add a result to the summary."""
        self.results.append(result)
        if result.success:
            self.successful += 1
        else:
            self.failed += 1


def read_show_titles_from_csv(csv_path: str) -> List[str]:
    """
    Read unique show titles from a CSV file.
    
    Args:
        csv_path: Path to the CSV file (expects 'show_title' or 'Show Title' column)
        
    Returns:
        List of unique show titles
        
    Raises:
        ValueError: If the CSV doesn't have a recognizable show title column
        FileNotFoundError: If the CSV file doesn't exist
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    titles = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Find the show title column (case-insensitive)
        fieldnames = reader.fieldnames or []
        title_column = None
        for col in fieldnames:
            if col.lower().replace("_", " ") in ["show title", "show_title", "title"]:
                title_column = col
                break
        
        if not title_column:
            raise ValueError(
                f"Could not find show title column in CSV. "
                f"Expected 'show_title', 'Show Title', or 'title'. "
                f"Found columns: {fieldnames}"
            )
        
        for row in reader:
            title = row.get(title_column, "").strip()
            if title:
                titles.append(title)
    
    logger.info(f"Read {len(titles)} show titles from {csv_path}")
    return titles


def deduplicate_titles(titles: List[str]) -> tuple[List[str], int]:
    """
    Deduplicate show titles while preserving order.
    
    Args:
        titles: List of show titles (may contain duplicates)
        
    Returns:
        Tuple of (unique_titles, duplicate_count)
    """
    seen = set()
    unique = []
    duplicates = 0
    
    for title in titles:
        # Normalize for comparison (case-insensitive)
        key = title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(title)
        else:
            duplicates += 1
    
    logger.info(f"Deduplicated {len(titles)} titles to {len(unique)} unique (removed {duplicates} duplicates)")
    return unique, duplicates


def generate_show_id(show_title: str, season: Optional[str] = None) -> str:
    """
    Generate a show ID from the title and optional season.
    
    Args:
        show_title: Show title
        season: Optional season (e.g., "2024-25")
        
    Returns:
        Sanitized show ID string
    """
    base_id = show_title.lower().replace(" ", "_")
    # Remove special characters
    base_id = "".join(c for c in base_id if c.isalnum() or c == "_")
    # Collapse multiple underscores
    while "__" in base_id:
        base_id = base_id.replace("__", "_")
    base_id = base_id.strip("_")
    
    if season:
        # Add season to make it unique
        season_suffix = season.replace("-", "_")
        base_id = f"{base_id}_{season_suffix}"
    
    return base_id


def process_single_show(
    show_title: str,
    season: Optional[str],
    city: Optional[str],
    tm_available: bool,
    archtics_available: bool,
    tm_only: bool,
    archtics_only: bool,
) -> BatchResult:
    """
    Process a single show for batch mode.
    
    Args:
        show_title: Show title to process
        season: Optional season filter
        city: Optional city filter
        tm_available: Whether Ticketmaster credentials are available
        archtics_available: Whether Archtics credentials are available
        tm_only: Only fetch from Ticketmaster
        archtics_only: Only fetch from Archtics
        
    Returns:
        BatchResult with success/failure info
    """
    show_id = generate_show_id(show_title, season)
    result = BatchResult(show_title=show_title, success=False)
    
    logger.info(f"\nProcessing: {show_title}")
    logger.info("-" * 40)
    
    try:
        tm_events = []
        archtics_performances = []
        archtics_sales = None
        archtics_venue = None
        
        # Ticketmaster
        if tm_available and not archtics_only:
            try:
                tm_client = TicketmasterClient()
                tm_events = pull_ticketmaster_data(
                    tm_client,
                    show_title,
                    city=city,
                    season=season,
                )
            except (TicketmasterAuthError, TicketmasterError) as e:
                logger.warning(f"Ticketmaster error for '{show_title}': {e}")
        
        # Archtics
        if archtics_available and not tm_only:
            try:
                archtics_client = ArchticsClient()
                archtics_performances, archtics_sales, archtics_venue = pull_archtics_data(
                    archtics_client,
                    show_title,
                    season=season,
                )
            except (ArchticsAuthError, ArchticsError) as e:
                logger.warning(f"Archtics error for '{show_title}': {e}")
        
        # Check if we got any data
        if not tm_events and not archtics_sales and not archtics_performances:
            result.error_message = "No data retrieved from APIs"
            logger.warning(f"No data found for '{show_title}'")
            return result
        
        # Normalize data
        normalizer = ShowDataNormalizer()
        normalized = normalizer.normalize(
            show_title=show_title,
            show_id=show_id,
            tm_events=tm_events,
            archtics_performances=archtics_performances,
            archtics_sales=archtics_sales,
            archtics_venue=archtics_venue,
            season=season,
        )
        
        # Export CSV
        output_path = export_show_csv(
            normalized,
            show_id=show_id,
        )
        
        result.success = True
        result.output_path = output_path
        result.total_tickets = normalized.total_tickets_all
        result.performances = normalized.performance_count_total
        
        logger.info(f"Success: {output_path} ({normalized.total_tickets_all:,} tickets)")
        
    except Exception as e:
        result.error_message = str(e)
        logger.error(f"Error processing '{show_title}': {e}")
    
    return result


def run_batch_mode(
    csv_path: str,
    season: Optional[str],
    city: Optional[str],
    tm_available: bool,
    archtics_available: bool,
    tm_only: bool,
    archtics_only: bool,
    dry_run: bool,
) -> int:
    """
    Run batch processing from a CSV file.
    
    Args:
        csv_path: Path to the CSV file with show titles
        season: Optional season filter
        city: Optional city filter
        tm_available: Whether Ticketmaster credentials are available
        archtics_available: Whether Archtics credentials are available
        tm_only: Only fetch from Ticketmaster
        archtics_only: Only fetch from Archtics
        dry_run: If True, show what would be done without API calls
        
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    logger.info("=" * 60)
    logger.info("Batch Mode: Pull Show Data from CSV")
    logger.info("=" * 60)
    logger.info(f"CSV file: {csv_path}")
    if season:
        logger.info(f"Season filter: {season}")
    if city:
        logger.info(f"City filter: {city}")
    
    # Read titles from CSV
    try:
        all_titles = read_show_titles_from_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1
    
    if not all_titles:
        logger.error("No show titles found in CSV")
        return 1
    
    # Deduplicate
    unique_titles, duplicate_count = deduplicate_titles(all_titles)
    
    logger.info(f"\nFound {len(unique_titles)} unique shows to process")
    if duplicate_count > 0:
        logger.info(f"Skipping {duplicate_count} duplicate entries")
    
    # Dry run mode
    if dry_run:
        logger.info("\n[DRY RUN] Would process the following shows:")
        for i, title in enumerate(unique_titles, 1):
            show_id = generate_show_id(title, season)
            logger.info(f"  {i:3}. {title} -> data/{show_id}_archtics_ticketmaster.csv")
        logger.info(f"\n[DRY RUN] Would fetch from:")
        if tm_available and not archtics_only:
            logger.info("  - Ticketmaster Discovery API")
        if archtics_available and not tm_only:
            logger.info("  - Archtics Reporting API")
        return 0
    
    # Process each show
    summary = BatchSummary(total_shows=len(unique_titles), skipped_duplicates=duplicate_count)
    
    for i, title in enumerate(unique_titles, 1):
        logger.info(f"\n[{i}/{len(unique_titles)}] Processing: {title}")
        
        result = process_single_show(
            show_title=title,
            season=season,
            city=city,
            tm_available=tm_available,
            archtics_available=archtics_available,
            tm_only=tm_only,
            archtics_only=archtics_only,
        )
        
        summary.add_result(result)
    
    # Print summary report
    print_batch_summary(summary)
    
    # Return success if at least some shows were processed
    return 0 if summary.successful > 0 else 1


def print_batch_summary(summary: BatchSummary) -> None:
    """Print a summary report of the batch run."""
    logger.info("\n" + "=" * 60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total shows in CSV:     {summary.total_shows + summary.skipped_duplicates}")
    logger.info(f"Unique shows processed: {summary.total_shows}")
    logger.info(f"Duplicates skipped:     {summary.skipped_duplicates}")
    logger.info("-" * 60)
    logger.info(f"Successful:  {summary.successful}")
    logger.info(f"Failed:      {summary.failed}")
    
    if summary.successful > 0:
        logger.info("\n✓ SUCCESSFUL:")
        for result in summary.results:
            if result.success:
                logger.info(
                    f"  - {result.show_title}: {result.total_tickets:,} tickets, "
                    f"{result.performances} performances"
                )
    
    if summary.failed > 0:
        logger.info("\n✗ FAILED:")
        for result in summary.results:
            if not result.success:
                error_msg = result.error_message or "Unknown error"
                logger.info(f"  - {result.show_title}: {error_msg}")
    
    logger.info("\n" + "=" * 60)
    
    # Calculate totals
    total_tickets = sum(r.total_tickets for r in summary.results if r.success)
    total_performances = sum(r.performances for r in summary.results if r.success)
    
    if summary.successful > 0:
        logger.info(f"Total tickets fetched:      {total_tickets:,}")
        logger.info(f"Total performances fetched: {total_performances}")
    
    logger.info("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pull show data from Ticketmaster and Archtics APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single show mode
  python scripts/pull_show_data.py --show_title "The Nutcracker" --season 2024-25
  python scripts/pull_show_data.py --show_id nutcracker-2024 --city Calgary
  python scripts/pull_show_data.py --show_title "Swan Lake" --output custom_output.csv

  # Batch mode (from CSV)
  python scripts/pull_show_data.py --from_csv data/productions/history_city_sales.csv
  python scripts/pull_show_data.py --from_csv data/productions/history_city_sales.csv --season 2024-25
  python scripts/pull_show_data.py --from_csv data/productions/history_city_sales.csv --dry-run

Environment Variables:
  TM_API_KEY           Ticketmaster API key
  ARCHTICS_API_KEY     Archtics API key
  ARCHTICS_BASE_URL    Archtics organization endpoint

For full documentation, see README.md section "Archtics + Ticketmaster Integration".
        """,
    )
    
    parser.add_argument(
        "--show_title",
        help="Show title to search for (e.g., 'The Nutcracker'). Required if --show_id or --from_csv is not provided.",
    )
    parser.add_argument(
        "--show_id",
        help="Show identifier for the output file (e.g., 'nutcracker-2024'). Required if --show_title or --from_csv is not provided.",
    )
    parser.add_argument(
        "--from_csv",
        metavar="CSV_PATH",
        help="Path to CSV file with show titles for batch processing (expects 'show_title' column).",
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
    
    # Check credentials early for both modes
    tm_available, archtics_available = check_credentials()
    
    # Handle batch mode
    if args.from_csv:
        # Dry run doesn't require credentials
        if not args.dry_run and not tm_available and not archtics_available:
            logger.error(
                "No API credentials available. Please set environment variables.\n"
                "See .env.example for required variables."
            )
            return 1
        
        return run_batch_mode(
            csv_path=args.from_csv,
            season=args.season,
            city=args.city,
            tm_available=tm_available,
            archtics_available=archtics_available,
            tm_only=args.tm_only,
            archtics_only=args.archtics_only,
            dry_run=args.dry_run,
        )
    
    # Single show mode - validate arguments
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
