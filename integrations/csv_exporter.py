"""
CSV Exporter for Show Data

This module exports normalized show data to CSV files with the target
schema column order.

Output Files:
    Saved to `data/<show_id>_archtics_ticketmaster.csv`

Column Order:
    See normalizer.NORMALIZED_COLUMNS for the exact column order.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from .normalizer import NormalizedShowData, get_normalized_columns

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"


def export_show_csv(
    show_data: Union[NormalizedShowData, List[NormalizedShowData]],
    output_path: Optional[str] = None,
    show_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_header: bool = True,
    append: bool = False,
) -> str:
    """
    Export normalized show data to a CSV file.
    
    The CSV file follows the target schema with columns in the correct order.
    
    Args:
        show_data: Single NormalizedShowData object or list of objects
        output_path: Full path for the output file. If not provided, uses
            `data/<show_id>_archtics_ticketmaster.csv`
        show_id: Show ID for auto-generated filename (used if output_path not provided)
        output_dir: Directory for output file (default: data/)
        include_header: Whether to include the header row
        append: If True, append to existing file instead of overwriting
        
    Returns:
        Path to the created CSV file
        
    Raises:
        ValueError: If neither output_path nor show_id is provided
    """
    # Handle single item or list
    if isinstance(show_data, NormalizedShowData):
        data_list = [show_data]
    else:
        data_list = show_data
    
    if not data_list:
        raise ValueError("No data to export")
    
    # Determine output path
    if output_path:
        file_path = Path(output_path)
    else:
        # Use show_id from parameter or first data item
        if not show_id:
            show_id = data_list[0].show_title_id
        if not show_id:
            raise ValueError("Either output_path or show_id must be provided")
        
        # Sanitize show_id for filename
        safe_id = _sanitize_filename(show_id)
        
        output_dir_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        file_path = output_dir_path / f"{safe_id}_archtics_ticketmaster.csv"
    
    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get column order
    columns = get_normalized_columns()
    
    # Determine file mode
    mode = "a" if append else "w"
    
    # Write CSV
    logger.info(f"Writing {len(data_list)} rows to {file_path}")
    
    with open(file_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        
        # Write header only if not appending or file is empty
        # Check file position to determine if header is needed (safer than stat)
        if include_header and not append:
            writer.writeheader()
        elif include_header and append:
            # For append mode, check if we're at the start of the file
            try:
                if f.tell() == 0 or file_path.stat().st_size == 0:
                    writer.writeheader()
            except OSError:
                # If stat fails, write header anyway to be safe
                writer.writeheader()
        
        for item in data_list:
            row = item.to_dict()
            writer.writerow(row)
    
    logger.info(f"Successfully wrote CSV to {file_path}")
    
    # Log summary
    _log_export_summary(data_list, file_path)
    
    return str(file_path)


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.
    
    Replaces special characters with underscores and removes unsafe characters.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized filename string
    """
    # Replace common separators with underscores
    sanitized = name.replace(" ", "_").replace("-", "_")
    
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, "")
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    # Convert to lowercase for consistency
    sanitized = sanitized.lower()
    
    return sanitized


def _log_export_summary(data_list: List[NormalizedShowData], file_path: Path) -> None:
    """Log a summary of the exported data."""
    total_tickets = sum(item.total_tickets_all for item in data_list)
    total_performances = sum(item.performance_count_total for item in data_list)
    
    # Count warnings
    all_warnings = []
    for item in data_list:
        all_warnings.extend(item.warnings)
    
    logger.info("-" * 60)
    logger.info("Export Summary")
    logger.info("-" * 60)
    logger.info(f"File: {file_path}")
    logger.info(f"Rows: {len(data_list)}")
    logger.info(f"Total tickets: {total_tickets:,}")
    logger.info(f"Total performances: {total_performances}")
    
    if all_warnings:
        logger.warning(f"Warnings ({len(all_warnings)}):")
        # Show unique warnings
        unique_warnings = sorted(set(all_warnings))
        for warning in unique_warnings[:5]:  # Show first 5
            logger.warning(f"  - {warning}")
        if len(unique_warnings) > 5:
            logger.warning(f"  ... and {len(unique_warnings) - 5} more")
    
    logger.info("-" * 60)


def validate_csv_schema(file_path: str) -> tuple[bool, List[str]]:
    """
    Validate that a CSV file matches the expected schema.
    
    Checks that all required columns are present and in the correct order.
    
    Args:
        file_path: Path to the CSV file to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    expected_columns = get_normalized_columns()
    errors: List[str] = []
    
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header is None:
                errors.append("CSV file is empty or has no header")
                return False, errors
            
            # Check for missing columns
            missing = set(expected_columns) - set(header)
            if missing:
                errors.append(f"Missing columns: {sorted(missing)}")
            
            # Check for extra columns
            extra = set(header) - set(expected_columns)
            if extra:
                errors.append(f"Extra columns: {sorted(extra)}")
            
            # Check column order
            if header != expected_columns:
                # Find first mismatch
                for i, (expected, actual) in enumerate(
                    zip(expected_columns, header)
                ):
                    if expected != actual:
                        errors.append(
                            f"Column order mismatch at position {i}: "
                            f"expected '{expected}', got '{actual}'"
                        )
                        break
    
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_export_stats(file_path: str) -> dict:
    """
    Get statistics about an exported CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary with file statistics
    """
    stats = {
        "file_path": file_path,
        "exists": False,
        "row_count": 0,
        "column_count": 0,
        "file_size_bytes": 0,
    }
    
    path = Path(file_path)
    if not path.exists():
        return stats
    
    stats["exists"] = True
    stats["file_size_bytes"] = path.stat().st_size
    
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header:
                stats["column_count"] = len(header)
                stats["row_count"] = sum(1 for _ in reader)
    except Exception as e:
        logger.warning(f"Error reading file stats: {e}")
    
    return stats
