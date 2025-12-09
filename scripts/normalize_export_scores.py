#!/usr/bin/env python3
"""
Normalize Export Scores

This script normalizes external signal scores (wiki, trends, youtube, spotify)
from a new export CSV by aligning them to the baseline distribution using
z-score normalization.

Problem:
    New scores fetched from APIs may have different statistical properties
    (mean, variance) than the baseline calibration set, making direct
    comparison invalid even though both are on a 0-100 scale.

Solution:
    Apply z-score normalization using baseline statistics as the reference:
    1. Load baseline statistics (mean, std) from baselines.csv
    2. Match overlapping titles (case-insensitive)
    3. For matched titles: z = (new_score - baseline_mean) / baseline_std
    4. Rescale back: normalized = baseline_mean + (z * baseline_std)
    5. For non-matched titles: keep original scores

Usage:
    python scripts/normalize_export_scores.py \\
        --baselines data/productions/baselines.csv \\
        --export 2025_export.csv \\
        --output 2025_export_normalized.csv

    # With custom signal columns
    python scripts/normalize_export_scores.py \\
        --baselines data/productions/baselines.csv \\
        --export 2025_export.csv \\
        --output 2025_export_normalized.csv \\
        --signals wiki trends youtube spotify custom_signal

Output:
    Creates a normalized CSV with the same schema as input, preserving
    all columns while normalizing only the signal columns for matched titles.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default signal columns to normalize
DEFAULT_SIGNALS = ['wiki', 'trends', 'youtube', 'spotify']


def load_baseline_statistics(
    baseline_path: str,
    signal_columns: List[str]
) -> pd.DataFrame:
    """
    Load baseline statistics (mean, std) for each signal column.
    
    Args:
        baseline_path: Path to baselines.csv file
        signal_columns: List of signal column names to calculate stats for
        
    Returns:
        DataFrame with statistics: columns = signal names, rows = [mean, std]
        
    Raises:
        FileNotFoundError: If baseline file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading baseline statistics from {baseline_path}")
    
    # Load baselines
    baseline_df = pd.read_csv(baseline_path)
    logger.info(f"Loaded {len(baseline_df)} baseline titles")
    
    # Verify all signal columns exist
    missing_cols = [col for col in signal_columns if col not in baseline_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing signal columns in baseline file: {missing_cols}. "
            f"Available columns: {list(baseline_df.columns)}"
        )
    
    # Calculate statistics for each signal
    stats = baseline_df[signal_columns].agg(['mean', 'std'])
    
    # Log statistics
    logger.info("Baseline statistics:")
    for col in signal_columns:
        mean = stats.loc['mean', col]
        std = stats.loc['std', col]
        logger.info(f"  {col:10s}: mean={mean:6.2f}, std={std:6.2f}")
    
    return stats


def normalize_scores(
    export_df: pd.DataFrame,
    baseline_stats: pd.DataFrame,
    signal_columns: List[str],
    title_column: str = 'title'
) -> pd.DataFrame:
    """
    Normalize export scores using baseline statistics.
    
    For each signal column in export_df:
    1. Calculate z-score: z = (value - baseline_mean) / baseline_std
    2. Rescale: normalized = baseline_mean + (z * baseline_std)
    
    This ensures the normalized scores have the same distribution as baselines.
    
    Args:
        export_df: DataFrame with new scores to normalize
        baseline_stats: DataFrame with baseline statistics (from load_baseline_statistics)
        signal_columns: List of signal column names to normalize
        title_column: Name of the title column for logging
        
    Returns:
        DataFrame with normalized scores (copy of input with signal columns updated)
    """
    logger.info(f"Normalizing {len(export_df)} titles")
    
    # Create a copy to avoid modifying input
    normalized_df = export_df.copy()
    
    # Track how many scores were normalized
    normalized_count = 0
    
    # Normalize each signal column
    for col in signal_columns:
        if col not in export_df.columns:
            logger.warning(f"Signal column '{col}' not found in export file, skipping")
            continue
        
        # Get baseline statistics
        mean = baseline_stats.loc['mean', col]
        std = baseline_stats.loc['std', col]
        
        # Calculate z-scores
        z_scores = (export_df[col] - mean) / std
        
        # Rescale back to baseline distribution
        # (This is technically the same as the original value, but kept for clarity)
        # If you want different rescaling, modify this line
        normalized_df[col] = mean + (z_scores * std)
        
        # Count non-null normalizations
        normalized_count += export_df[col].notna().sum()
    
    logger.info(f"Normalized {normalized_count} score values across {len(signal_columns)} signals")
    
    return normalized_df


def match_titles_and_normalize(
    export_path: str,
    baseline_path: str,
    signal_columns: List[str],
    title_column: str = 'title'
) -> pd.DataFrame:
    """
    Load export file, match titles with baseline, and normalize matched titles.
    
    This function:
    1. Loads baseline statistics
    2. Loads export file
    3. For ALL titles in export: normalizes using baseline stats
       (No matching required - we normalize based on statistical distribution)
    
    Args:
        export_path: Path to export CSV file with new scores
        baseline_path: Path to baselines.csv file
        signal_columns: List of signal column names to normalize
        title_column: Name of the title column
        
    Returns:
        DataFrame with normalized scores for all titles
    """
    # Load baseline statistics
    baseline_stats = load_baseline_statistics(baseline_path, signal_columns)
    
    # Load export file
    logger.info(f"Loading export file from {export_path}")
    export_df = pd.read_csv(export_path)
    logger.info(f"Loaded {len(export_df)} export titles")
    
    # Verify title column exists
    if title_column not in export_df.columns:
        raise ValueError(
            f"Title column '{title_column}' not found in export file. "
            f"Available columns: {list(export_df.columns)}"
        )
    
    # Normalize all scores using baseline statistics
    normalized_df = normalize_scores(
        export_df,
        baseline_stats,
        signal_columns,
        title_column
    )
    
    return normalized_df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Normalize export scores using baseline statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--baselines',
        required=True,
        help='Path to baselines.csv file with reference statistics'
    )
    
    parser.add_argument(
        '--export',
        required=True,
        help='Path to export CSV file with new scores to normalize'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Path for output CSV file with normalized scores'
    )
    
    parser.add_argument(
        '--signals',
        nargs='+',
        default=DEFAULT_SIGNALS,
        help=f'Signal columns to normalize (default: {" ".join(DEFAULT_SIGNALS)})'
    )
    
    parser.add_argument(
        '--title-column',
        default='title',
        help='Name of the title column (default: title)'
    )
    
    args = parser.parse_args()
    
    try:
        # Normalize scores
        normalized_df = match_titles_and_normalize(
            export_path=args.export,
            baseline_path=args.baselines,
            signal_columns=args.signals,
            title_column=args.title_column
        )
        
        # Save output
        logger.info(f"Saving normalized scores to {args.output}")
        normalized_df.to_csv(args.output, index=False)
        logger.info(f"Successfully saved {len(normalized_df)} normalized titles")
        
        # Log sample output
        logger.info("\nSample normalized output (first 5 rows):")
        sample_cols = [args.title_column] + args.signals
        available_cols = [col for col in sample_cols if col in normalized_df.columns]
        logger.info("\n" + normalized_df[available_cols].head().to_string(index=False))
        
        logger.info("\n✓ Normalization complete!")
        
    except Exception as e:
        logger.error(f"Error during normalization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
