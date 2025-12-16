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
    python scripts/normalize_export_scores.py \
        --baselines data/productions/baselines.csv \
        --export 2025_export.csv \
        --output 2025_export_normalized.csv

    # With custom signal columns
    python scripts/normalize_export_scores.py \
        --baselines data/productions/baselines.csv \
        --export 2025_export.csv \
        --output 2025_export_normalized.csv \
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
DEFAULT_SIGNALS = ['wiki', 'trends', 'youtube', 'chartmetric']


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
    Align export scores to baseline distribution using statistical normalization.
    
    This function adjusts new scores to match the baseline distribution by:
    1. Computing statistics (mean, std) from the NEW export scores
    2. Converting new scores to z-scores using NEW statistics
    3. Rescaling z-scores using BASELINE statistics
    
    Formula: aligned = baseline_mean + ((new - new_mean) / new_std) * baseline_std
    
    This ensures the aligned scores have the same statistical properties as the
    baseline distribution (same mean and standard deviation), making them directly
    comparable and suitable for ML models trained on baseline data.
    
    Example:
        If new scores have mean=90, std=5 and baseline has mean=60, std=15:
        - A new score of 95 (1 std above new mean) becomes 75 (1 std above baseline mean)
        - This preserves relative position while aligning absolute scale
    
    Args:
        export_df: DataFrame with new scores to normalize
        baseline_stats: DataFrame with baseline statistics (from load_baseline_statistics)
        signal_columns: List of signal column names to normalize
        title_column: Name of the title column for logging
        
    Returns:
        DataFrame with aligned scores (copy of input with signal columns adjusted)
    """
    logger.info(f"Aligning {len(export_df)} titles to baseline distribution")
    
    # Create a copy to avoid modifying input
    normalized_df = export_df.copy()
    
    # First, calculate statistics from the NEW export data
    export_stats = export_df[signal_columns].agg(['mean', 'std'])
    
    logger.info("Export data statistics:")
    for col in signal_columns:
        if col in export_df.columns:
            exp_mean = export_stats.loc['mean', col]
            exp_std = export_stats.loc['std', col]
            base_mean = baseline_stats.loc['mean', col]
            base_std = baseline_stats.loc['std', col]
            logger.info(f"  {col:10s}: export mean={exp_mean:6.2f}, std={exp_std:6.2f} | "
                       f"baseline mean={base_mean:6.2f}, std={base_std:6.2f}")
    
    # Track how many scores were normalized
    normalized_count = 0
    
    # Align each signal column to baseline distribution
    for col in signal_columns:
        if col not in export_df.columns:
            logger.warning(f"Signal column '{col}' not found in export file, skipping")
            continue
        
        # Get export statistics (from new data)
        export_mean = export_stats.loc['mean', col]
        export_std = export_stats.loc['std', col]
        
        # Get baseline statistics (reference distribution)
        baseline_mean = baseline_stats.loc['mean', col]
        baseline_std = baseline_stats.loc['std', col]
        
        # Handle edge case: zero or near-zero standard deviation
        if pd.isna(export_std) or export_std < 1e-10:
            logger.warning(f"Signal '{col}' has zero or invalid std in export, keeping original values")
            continue
        
        # Distribution alignment transformation:
        # 1. Convert to z-scores using EXPORT statistics (where is value in NEW distribution)
        z_scores = (export_df[col] - export_mean) / export_std
        
        # 2. Rescale using BASELINE statistics (map to BASELINE distribution)
        aligned_scores = baseline_mean + (z_scores * baseline_std)
        
        # Replace original scores with aligned scores
        normalized_df[col] = aligned_scores
        
        # Count non-null normalizations
        normalized_count += export_df[col].notna().sum()
    
    # Verify alignment worked
    final_stats = normalized_df[signal_columns].agg(['mean', 'std'])
    logger.info(f"\nAligned {normalized_count} scores across {len(signal_columns)} signals")
    logger.info("Aligned scores now match baseline distribution:")
    for col in signal_columns:
        if col in normalized_df.columns:
            logger.info(f"  {col:10s}: mean={final_stats.loc['mean', col]:6.2f}, "
                       f"std={final_stats.loc['std', col]:6.2f}")
    
    return normalized_df


def normalize_export_to_baseline(
    export_path: str,
    baseline_path: str,
    signal_columns: List[str],
    title_column: str = 'title'
) -> pd.DataFrame:
    """
    Load export file and align scores to baseline distribution.
    
    This function adjusts new export scores to match the baseline's statistical
    properties (mean and standard deviation). This is necessary when new scores
    from APIs have different calibration than the baseline reference set.
    
    The transformation:
    - Calculates statistics from both NEW export and BASELINE data
    - Maps new scores to baseline distribution via: 
      aligned = baseline_mean + ((new - new_mean) / new_std) * baseline_std
    - Ensures aligned scores are directly comparable to historical data
    
    Process:
    1. Load baseline statistics (mean, std for each signal)
    2. Load export file with new scores
    3. Calculate export statistics (mean, std)
    4. Transform scores to align with baseline distribution
    
    Args:
        export_path: Path to export CSV file with new scores
        baseline_path: Path to baselines.csv file with reference statistics
        signal_columns: List of signal column names to normalize (e.g., ['wiki', 'trends'])
        title_column: Name of the title column in export file
        
    Returns:
        DataFrame with scores aligned to baseline distribution
        
    Raises:
        FileNotFoundError: If baseline or export file doesn't exist
        ValueError: If required columns are missing
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
    
    # Align all scores to baseline distribution
    aligned_df = normalize_scores(
        export_df,
        baseline_stats,
        signal_columns,
        title_column
    )
    
    return aligned_df


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
        # Align scores to baseline distribution
        aligned_df = normalize_export_to_baseline(
            export_path=args.export,
            baseline_path=args.baselines,
            signal_columns=args.signals,
            title_column=args.title_column
        )
        
        # Save output
        logger.info(f"\nSaving aligned scores to {args.output}")
        aligned_df.to_csv(args.output, index=False)
        logger.info(f"Successfully saved {len(aligned_df)} titles with aligned scores")
        
        # Log sample output
        logger.info("\nSample aligned output (first 5 rows):")
        sample_cols = [args.title_column] + args.signals
        available_cols = [col for col in sample_cols if col in aligned_df.columns]
        logger.info("\n" + aligned_df[available_cols].head().to_string(index=False))
        
        logger.info("\n✓ Score alignment complete!")
        logger.info("Output scores now match baseline distribution (same mean and std)")
        
    except Exception as e:
        logger.error(f"Error during alignment: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
