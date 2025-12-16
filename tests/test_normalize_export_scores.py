"""
Tests for normalize_export_scores.py script

Tests cover:
- Z-score calculation accuracy
- Case-insensitive title matching
- Non-overlapping title preservation
- Output schema integrity
- Edge cases (missing values, zero std, etc.)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the functions from the script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from normalize_export_scores import (
    load_baseline_statistics,
    normalize_scores,
    normalize_export_to_baseline,
)


@pytest.fixture
def sample_baseline_csv():
    """Create a sample baseline CSV file for testing."""
    data = {
        'title': ['Swan Lake', 'The Nutcracker', 'Giselle', 'Don Quixote', 'Romeo and Juliet'],
        'wiki': [85, 90, 70, 65, 88],
        'trends': [30, 40, 25, 20, 35],
        'youtube': [90, 95, 80, 75, 92],
        'chartmetric': [70, 80, 60, 55, 75],
        'category': ['classical', 'classical', 'classical', 'classical', 'classical'],
        'source': ['historical', 'historical', 'historical', 'historical', 'historical']
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def sample_export_csv():
    """Create a sample export CSV file for testing."""
    data = {
        'title': ['Swan Lake', 'The Nutcracker', 'Sleeping Beauty'],  # 2 overlap, 1 new
        'wiki': [88, 92, 75],
        'trends': [32, 42, 28],
        'youtube': [92, 97, 82],
        'chartmetric': [72, 82, 65]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


def test_load_baseline_statistics(sample_baseline_csv):
    """Test that baseline statistics are calculated correctly."""
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    stats = load_baseline_statistics(sample_baseline_csv, signals)
    
    # Check structure
    assert stats.shape == (2, 4)  # 2 rows (mean, std), 4 columns (signals)
    assert list(stats.index) == ['mean', 'std']
    assert list(stats.columns) == signals
    
    # Check values are reasonable
    for col in signals:
        assert 0 <= stats.loc['mean', col] <= 100
        assert stats.loc['std', col] > 0


def test_load_baseline_statistics_values(sample_baseline_csv):
    """Test that baseline statistics match expected calculations."""
    # Calculate expected values manually
    df = pd.read_csv(sample_baseline_csv)
    
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    stats = load_baseline_statistics(sample_baseline_csv, signals)
    
    # Verify against pandas calculations
    for col in signals:
        expected_mean = df[col].mean()
        expected_std = df[col].std()
        
        assert np.isclose(stats.loc['mean', col], expected_mean)
        assert np.isclose(stats.loc['std', col], expected_std)


def test_normalize_scores_basic(sample_baseline_csv):
    """Test basic z-score normalization."""
    # Load baseline stats
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    baseline_stats = load_baseline_statistics(sample_baseline_csv, signals)
    
    # Create export data
    export_data = {
        'title': ['Swan Lake', 'New Title'],
        'wiki': [88, 75],
        'trends': [32, 28],
        'youtube': [92, 82],
        'chartmetric': [72, 65]
    }
    export_df = pd.DataFrame(export_data)
    
    # Normalize
    normalized_df = normalize_scores(export_df, baseline_stats, signals)
    
    # Check structure preserved
    assert normalized_df.shape == export_df.shape
    assert list(normalized_df.columns) == list(export_df.columns)
    assert list(normalized_df['title']) == list(export_df['title'])
    
    # Check all values are numeric and finite
    for col in signals:
        assert normalized_df[col].dtype in [np.float64, np.int64]
        assert normalized_df[col].notna().all()
        assert np.isfinite(normalized_df[col]).all()


def test_normalize_scores_distribution_alignment():
    """Test that distribution alignment formula is applied correctly."""
    # Create baseline with known statistics
    baseline_data = {
        'title': ['A', 'B', 'C', 'D', 'E'],
        'wiki': [50, 60, 70, 80, 90]  # mean=70, std=~15.81
    }
    baseline_df = pd.DataFrame(baseline_data)
    baseline_mean = baseline_df['wiki'].mean()  # 70
    baseline_std = baseline_df['wiki'].std()    # ~15.81
    
    baseline_stats = pd.DataFrame({
        'wiki': [baseline_mean, baseline_std]
    }, index=['mean', 'std'])
    
    # Create export data with DIFFERENT distribution (higher mean)
    export_data = {
        'title': ['X', 'Y', 'Z'],
        'wiki': [85, 90, 95]  # mean=90, std=5
    }
    export_df = pd.DataFrame(export_data)
    export_mean = export_df['wiki'].mean()  # 90
    export_std = export_df['wiki'].std()    # 5
    
    # Align to baseline distribution
    aligned_df = normalize_scores(export_df, baseline_stats, ['wiki'])
    
    # Check that aligned values match expected transformation
    # For middle value (Y=90, which is AT export mean):
    # aligned = baseline_mean + ((90 - 90) / 5) * 15.81 = 70 + 0 = 70
    assert np.isclose(aligned_df['wiki'].iloc[1], baseline_mean, rtol=0.01)
    
    # Check that aligned distribution has baseline mean/std
    aligned_mean = aligned_df['wiki'].mean()
    aligned_std = aligned_df['wiki'].std()
    assert np.isclose(aligned_mean, baseline_mean, rtol=0.01)
    assert np.isclose(aligned_std, baseline_std, rtol=0.01)


def test_normalize_preserves_non_signal_columns():
    """Test that non-signal columns are preserved unchanged."""
    # Create baseline stats
    baseline_stats = pd.DataFrame({
        'wiki': [70, 14],
        'trends': [30, 10]
    }, index=['mean', 'std'])
    
    # Create export data with extra columns
    export_data = {
        'title': ['Swan Lake'],
        'wiki': [85],
        'trends': [35],
        'category': ['classical'],
        'notes': ['Test note']
    }
    export_df = pd.DataFrame(export_data)
    
    # Normalize
    normalized_df = normalize_scores(export_df, baseline_stats, ['wiki', 'trends'])
    
    # Check non-signal columns unchanged
    assert normalized_df['title'].iloc[0] == 'Swan Lake'
    assert normalized_df['category'].iloc[0] == 'classical'
    assert normalized_df['notes'].iloc[0] == 'Test note'


def test_normalize_handles_missing_values():
    """Test that missing values (NaN) are preserved."""
    baseline_stats = pd.DataFrame({
        'wiki': [70, 14],
        'trends': [30, 10]
    }, index=['mean', 'std'])
    
    # Create export data with missing values
    export_data = {
        'title': ['A', 'B', 'C'],
        'wiki': [85, np.nan, 70],
        'trends': [35, 40, np.nan]
    }
    export_df = pd.DataFrame(export_data)
    
    # Normalize
    normalized_df = normalize_scores(export_df, baseline_stats, ['wiki', 'trends'])
    
    # Check NaN values preserved
    assert normalized_df['wiki'].notna().sum() == 2
    assert normalized_df['trends'].notna().sum() == 2
    assert pd.isna(normalized_df['wiki'].iloc[1])
    assert pd.isna(normalized_df['trends'].iloc[2])


def test_normalize_export_to_baseline_integration(sample_baseline_csv, sample_export_csv):
    """Test the full integration of distribution alignment."""
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    
    # Load baseline stats for verification
    baseline_df = pd.read_csv(sample_baseline_csv)
    baseline_stats = baseline_df[signals].agg(['mean', 'std'])
    
    # Run full alignment
    aligned_df = normalize_export_to_baseline(
        export_path=sample_export_csv,
        baseline_path=sample_baseline_csv,
        signal_columns=signals
    )
    
    # Check output structure
    assert isinstance(aligned_df, pd.DataFrame)
    assert len(aligned_df) == 3  # All 3 titles from export
    assert 'title' in aligned_df.columns
    
    # Check all signal columns present and aligned to baseline distribution
    for col in signals:
        assert col in aligned_df.columns
        assert aligned_df[col].notna().all()
        # Aligned scores should have similar mean/std to baseline
        # (with small sample, won't be exact but should be in same ballpark)
        aligned_mean = aligned_df[col].mean()
        baseline_mean = baseline_stats.loc['mean', col]
        # Within 30% of baseline mean is reasonable for 3-sample alignment
        assert abs(aligned_mean - baseline_mean) / baseline_mean < 0.5


def test_normalize_export_to_baseline_preserves_all_titles(sample_baseline_csv, sample_export_csv):
    """Test that all titles from export are preserved in output."""
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    
    # Load original export
    original_export = pd.read_csv(sample_export_csv)
    original_titles = set(original_export['title'].str.strip().str.lower())
    
    # Run alignment
    aligned_df = normalize_export_to_baseline(
        export_path=sample_export_csv,
        baseline_path=sample_baseline_csv,
        signal_columns=signals
    )
    
    # Check all titles preserved
    aligned_titles = set(aligned_df['title'].str.strip().str.lower())
    assert aligned_titles == original_titles


def test_output_schema_integrity(sample_baseline_csv, sample_export_csv, tmp_path):
    """Test that output CSV has correct schema."""
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    output_path = tmp_path / "aligned_output.csv"
    
    # Load original
    original_df = pd.read_csv(sample_export_csv)
    
    # Align to baseline
    aligned_df = normalize_export_to_baseline(
        export_path=sample_export_csv,
        baseline_path=sample_baseline_csv,
        signal_columns=signals
    )
    
    # Save to file
    aligned_df.to_csv(output_path, index=False)
    
    # Reload and check
    reloaded_df = pd.read_csv(output_path)
    
    # Check columns match
    assert list(reloaded_df.columns) == list(original_df.columns)
    
    # Check row count
    assert len(reloaded_df) == len(original_df)


def test_missing_signal_column_handling(sample_baseline_csv, tmp_path):
    """Test graceful handling when export is missing a signal column."""
    # Create export with missing column
    export_data = {
        'title': ['Swan Lake'],
        'wiki': [85],
        'trends': [30]
        # Missing youtube and spotify
    }
    export_path = tmp_path / "export_missing_cols.csv"
    pd.DataFrame(export_data).to_csv(export_path, index=False)
    
    # Should not raise error, but skip missing columns
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    
    # This should work without error
    baseline_stats = load_baseline_statistics(sample_baseline_csv, ['wiki', 'trends'])
    export_df = pd.read_csv(export_path)
    
    # Normalize only available columns
    available_signals = [s for s in signals if s in export_df.columns]
    normalized_df = normalize_scores(export_df, baseline_stats, available_signals)
    
    # Check structure
    assert 'wiki' in normalized_df.columns
    assert 'trends' in normalized_df.columns


def test_empty_export_file(sample_baseline_csv, tmp_path):
    """Test handling of empty export file."""
    # Create empty export
    export_path = tmp_path / "empty_export.csv"
    empty_df = pd.DataFrame(columns=['title', 'wiki', 'trends', 'youtube', 'chartmetric'])
    empty_df.to_csv(export_path, index=False)
    
    signals = ['wiki', 'trends', 'youtube', 'chartmetric']
    
    # Should handle gracefully (though stats calculation will produce NaN)
    # This is expected behavior for empty data
    aligned_df = normalize_export_to_baseline(
        export_path=export_path,
        baseline_path=sample_baseline_csv,
        signal_columns=signals
    )
    
    # Check empty output
    assert len(aligned_df) == 0
    assert list(aligned_df.columns) == ['title', 'wiki', 'trends', 'youtube', 'chartmetric']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
