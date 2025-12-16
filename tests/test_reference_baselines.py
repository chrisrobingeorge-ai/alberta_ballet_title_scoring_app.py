"""Tests for baseline loading and k-NN similarity functions.

Baselines are stored in a single file (baselines.csv) with a 'source' column that
distinguishes between:
- 'historical': Alberta Ballet performances with ticket data
- 'external_reference': Well-known titles without AB history (used for k-NN similarity)
"""

import pandas as pd
import pytest

from data.loader import (
    load_baselines,
    load_reference_baselines,
    load_all_baselines,
)
from ml.knn_fallback import find_similar_titles, estimate_category_benchmark


def test_load_baselines():
    """Test that baselines.csv loads correctly with all titles."""
    df = load_baselines()
    assert not df.empty
    # Check expected columns (now includes source and notes)
    for col in ["title", "wiki", "trends", "youtube", "chartmetric", "category", "gender", "source"]:
        assert col in df.columns, f"Missing column: {col}"


def test_load_reference_baselines():
    """Test that load_reference_baselines filters to external_reference titles only."""
    df = load_reference_baselines()
    assert not df.empty
    # Check expected columns
    for col in ["title", "wiki", "trends", "youtube", "chartmetric", "category", "gender", "source"]:
        assert col in df.columns, f"Missing column: {col}"
    # Check source column has expected value (filtered to external_reference only)
    assert (df["source"] == "external_reference").all()


def test_load_all_baselines_with_reference():
    """Test that combined baselines load correctly."""
    df = load_all_baselines(include_reference=True)
    assert not df.empty
    # Should have both sources
    assert "source" in df.columns
    sources = df["source"].unique()
    assert "historical" in sources
    assert "external_reference" in sources


def test_load_all_baselines_without_reference():
    """Test loading baselines without reference data."""
    df = load_all_baselines(include_reference=False)
    assert not df.empty
    # Should only have historical source
    assert "source" in df.columns
    assert (df["source"] == "historical").all()


def test_all_baselines_no_duplicates():
    """Test that combined baselines have no duplicate titles."""
    df = load_all_baselines(include_reference=True)
    assert df["title"].is_unique, "Duplicate titles found in combined baselines"


def test_signal_values_in_range():
    """Test that signal values are in valid 0-100 range."""
    df = load_all_baselines(include_reference=True)
    for col in ["wiki", "trends", "youtube", "chartmetric"]:
        assert df[col].min() >= 0, f"{col} has values below 0"
        assert df[col].max() <= 100, f"{col} has values above 100"


def test_find_similar_titles():
    """Test finding similar titles from reference baselines."""
    all_baselines = load_all_baselines(include_reference=True)
    
    # Query for a family classic type show
    query = {"wiki": 80, "trends": 30, "youtube": 95, "chartmetric": 70}
    similar = find_similar_titles(query, all_baselines, k=5)
    
    assert len(similar) == 5
    assert "title" in similar.columns
    assert "similarity" in similar.columns
    assert "distance" in similar.columns
    # Similarity should be in descending order (highest first)
    assert similar["similarity"].is_monotonic_decreasing


def test_estimate_category_benchmark():
    """Test computing category benchmarks."""
    all_baselines = load_all_baselines(include_reference=True)
    
    benchmarks = estimate_category_benchmark("family_classic", all_baselines)
    
    assert "wiki" in benchmarks
    assert "trends" in benchmarks
    assert "youtube" in benchmarks
    assert "chartmetric" in benchmarks
    
    # Values should be reasonable (between 0 and 100)
    for col, value in benchmarks.items():
        assert 0 <= value <= 100, f"Benchmark {col} = {value} out of range"


def test_estimate_category_benchmark_unknown_category():
    """Test benchmark for unknown category falls back to overall average."""
    all_baselines = load_all_baselines(include_reference=True)
    
    benchmarks = estimate_category_benchmark("nonexistent_category", all_baselines)
    
    # Should still return valid benchmarks (overall averages)
    assert len(benchmarks) == 4
    for col, value in benchmarks.items():
        assert 0 <= value <= 100
