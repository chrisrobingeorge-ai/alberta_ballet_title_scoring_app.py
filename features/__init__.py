"""
Feature engineering modules for ballet ticket sales prediction.
"""

from .title_features import (
    add_title_features,
    is_benchmark_classic,
    count_title_words,
    get_feature_names as get_title_feature_names,
)

from .economic_features import (
    add_economic_features,
    compute_boc_factor,
    compute_alberta_factor,
    get_feature_names as get_economic_feature_names,
)

from .buzz_features import (
    add_buzz_features,
    compute_composite_buzz_score,
    get_feature_names as get_buzz_feature_names,
)

__all__ = [
    # Title features
    'add_title_features',
    'is_benchmark_classic',
    'count_title_words',
    'get_title_feature_names',
    # Economic features
    'add_economic_features',
    'compute_boc_factor',
    'compute_alberta_factor',
    'get_economic_feature_names',
    # Buzz features
    'add_buzz_features',
    'compute_composite_buzz_score',
    'get_buzz_feature_names',
]
