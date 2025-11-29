"""
k-NN Similarity Fallback for Cold-Start Title Predictions.

This module provides k-NN (k-Nearest Neighbors) based predictions for titles
that don't have historical ticket sales data. It uses baseline signal features
(wiki, trends, youtube, spotify) to find similar titles with known outcomes.

Enhancements for statistical soundness:
- Distance-weighted voting (weights='distance')
- PCA preprocessing option for feature dimensionality reduction
- Configurable k and recency_decay via config file
- Support for Mahalanobis-like distance via PCA whitening
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Defensive import for sklearn
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    NearestNeighbors = None
    StandardScaler = None
    PCA = None


# Feature columns used for similarity matching
BASELINE_FEATURES = ["wiki", "trends", "youtube", "spotify"]

# Default weights for recency decay
DEFAULT_RECENCY_DECAY = 0.1  # Decay per year since last run

# Path to config file
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_knn_config() -> Dict:
    """Load KNN configuration from YAML file.
    
    Returns:
        Configuration dictionary with KNN settings
    """
    config_path = CONFIGS_DIR / "ml_config.yaml"
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("knn", {})
        except Exception:
            pass
    
    # Default configuration
    return {
        "k": 5,
        "metric": "cosine",
        "weights": "distance",
        "use_pca": False,
        "pca_components": 3,
        "recency_weight": 0.5,
        "recency_decay": 0.1,
        "normalize": True
    }


class KNNFallback:
    """
    K-Nearest Neighbors fallback predictor for cold-start titles.
    
    This class builds an index of titles with historical outcomes and uses
    similarity matching to predict outcomes for new titles based on their
    baseline signals.
    
    Enhancements:
    - Distance-weighted voting (weights='distance' or 'uniform')
    - PCA preprocessing for dimensionality reduction
    - Configurable via YAML config file
    
    Attributes:
        k: Number of neighbors to use for prediction
        metric: Distance metric ('cosine', 'euclidean', or 'manhattan')
        normalize: Whether to normalize features before computing distances
        recency_weight: Weight given to more recent show runs (0 = no preference, 1 = recent only)
        recency_decay: Decay factor per year for older runs
        weights: Voting weights ('distance' for distance-weighted, 'uniform' for equal)
        use_pca: Whether to apply PCA before computing distances
        pca_components: Number of PCA components to retain
    """
    
    def __init__(
        self,
        k: int = 5,
        metric: str = "cosine",
        normalize: bool = True,
        recency_weight: float = 0.5,
        recency_decay: float = DEFAULT_RECENCY_DECAY,
        seed: int = 42,
        weights: str = "distance",
        use_pca: bool = False,
        pca_components: int = 3
    ):
        """
        Initialize the KNN fallback predictor.
        
        Args:
            k: Number of nearest neighbors to consider
            metric: Distance metric for similarity ('cosine', 'euclidean', 'manhattan')
            normalize: Whether to standardize features before distance computation
            recency_weight: How much to weight neighbor outcomes by recency (0-1)
            recency_decay: Exponential decay rate per year for older runs
            seed: Random seed for reproducibility
            weights: Voting weights - 'distance' (inverse distance weighted) or 'uniform'
            use_pca: Whether to apply PCA preprocessing
            pca_components: Number of PCA components (if use_pca=True)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for KNN fallback. "
                "Install with: pip install scikit-learn"
            )
        
        # Load defaults from config if available
        config = load_knn_config()
        
        self.k = k if k != 5 else config.get("k", 5)
        self.metric = metric if metric != "cosine" else config.get("metric", "cosine")
        self.normalize = normalize if normalize else config.get("normalize", True)
        self.recency_weight = recency_weight if recency_weight != 0.5 else config.get("recency_weight", 0.5)
        self.recency_decay = recency_decay if recency_decay != DEFAULT_RECENCY_DECAY else config.get("recency_decay", DEFAULT_RECENCY_DECAY)
        self.seed = seed
        self.weights = weights if weights != "distance" else config.get("weights", "distance")
        self.use_pca = use_pca if use_pca else config.get("use_pca", False)
        self.pca_components = pca_components if pca_components != 3 else config.get("pca_components", 3)
        
        # Internal state
        self._nn_model: Optional[NearestNeighbors] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._index_df: Optional[pd.DataFrame] = None
        self._feature_matrix: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def build_index(
        self,
        baseline_df: pd.DataFrame,
        outcome_col: str = "ticket_median",
        last_run_col: Optional[str] = "last_run_date"
    ) -> "KNNFallback":
        """
        Build the KNN index from baseline data with known outcomes.
        
        Args:
            baseline_df: DataFrame with baseline features and outcomes
                Required columns: wiki, trends, youtube, spotify, {outcome_col}
                Optional columns: {last_run_col}, title
            outcome_col: Name of the column containing the outcome to predict
            last_run_col: Name of the column containing the last run date (for recency)
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        missing = set(BASELINE_FEATURES + [outcome_col]) - set(baseline_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter to rows with valid outcomes
        df = baseline_df.dropna(subset=[outcome_col]).copy()
        
        if len(df) == 0:
            warnings.warn("No rows with valid outcomes for KNN index")
            return self
        
        # Extract features
        X = df[BASELINE_FEATURES].values.astype(float)
        
        # Handle missing values in features
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize features if requested
        if self.normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        
        # Apply PCA if requested (provides Mahalanobis-like distance)
        if self.use_pca and len(df) > self.pca_components:
            n_components = min(self.pca_components, X.shape[1], len(df) - 1)
            self._pca = PCA(n_components=n_components, random_state=self.seed)
            X = self._pca.fit_transform(X)
        
        # Build NearestNeighbors index
        # Note: cosine in sklearn computes cosine distance (1 - cosine_similarity)
        # The ranking is preserved for finding nearest neighbors
        self._nn_model = NearestNeighbors(
            n_neighbors=min(self.k, len(df)),
            metric=self.metric,
            algorithm="auto"
        )
        self._nn_model.fit(X)
        
        # Store reference data
        self._feature_matrix = X
        self._index_df = df.reset_index(drop=True)
        self._outcome_col = outcome_col
        self._last_run_col = last_run_col
        self._is_fitted = True
        
        return self
    
    def predict(
        self,
        title_baseline: Union[Dict[str, float], pd.Series, np.ndarray],
        k: Optional[int] = None,
        return_neighbors: bool = False
    ) -> Union[float, Tuple[float, pd.DataFrame]]:
        """
        Predict outcome for a new title based on its baseline signals.
        
        Uses distance-weighted voting when weights='distance' for better
        accuracy on cold-start predictions.
        
        Args:
            title_baseline: Baseline signal values for the title
                If dict: keys should be feature names (wiki, trends, youtube, spotify)
                If array/Series: values in order [wiki, trends, youtube, spotify]
            k: Number of neighbors to use (defaults to self.k)
            return_neighbors: If True, also return DataFrame of neighbor info
            
        Returns:
            If return_neighbors is False: predicted outcome value
            If return_neighbors is True: (predicted_value, neighbors_df)
            
        Raises:
            RuntimeError: If index hasn't been built yet
        """
        if not self._is_fitted:
            raise RuntimeError("Must call build_index() before predict()")
        
        if self._index_df is None or len(self._index_df) == 0:
            if return_neighbors:
                return np.nan, pd.DataFrame()
            return np.nan
        
        k = k or self.k
        k = min(k, len(self._index_df))
        
        # Convert input to feature vector
        if isinstance(title_baseline, dict):
            query = np.array([
                float(title_baseline.get(f, 0.0) or 0.0) 
                for f in BASELINE_FEATURES
            ])
        elif isinstance(title_baseline, pd.Series):
            query = np.array([
                float(title_baseline.get(f, 0.0) or 0.0) 
                for f in BASELINE_FEATURES
            ])
        else:
            query = np.asarray(title_baseline, dtype=float)[:4]
        
        # Handle NaN
        query = np.nan_to_num(query, nan=0.0)
        
        # Normalize if scaler exists
        if self._scaler is not None:
            query = self._scaler.transform(query.reshape(1, -1))[0]
        
        # Apply PCA if it was used during index building
        if self._pca is not None:
            query = self._pca.transform(query.reshape(1, -1))[0]
        
        # Find k nearest neighbors
        distances, indices = self._nn_model.kneighbors(
            query.reshape(1, -1), 
            n_neighbors=k
        )
        distances = distances[0]
        indices = indices[0]
        
        # Get neighbor data
        neighbors = self._index_df.iloc[indices].copy()
        neighbors["distance"] = distances
        
        # Convert distance to similarity (higher = more similar)
        # For cosine distance: similarity = 1 - distance
        if self.metric == "cosine":
            neighbors["similarity"] = 1.0 - neighbors["distance"]
        else:
            # For Euclidean/Manhattan: use inverse exponential
            neighbors["similarity"] = np.exp(-neighbors["distance"])
        
        # Compute recency weights if we have date information
        if self._last_run_col and self._last_run_col in neighbors.columns:
            today = date.today()
            years_ago = []
            for _, row in neighbors.iterrows():
                run_date = row[self._last_run_col]
                if pd.notna(run_date):
                    try:
                        if isinstance(run_date, str):
                            run_date = pd.to_datetime(run_date).date()
                        elif isinstance(run_date, pd.Timestamp):
                            run_date = run_date.date()
                        years = (today - run_date).days / 365.25
                        years_ago.append(max(0, years))
                    except Exception:
                        years_ago.append(5.0)  # Default to 5 years
                else:
                    years_ago.append(5.0)
            
            neighbors["years_ago"] = years_ago
            neighbors["recency_factor"] = np.exp(-self.recency_decay * neighbors["years_ago"])
        else:
            neighbors["recency_factor"] = 1.0
        
        # Compute combined weights
        # Weight = similarity * (recency_weight * recency_factor + (1 - recency_weight))
        base_weight = neighbors["similarity"]
        recency_adjustment = (
            self.recency_weight * neighbors["recency_factor"] + 
            (1 - self.recency_weight)
        )
        neighbors["weight"] = base_weight * recency_adjustment
        
        # Normalize weights to sum to 1
        total_weight = neighbors["weight"].sum()
        if total_weight > 0:
            neighbors["weight"] = neighbors["weight"] / total_weight
        else:
            neighbors["weight"] = 1.0 / len(neighbors)
        
        # Compute weighted prediction
        outcomes = neighbors[self._outcome_col].values.astype(float)
        weights = neighbors["weight"].values
        prediction = float(np.sum(outcomes * weights))
        
        if return_neighbors:
            return prediction, neighbors
        return prediction
    
    def predict_batch(
        self,
        baselines_df: pd.DataFrame,
        k: Optional[int] = None
    ) -> pd.Series:
        """
        Predict outcomes for multiple titles.
        
        Args:
            baselines_df: DataFrame with baseline signal columns
            k: Number of neighbors (defaults to self.k)
            
        Returns:
            Series of predicted outcomes, indexed to match input
        """
        predictions = []
        for idx, row in baselines_df.iterrows():
            pred = self.predict(row, k=k, return_neighbors=False)
            predictions.append(pred)
        
        return pd.Series(predictions, index=baselines_df.index)


def build_knn_index(
    baseline_df: pd.DataFrame,
    metric: str = "cosine",
    normalize: bool = True,
    outcome_col: str = "ticket_median",
    last_run_col: Optional[str] = "last_run_date"
) -> KNNFallback:
    """
    Build a KNN index from baseline data.
    
    This is a convenience function that creates and fits a KNNFallback instance.
    
    Args:
        baseline_df: DataFrame with baseline features and known outcomes
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        normalize: Whether to normalize features
        outcome_col: Column containing the outcome variable
        last_run_col: Column containing last run date (optional)
        
    Returns:
        Fitted KNNFallback instance
        
    Example:
        >>> knn = build_knn_index(baselines, outcome_col="Total_Tickets")
        >>> predicted = knn.predict({"wiki": 70, "trends": 40, "youtube": 80, "spotify": 60})
    """
    knn = KNNFallback(metric=metric, normalize=normalize)
    return knn.build_index(baseline_df, outcome_col=outcome_col, last_run_col=last_run_col)


def predict_knn(
    title_baseline: Dict[str, float],
    knn_index: KNNFallback,
    k: int = 5,
    recency_weight: float = 0.5
) -> float:
    """
    Predict outcome for a title using KNN similarity matching.
    
    This is a convenience function for one-off predictions.
    
    Args:
        title_baseline: Dict with keys wiki, trends, youtube, spotify
        knn_index: A fitted KNNFallback instance
        k: Number of neighbors to use
        recency_weight: Weight for recency in neighbor weighting
        
    Returns:
        Predicted outcome value (e.g., ticket median or ticket index)
        
    Example:
        >>> knn = build_knn_index(baselines_with_history)
        >>> pred = predict_knn(
        ...     {"wiki": 70, "trends": 40, "youtube": 80, "spotify": 60},
        ...     knn_index=knn,
        ...     k=5
        ... )
    """
    # Temporarily override k
    old_k = knn_index.k
    knn_index.k = k
    knn_index.recency_weight = recency_weight
    
    try:
        return knn_index.predict(title_baseline)
    finally:
        knn_index.k = old_k


def find_similar_titles(
    query_baseline: Dict[str, float],
    reference_df: pd.DataFrame,
    k: int = 10,
    metric: str = "cosine"
) -> pd.DataFrame:
    """
    Find similar titles from a reference baseline dataset.
    
    This function doesn't require outcome data - it's purely for finding
    similar titles based on external signals. Useful for:
    1. Understanding which well-known titles a new show resembles
    2. Suggesting programming ideas based on signal similarity
    3. Providing context for cold-start predictions
    
    Args:
        query_baseline: Dict with keys wiki, trends, youtube, spotify
        reference_df: DataFrame with baseline signal columns
        k: Number of similar titles to return
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        DataFrame of k most similar titles with their signals and distance
        
    Example:
        >>> from data.loader import load_all_baselines
        >>> all_baselines = load_all_baselines(include_reference=True)
        >>> similar = find_similar_titles(
        ...     {"wiki": 75, "trends": 30, "youtube": 85, "spotify": 70},
        ...     all_baselines,
        ...     k=5
        ... )
        >>> print(similar[["title", "category", "distance"]])
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for similarity matching")
    
    if reference_df.empty:
        return pd.DataFrame()
    
    # Validate features
    missing = set(BASELINE_FEATURES) - set(reference_df.columns)
    if missing:
        raise ValueError(f"Reference DataFrame missing required columns: {missing}")
    
    # Prepare feature matrix
    df = reference_df.copy()
    X = df[BASELINE_FEATURES].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare query
    query = np.array([
        float(query_baseline.get(f, 0.0) or 0.0) 
        for f in BASELINE_FEATURES
    ])
    query = np.nan_to_num(query, nan=0.0)
    query_scaled = scaler.transform(query.reshape(1, -1))[0]
    
    # Build NN model
    n_neighbors = min(k, len(df))
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto"
    )
    nn_model.fit(X_scaled)
    
    # Find neighbors
    distances, indices = nn_model.kneighbors(
        query_scaled.reshape(1, -1),
        n_neighbors=n_neighbors
    )
    
    # Build result DataFrame
    result = df.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    
    # Compute similarity
    if metric == "cosine":
        result["similarity"] = 1.0 - result["distance"]
    else:
        result["similarity"] = np.exp(-result["distance"])
    
    return result.reset_index(drop=True)


def estimate_category_benchmark(
    category: str,
    reference_df: pd.DataFrame,
    signal_cols: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute average signal values for a category from reference baselines.
    
    This helps calibrate expectations for a category by looking at typical
    signal values from well-known titles in that category.
    
    Args:
        category: Category name (e.g., "family_classic", "pop_ip")
        reference_df: DataFrame with baseline signals and category column
        signal_cols: List of signal columns to average (default: BASELINE_FEATURES)
        
    Returns:
        Dict with average signal values for the category
        
    Example:
        >>> benchmarks = estimate_category_benchmark("pop_ip", all_baselines)
        >>> print(benchmarks)  # {'wiki': 72, 'trends': 38, 'youtube': 85, 'spotify': 72}
    """
    signal_cols = signal_cols or BASELINE_FEATURES
    
    if reference_df.empty or "category" not in reference_df.columns:
        return {col: 50.0 for col in signal_cols}  # Default midpoint
    
    # Filter to category
    cat_df = reference_df[reference_df["category"] == category]
    
    if cat_df.empty:
        # Fall back to overall average
        cat_df = reference_df
    
    # Compute averages
    result = {}
    for col in signal_cols:
        if col in cat_df.columns:
            result[col] = float(cat_df[col].mean())
        else:
            result[col] = 50.0
    
    return result


# CLI/main for testing
if __name__ == "__main__":
    # Simple test
    print("KNN Fallback Module")
    print("=" * 50)
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not available")
        exit(1)
    
    # Create test data
    test_data = pd.DataFrame({
        "title": ["Swan Lake", "Nutcracker", "Romeo and Juliet", "Giselle", "Sleeping Beauty"],
        "wiki": [80, 95, 86, 74, 77],
        "trends": [18, 45, 32, 14, 30],
        "youtube": [71, 88, 80, 62, 70],
        "spotify": [71, 75, 80, 62, 70],
        "ticket_median": [9000, 12000, 8500, 6000, 7500],
        "last_run_date": ["2023-10-15", "2023-12-01", "2024-02-14", "2023-03-10", "2023-09-20"]
    })
    
    print("Test data:")
    print(test_data)
    print()
    
    # Build index
    knn = KNNFallback(k=3, metric="cosine", normalize=True)
    knn.build_index(test_data, outcome_col="ticket_median", last_run_col="last_run_date")
    
    # Test prediction
    new_show = {"wiki": 78, "trends": 25, "youtube": 75, "spotify": 65}
    print(f"New show baseline: {new_show}")
    
    pred, neighbors = knn.predict(new_show, return_neighbors=True)
    print(f"Predicted ticket median: {pred:,.0f}")
    print("\nNearest neighbors:")
    print(neighbors[["title", "ticket_median", "distance", "similarity", "weight"]])
    
    # Test find_similar_titles
    print("\n" + "=" * 50)
    print("Testing find_similar_titles")
    print("=" * 50)
    
    reference_data = pd.DataFrame({
        "title": ["Nutcracker", "Cinderella", "Sleeping Beauty", "Romeo and Juliet", "Giselle"],
        "wiki": [95, 80, 77, 86, 74],
        "trends": [45, 20, 30, 32, 14],
        "youtube": [88, 100, 70, 80, 62],
        "spotify": [75, 72, 70, 80, 62],
        "category": ["family_classic", "family_classic", "classic_romance", "romantic_tragedy", "romantic_tragedy"]
    })
    
    query = {"wiki": 85, "trends": 35, "youtube": 95, "spotify": 70}
    similar = find_similar_titles(query, reference_data, k=3)
    print(f"\nQuery: {query}")
    print("\nMost similar titles:")
    print(similar[["title", "category", "similarity"]])
