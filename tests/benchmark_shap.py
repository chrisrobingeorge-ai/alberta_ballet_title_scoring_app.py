"""
SHAP Performance Benchmark Suite

Benchmarks:
- SHAP computation speed (cold vs warm cache)
- Memory usage
- Scaling behavior (increasing number of titles)
- Cache efficiency

Run with: python tests/benchmark_shap.py
"""

import time
import numpy as np
import pandas as pd
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from ml.shap_explainer import SHAPExplainer, explain_predictions_batch, clear_shap_cache


class SHAPBenchmark:
    """Benchmark suite for SHAP performance."""
    
    def __init__(self, n_train: int = 50, n_test: int = 20, n_features: int = 4):
        """Initialize benchmark data."""
        np.random.seed(42)
        
        self.n_train = n_train
        self.n_test = n_test
        self.n_features = n_features
        
        # Create training data
        self.X_train = pd.DataFrame(
            np.random.uniform(10, 100, (n_train, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.uniform(30, 150, n_train)
        
        # Create test data
        self.X_test = pd.DataFrame(
            np.random.uniform(10, 100, (n_test, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Train model
        self.model = Ridge(alpha=5.0, random_state=42)
        self.model.fit(self.X_train, self.y_train)
    
    def benchmark_cold_cache(self) -> dict:
        """Benchmark SHAP with no caching (cold)."""
        print("\n" + "="*60)
        print("BENCHMARK: Cold Cache (no caching)")
        print("="*60)
        
        results = {}
        
        # Create explainer without cache
        explainer = SHAPExplainer(self.model, self.X_train)
        
        # Benchmark batch computation
        start = time.time()
        explanations = explain_predictions_batch(
            explainer, self.X_test, use_cache=False, verbose=False
        )
        elapsed = time.time() - start
        
        results['total_time'] = elapsed
        results['time_per_prediction'] = elapsed / len(self.X_test)
        results['predictions_per_second'] = len(self.X_test) / elapsed
        
        print(f"Total time: {elapsed:.4f}s")
        print(f"Time per prediction: {results['time_per_prediction']*1000:.2f}ms")
        print(f"Predictions/second: {results['predictions_per_second']:.1f}")
        print(f"Test set size: {len(self.X_test)} predictions")
        
        return results
    
    def benchmark_warm_cache(self) -> dict:
        """Benchmark SHAP with warm cache (repeated data)."""
        print("\n" + "="*60)
        print("BENCHMARK: Warm Cache (repeated predictions)")
        print("="*60)
        
        results = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                self.model, self.X_train, cache_dir=tmpdir
            )
            
            # First pass (populate cache)
            print("Populating cache...")
            start = time.time()
            explain_predictions_batch(
                explainer, self.X_test, use_cache=True, verbose=False
            )
            populate_time = time.time() - start
            print(f"Cache population: {populate_time:.4f}s")
            
            # Second pass (use cache)
            print("Using warm cache...")
            start = time.time()
            explanations = explain_predictions_batch(
                explainer, self.X_test, use_cache=True, verbose=False
            )
            elapsed = time.time() - start
            
            results['total_time'] = elapsed
            results['time_per_prediction'] = elapsed / len(self.X_test)
            results['predictions_per_second'] = len(self.X_test) / elapsed if elapsed > 0 else float('inf')
            results['populate_time'] = populate_time
            
            print(f"Total time (warm): {elapsed:.6f}s")
            print(f"Time per prediction: {results['time_per_prediction']*1e6:.2f}µs")
            print(f"Predictions/second: {results['predictions_per_second']:.0f}")
        
        return results
    
    def benchmark_scaling(self) -> dict:
        """Benchmark scaling with increasing dataset size."""
        print("\n" + "="*60)
        print("BENCHMARK: Scaling (increasing dataset size)")
        print("="*60)
        
        results = {}
        sizes = [5, 10, 20, 50]
        
        for size in sizes:
            X_test = self.X_test[:size]
            
            explainer = SHAPExplainer(self.model, self.X_train)
            
            start = time.time()
            explain_predictions_batch(
                explainer, X_test, use_cache=False, verbose=False
            )
            elapsed = time.time() - start
            
            results[size] = {
                'size': size,
                'time': elapsed,
                'time_per_pred': elapsed / size
            }
            
            print(f"  {size:3d} predictions: {elapsed:8.4f}s ({elapsed/size*1000:6.2f}ms each)")
        
        return results
    
    def benchmark_memory(self) -> dict:
        """Benchmark memory usage with caching."""
        print("\n" + "="*60)
        print("BENCHMARK: Memory Usage (cache)")
        print("="*60)
        
        import sys
        
        results = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                self.model, self.X_train, cache_dir=tmpdir
            )
            
            # Populate cache
            explain_predictions_batch(
                explainer, self.X_test, use_cache=True, verbose=False
            )
            
            # Estimate memory
            cache_size_bytes = sum(
                sys.getsizeof(v) for v in explainer._explanation_cache.values()
            )
            cache_size_mb = cache_size_bytes / (1024 * 1024)
            per_entry = cache_size_bytes / len(explainer._explanation_cache)
            
            results['cache_entries'] = len(explainer._explanation_cache)
            results['total_bytes'] = cache_size_bytes
            results['total_mb'] = cache_size_mb
            results['bytes_per_entry'] = per_entry
            
            print(f"Cache entries: {results['cache_entries']}")
            print(f"Total cache size: {cache_size_mb:.2f} MB")
            print(f"Per-entry overhead: {per_entry:.0f} bytes")
        
        return results
    
    def benchmark_cache_speedup(self) -> dict:
        """Benchmark speedup from caching."""
        print("\n" + "="*60)
        print("BENCHMARK: Cache Speedup")
        print("="*60)
        
        results = {}
        
        # Cold time
        explainer = SHAPExplainer(self.model, self.X_train)
        start = time.time()
        explain_predictions_batch(
            explainer, self.X_test[:5], use_cache=False, verbose=False
        )
        cold_time = time.time() - start
        
        # Warm time
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                self.model, self.X_train, cache_dir=tmpdir
            )
            
            # Populate
            explain_predictions_batch(
                explainer, self.X_test[:5], use_cache=True, verbose=False
            )
            
            # Measure warm
            start = time.time()
            explain_predictions_batch(
                explainer, self.X_test[:5], use_cache=True, verbose=False
            )
            warm_time = time.time() - start
        
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        
        results['cold_time'] = cold_time
        results['warm_time'] = warm_time
        results['speedup'] = speedup
        
        print(f"Cold cache (5 predictions): {cold_time:.4f}s")
        print(f"Warm cache (5 predictions): {warm_time:.6f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        return results
    
    def run_all(self) -> dict:
        """Run all benchmarks."""
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "SHAP IMPLEMENTATION PERFORMANCE BENCHMARK SUITE".center(58) + "║")
        print("║" + f"Training samples: {self.n_train}, Test size: {self.n_test}, Features: {self.n_features}".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")
        
        all_results = {}
        
        try:
            all_results['cold_cache'] = self.benchmark_cold_cache()
            all_results['warm_cache'] = self.benchmark_warm_cache()
            all_results['scaling'] = self.benchmark_scaling()
            all_results['memory'] = self.benchmark_memory()
            all_results['cache_speedup'] = self.benchmark_cache_speedup()
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if 'cache_speedup' in all_results:
            speedup = all_results['cache_speedup']['speedup']
            print(f"✓ Cache speedup: {speedup:.1f}x")
        
        if 'cold_cache' in all_results:
            rate = all_results['cold_cache']['predictions_per_second']
            print(f"✓ Throughput (cold): {rate:.1f} predictions/sec")
        
        if 'warm_cache' in all_results:
            rate = all_results['warm_cache']['predictions_per_second']
            print(f"✓ Throughput (warm): {rate:.0f} predictions/sec")
        
        if 'memory' in all_results:
            mb = all_results['memory']['total_mb']
            print(f"✓ Cache memory: {mb:.2f} MB for {all_results['memory']['cache_entries']} entries")
        
        print("\n✅ All benchmarks completed successfully!")
        
        return all_results


def main():
    """Run benchmark suite."""
    # Small dataset
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + "SMALL DATASET BENCHMARK (30 train, 10 test)".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    benchmark_small = SHAPBenchmark(n_train=30, n_test=10, n_features=4)
    results_small = benchmark_small.run_all()
    
    # Larger dataset
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + "LARGER DATASET BENCHMARK (100 train, 30 test)".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    benchmark_large = SHAPBenchmark(n_train=100, n_test=30, n_features=4)
    results_large = benchmark_large.run_all()
    
    print("\n\n✅ Full benchmark suite completed!")


if __name__ == "__main__":
    main()
