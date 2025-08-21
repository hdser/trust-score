#!/usr/bin/env python3
"""Performance testing for trust network."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import pandas as pd
from trust_network import TrustNetwork
from trust_network.config.settings import NetworkConfig, setup_logging

def generate_random_network(n_nodes: int, n_edges: int, n_converters: int = None) -> dict:
    """Generate a random network for testing."""
    if n_converters is None:
        n_converters = max(1, n_nodes // 20)
    
    # Generate nodes
    nodes = []
    for i in range(n_nodes):
        is_converter = i < n_converters
        nodes.append({
            "label": f"node_{i}",
            "is_converter": is_converter,
            "token_symbol": f"T_{i}" if not is_converter else None,
            "metadata": {"type": "converter" if is_converter else "regular"}
        })
    
    # Generate edges
    edges = []
    for _ in range(n_edges):
        source = f"node_{np.random.randint(n_nodes)}"
        target = f"node_{np.random.randint(n_nodes)}"
        if source != target:  # Avoid self-loops
            weight = np.random.uniform(0.1, 1.0)
            edges.append({
                "source": source,
                "target": target,
                "weight": weight,
                "metadata": {"type": "random"}
            })
    
    # Generate balances
    balances = []
    for i in range(n_nodes):
        holder = f"node_{i}"
        # Each node holds some of several tokens
        for j in range(min(5, n_nodes)):
            token = f"node_{j}"
            amount = np.random.uniform(100, 1000)
            balances.append({
                "holder": holder,
                "token": token,
                "amount": amount
            })
    
    # Generate rates
    rates = []
    for i in range(n_converters):
        converter = f"node_{i}"
        # Each converter has rates for several tokens
        for j in range(min(10, n_nodes)):
            token = f"node_{j}"
            rate = np.random.uniform(0.5, 1.0)
            rates.append({
                "converter": converter,
                "token": token,
                "rate": rate
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "balances": balances,
        "rates": rates
    }

def benchmark_network_size(sizes: list, algorithms: list) -> pd.DataFrame:
    """Benchmark trust computation for different network sizes."""
    results = []
    
    for n_nodes in sizes:
        n_edges = n_nodes * 5  # Average degree of 5
        
        print(f"\nTesting network size: {n_nodes} nodes, {n_edges} edges")
        
        # Generate network
        data = generate_random_network(n_nodes, n_edges)
        
        for social_alg in algorithms:
            for liquidity_alg in ["hybrid", "conductance"]:
                print(f"  Algorithm: {social_alg} + {liquidity_alg}")
                
                # Create configuration
                config = NetworkConfig()
                config.trust_network.algorithms.social = social_alg
                config.trust_network.algorithms.liquidity = liquidity_alg
                
                # Create network
                network = TrustNetwork(config)
                network.load_data(data)
                
                # Measure computation time
                start_time = time.time()
                try:
                    result = network.compute_trust_scores()
                    end_time = time.time()
                    
                    computation_time = end_time - start_time
                    success = True
                    error = None
                    
                except Exception as e:
                    end_time = time.time()
                    computation_time = end_time - start_time
                    success = False
                    error = str(e)
                
                results.append({
                    "nodes": n_nodes,
                    "edges": n_edges,
                    "social_algorithm": social_alg,
                    "liquidity_algorithm": liquidity_alg,
                    "computation_time": computation_time,
                    "time_per_node": computation_time / n_nodes,
                    "success": success,
                    "error": error
                })
                
                print(f"    Time: {computation_time:.3f}s ({computation_time/n_nodes*1000:.2f}ms/node)")
    
    return pd.DataFrame(results)

def benchmark_algorithms(n_nodes: int = 1000) -> pd.DataFrame:
    """Benchmark different algorithm combinations."""
    n_edges = n_nodes * 5
    
    print(f"\nBenchmarking algorithms on {n_nodes} nodes, {n_edges} edges")
    
    # Generate network once
    data = generate_random_network(n_nodes, n_edges)
    
    algorithms = [
        ("eigentrust", "hybrid"),
        ("eigentrust", "conductance"),
        ("appleseed", "hybrid"),
        ("appleseed", "conductance"),
        ("pagerank", "hybrid"),
        ("pagerank", "conductance")
    ]
    
    results = []
    
    for social_alg, liquidity_alg in algorithms:
        print(f"  Testing: {social_alg} + {liquidity_alg}")
        
        # Create configuration
        config = NetworkConfig()
        config.trust_network.algorithms.social = social_alg
        config.trust_network.algorithms.liquidity = liquidity_alg
        
        # Run multiple times for stability
        times = []
        for run in range(3):
            network = TrustNetwork(config)
            network.load_data(data)
            
            start_time = time.time()
            try:
                result = network.compute_trust_scores()
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"    Error: {e}")
                times.append(float('inf'))
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            "social_algorithm": social_alg,
            "liquidity_algorithm": liquidity_alg,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min(times),
            "max_time": max(times)
        })
        
        print(f"    Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    
    return pd.DataFrame(results)

def test_cache_performance(n_nodes: int = 500) -> dict:
    """Test performance impact of caching."""
    n_edges = n_nodes * 5
    data = generate_random_network(n_nodes, n_edges)
    
    print(f"\nTesting cache performance on {n_nodes} nodes")
    
    # Test with cache enabled
    config_cached = NetworkConfig()
    config_cached.trust_network.performance.enable_caching = True
    
    network_cached = TrustNetwork(config_cached)
    network_cached.load_data(data)
    
    # First computation (cold cache)
    start_time = time.time()
    result1 = network_cached.compute_trust_scores()
    time_cold = time.time() - start_time
    
    # Second computation (warm cache)
    start_time = time.time()
    result2 = network_cached.compute_trust_scores()
    time_warm = time.time() - start_time
    
    # Test with cache disabled
    config_no_cache = NetworkConfig()
    config_no_cache.trust_network.performance.enable_caching = False
    
    network_no_cache = TrustNetwork(config_no_cache)
    network_no_cache.load_data(data)
    
    # Computation without cache
    start_time = time.time()
    result3 = network_no_cache.compute_trust_scores()
    time_no_cache = time.time() - start_time
    
    speedup_cache = time_cold / time_warm if time_warm > 0 else 0
    
    print(f"  Cold cache: {time_cold:.3f}s")
    print(f"  Warm cache: {time_warm:.3f}s")
    print(f"  No cache: {time_no_cache:.3f}s")
    print(f"  Cache speedup: {speedup_cache:.1f}x")
    
    return {
        "time_cold": time_cold,
        "time_warm": time_warm,
        "time_no_cache": time_no_cache,
        "speedup": speedup_cache
    }

def main():
    """Run performance tests."""
    print("=" * 70)
    print("TRUST NETWORK - PERFORMANCE TESTING")
    print("=" * 70)
    
   
    
    # Test 1: Network size scaling
    print("\n1. Network Size Scaling Test")
    print("-" * 70)
    
    sizes = [100, 200, 500, 1000]
    algorithms = ["eigentrust", "pagerank"]
    
    size_results = benchmark_network_size(sizes, algorithms)
    
    print("\nScaling Results:")
    print(size_results.groupby(['social_algorithm', 'liquidity_algorithm']).agg({
        'nodes': 'max',
        'computation_time': ['mean', 'std'],
        'time_per_node': ['mean', 'std']
    }).round(4))
    
    # Test 2: Algorithm comparison
    print("\n2. Algorithm Comparison Test")
    print("-" * 70)
    
    alg_results = benchmark_algorithms(n_nodes=1000)
    
    print("\nAlgorithm Performance:")
    print(alg_results.sort_values('avg_time')[['social_algorithm', 'liquidity_algorithm', 'avg_time', 'std_time']].round(4))
    
    # Test 3: Cache performance
    print("\n3. Cache Performance Test")
    print("-" * 70)
    
    cache_results = test_cache_performance(n_nodes=500)
    
    # Test 4: Memory usage (simplified)
    print("\n4. Memory Usage Test")
    print("-" * 70)
    
    import psutil
    process = psutil.Process()
    
    # Measure memory for different sizes
    for n_nodes in [100, 500, 1000]:
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        data = generate_random_network(n_nodes, n_nodes * 5)
        config = NetworkConfig()

        # Setup logging
        setup_logging(config.logging)

        network = TrustNetwork(config)
        network.load_data(data)
        network.compute_trust_scores()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"  {n_nodes} nodes: {memory_used:.1f} MB ({memory_used/n_nodes*1000:.2f} KB/node)")
    
    # Save results
    print("\n5. Saving Results")
    print("-" * 70)
    
    size_results.to_csv("performance_scaling.csv", index=False)
    alg_results.to_csv("performance_algorithms.csv", index=False)
    
    print("  Results saved to:")
    print("    - performance_scaling.csv")
    print("    - performance_algorithms.csv")
    
    # Summary
    print("\n6. Performance Summary")
    print("-" * 70)
    
    best_alg = alg_results.loc[alg_results['avg_time'].idxmin()]
    print(f"  Best algorithm combination: {best_alg['social_algorithm']} + {best_alg['liquidity_algorithm']}")
    print(f"  Best time: {best_alg['avg_time']:.3f}s")
    
    # Estimate maximum network size
    best_time_per_node = size_results[
        (size_results['social_algorithm'] == best_alg['social_algorithm']) &
        (size_results['liquidity_algorithm'] == best_alg['liquidity_algorithm'])
    ]['time_per_node'].mean()
    
    max_nodes_1min = int(60 / best_time_per_node) if best_time_per_node > 0 else 0
    max_nodes_10min = int(600 / best_time_per_node) if best_time_per_node > 0 else 0
    
    print(f"  Estimated capacity (1 minute): {max_nodes_1min:,} nodes")
    print(f"  Estimated capacity (10 minutes): {max_nodes_10min:,} nodes")
    
    print(f"  Cache speedup: {cache_results['speedup']:.1f}x")
    
    print("\n" + "=" * 70)
    print("PERFORMANCE TESTING COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()