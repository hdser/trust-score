#!/usr/bin/env python3
"""Basic usage example for trust network."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trust_network import TrustNetwork
from trust_network.data import DataLoader
from trust_network.config.settings import load_config, setup_logging

def main():
    """Run basic trust network example."""
    
    # Load configuration
    config = load_config()
    setup_logging(config.logging)
    
    print("=" * 60)
    print("TRUST NETWORK - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading network data...")
    data = DataLoader.load("data/sample", format="csv")
    
    print(f"   - Loaded {len(data['nodes'])} nodes")
    print(f"   - Loaded {len(data['edges'])} edges")
    print(f"   - Loaded {len(data['balances'])} balances")
    print(f"   - Loaded {len(data['rates'])} rates")
    
    # Create network
    print("\n2. Creating trust network...")
    network = TrustNetwork(config)
    network.load_data(data)
    
    # Compute trust scores
    print("\n3. Computing trust scores...")
    results = network.compute_trust_scores()
    
    print(f"   - Computation time: {results.computation_time:.3f} seconds")
    print(f"   - Algorithm used: {results.algorithm_used}")
    
    # Display rankings
    print("\n4. Trust Score Rankings:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Node':<10} {'Composite':<12} {'Social':<12} {'Liquidity':<12}")
    print("-" * 60)
    
    ranking = network.get_ranking(top_n=10)
    for i, (node, score) in enumerate(ranking, 1):
        node_score = results.scores[node]
        print(f"{i:<6} {node:<10} {node_score.composite_score:<12.4f} "
              f"{node_score.social_score:<12.4f} {node_score.liquidity_score:<12.4f}")
    
    # Find some paths
    print("\n5. Sample Trust Paths:")
    print("-" * 60)
    
    test_paths = [("Alice", "ConvX"), ("Henry", "ConvY"), ("David", "Eve")]
    
    for source, target in test_paths:
        path = network.find_trust_path(source, target)
        if path and path.is_valid:
            path_str = " -> ".join(path.nodes)
            print(f"   {source} to {target}:")
            print(f"     Path: {path_str}")
            print(f"     Trust: {path.trust_value:.4f}")
            print(f"     Hops: {path.hop_count}")
        else:
            print(f"   {source} to {target}: No viable path")
    
    # Network statistics
    print("\n6. Network Statistics:")
    print("-" * 60)
    stats = results.network_stats
    print(f"   - Nodes: {stats.num_nodes}")
    print(f"   - Edges: {stats.num_edges}")
    print(f"   - Converters: {stats.num_converters}")
    print(f"   - Density: {stats.density:.4f}")
    print(f"   - Clustering: {stats.avg_clustering:.4f}")
    print(f"   - Diameter: {stats.diameter}")
    
    print("\n" + "=" * 60)
    print("BASIC EXAMPLE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()