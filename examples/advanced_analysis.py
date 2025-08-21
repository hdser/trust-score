#!/usr/bin/env python3
"""Advanced analysis example for trust network."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trust_network import TrustNetwork, TrustNetworkAnalyzer
from trust_network.data import DataLoader
from trust_network.config.settings import load_config, setup_logging
import json

def main():
    """Run advanced trust network analysis."""
    
    # Load configuration
    config = load_config()
    setup_logging(config.logging)
    
    print("=" * 70)
    print("TRUST NETWORK - ADVANCED ANALYSIS EXAMPLE")
    print("=" * 70)
    
    # Load data and create network
    print("\n1. Setting up network...")
    data = DataLoader.load("data/sample", format="csv")
    network = TrustNetwork(config)
    network.load_data(data)
    
    # Create analyzer
    analyzer = TrustNetworkAnalyzer(network)
    
    # Compute trust scores
    print("\n2. Computing trust scores...")
    results = network.compute_trust_scores()
    
    # Centrality analysis
    print("\n3. Centrality Analysis:")
    print("-" * 70)
    centrality_metrics = analyzer.centrality_analyzer.compute_all_centralities()
    
    print("   Top 5 nodes by different centrality measures:")
    for metric_name, metric_values in centrality_metrics.items():
        print(f"\n   {metric_name.title()} Centrality:")
        sorted_nodes = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        for i, (node, value) in enumerate(sorted_nodes[:5], 1):
            print(f"     {i}. {node}: {value:.4f}")
    
    # Path analysis
    print("\n4. Path Analysis:")
    print("-" * 70)
    path_metrics = analyzer.path_analyzer.analyze_path_distribution(sample_size=50)
    
    if "path_length_stats" in path_metrics:
        print(f"   Connectivity rate: {path_metrics['connectivity_rate']:.2%}")
        print(f"   Average path length: {path_metrics['path_length_stats']['mean']:.2f}")
        print(f"   Average trust value: {path_metrics['trust_value_stats']['mean']:.4f}")
    
    # Critical paths
    critical_paths = analyzer.path_analyzer.find_critical_paths(top_k=5)
    print(f"\n   Top 5 critical paths:")
    for i, path_info in enumerate(critical_paths, 1):
        print(f"     {i}. {path_info['source']} -> {path_info['target']}")
        print(f"        Trust: {path_info['trust_value']:.4f}, Hops: {path_info['hop_count']}")
    
    # Community detection
    print("\n5. Community Structure:")
    print("-" * 70)
    communities = analyzer.community_analyzer.detect_communities()
    
    print(f"   Number of communities: {communities.num_communities}")
    print(f"   Modularity: {communities.modularity:.4f}")
    
    # Group nodes by community
    community_groups = {}
    for node, comm_id in communities.communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(node)
    
    print("   Community assignments:")
    for comm_id, nodes in community_groups.items():
        print(f"     Community {comm_id}: {', '.join(sorted(nodes))}")
    
    # Flow analysis
    print("\n6. Token Flow Analysis:")
    print("-" * 70)
    flow_analysis = analyzer.analyze_token_flows()
    
    print(f"   Total token supply: {flow_analysis.total_supply:.0f}")
    print(f"   Gini coefficient: {flow_analysis.gini_coefficient:.4f}")
    print(f"   Average velocity: {flow_analysis.avg_velocity:.2f}")
    
    if flow_analysis.token_velocities:
        print("   Top 5 tokens by velocity:")
        sorted_velocities = sorted(
           flow_analysis.token_velocities.items(), 
           key=lambda x: x[1], reverse=True
        )
        for i, (token, velocity) in enumerate(sorted_velocities[:5], 1):
           print(f"     {i}. {token}: {velocity:.2f}")
   
    # Sybil resistance analysis
    print("\n7. Sybil Resistance Analysis:")
    print("-" * 70)
    
    # Simulate Sybil attack with some nodes
    sybil_nodes = {"Eve", "Grace", "Henry"}
    sybil_analysis = analyzer.analyze_sybil_resistance(sybil_nodes)
    
    print(f"   Sybil nodes: {', '.join(sybil_nodes)}")
    print(f"   Total Sybil score: {sybil_analysis.total_sybil_score:.4f}")
    print(f"   Theoretical bound: {sybil_analysis.theoretical_bound:.4f}")
    print(f"   Actual/Bound ratio: {sybil_analysis.actual_vs_bound_ratio:.2f}")
    print(f"   Resistance effective: {sybil_analysis.resistance_effective}")
    
    # Network resilience
    print("\n8. Network Resilience:")
    print("-" * 70)
    resilience = analyzer.compute_network_resilience()
    
    print(f"   Edge connectivity: {resilience['edge_connectivity']}")
    print(f"   Node connectivity: {resilience['node_connectivity']}")
    print(f"   Clustering resilience: {resilience['clustering_resilience']:.4f}")
    print(f"   Redundancy ratio: {resilience['redundancy_ratio']:.4f}")
    
    if "converter_dependency" in resilience:
        conv_dep = resilience["converter_dependency"]
        print(f"   Converter dependency: {conv_dep['converter_score_ratio']:.4f}")
        print(f"   Converter centralization: {conv_dep['converter_centralization']:.4f}")
    
    # Payment capacity analysis
    print("\n9. Payment Capacity Analysis:")
    print("-" * 70)
    
    # Analyze capacity for different tokens
    tokens = ["T_A", "T_B", "T_C"]
    for token in tokens:
        capacity_analysis = analyzer.path_analyzer.analyze_payment_capacity(
            token, sample_size=30
        )
        
        if "capacity_stats" in capacity_analysis:
            stats = capacity_analysis["capacity_stats"]
            print(f"   {token}:")
            print(f"     Feasible payments: {capacity_analysis['feasible_payments']}/30")
            print(f"     Average capacity: {stats['mean']:.2f}")
            print(f"     Max capacity: {stats['max']:.2f}")
    
    # Generate comprehensive report
    print("\n10. Generating Comprehensive Report...")
    print("-" * 70)
    
    report = analyzer.generate_report()
    
    # Save report
    with open("analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("   Report saved to: analysis_report.json")
    
    # Display key insights
    print("\n11. Key Insights:")
    print("-" * 70)
    
    # Top node
    top_node = results.get_ranking(top_n=1)[0]
    print(f"   • Most trusted node: {top_node[0]} (score: {top_node[1]:.4f})")
    
    # Network health
    connectivity_rate = path_metrics.get('connectivity_rate', 0)
    if connectivity_rate > 0.8:
        health = "Excellent"
    elif connectivity_rate > 0.6:
        health = "Good"
    elif connectivity_rate > 0.4:
        health = "Fair"
    else:
        health = "Poor"
    
    print(f"   • Network connectivity health: {health} ({connectivity_rate:.1%})")
    
    # Community insights
    largest_community = max(communities.community_sizes.values()) if communities.community_sizes else 0
    community_dominance = largest_community / len(network.nodes) if network.nodes else 0
    
    if community_dominance > 0.7:
        structure = "Highly centralized"
    elif community_dominance > 0.4:
        structure = "Moderately centralized"
    else:
        structure = "Well distributed"
    
    print(f"   • Community structure: {structure}")
    
    # Token distribution
    gini = flow_analysis.gini_coefficient
    if gini > 0.7:
        distribution = "Highly concentrated"
    elif gini > 0.4:
        distribution = "Moderately concentrated"
    else:
        distribution = "Well distributed"
    
    print(f"   • Token distribution: {distribution} (Gini: {gini:.3f})")
    
    # Security assessment
    if sybil_analysis.resistance_effective:
        security = "Strong"
    elif sybil_analysis.actual_vs_bound_ratio < 3:
        security = "Moderate"
    else:
        security = "Weak"
    
    print(f"   • Sybil resistance: {security}")
    
    print("\n" + "=" * 70)
    print("ADVANCED ANALYSIS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
   main()