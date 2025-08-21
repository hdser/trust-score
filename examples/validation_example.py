#!/usr/bin/env python3
"""
Validation example demonstrating the theoretical framework implementation.

This example validates that:
1. Balance-aware algorithms correlate better with payment capacity
2. Payment capacity formula works correctly
3. Trust vs balance bottlenecks are properly identified
4. Algorithm comparison validates theoretical predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from trust_network import TrustNetwork
from trust_network.config.settings import NetworkConfig
from trust_network.analysis.analyzer import TrustNetworkAnalyzer
from trust_network.core.payment_capacity import PaymentCapacityCalculator
import json


def create_validation_network() -> TrustNetwork:
    """Create a carefully designed network for validation."""
    
    config = NetworkConfig()
    network = TrustNetwork(config)
    
    # Create nodes with specific patterns
    nodes = {
        # High trust, high balance nodes
        "Alice": {"type": "high_trust_high_balance"},
        "Bob": {"type": "high_trust_high_balance"},
        
        # High trust, low balance nodes  
        "Charlie": {"type": "high_trust_low_balance"},
        "David": {"type": "high_trust_low_balance"},
        
        # Low trust, high balance nodes
        "Eve": {"type": "low_trust_high_balance"},
        "Frank": {"type": "low_trust_high_balance"},
        
        # Converter nodes
        "ConvX": {"type": "converter"},
        "ConvY": {"type": "converter"},
        
        # Isolated nodes
        "Grace": {"type": "isolated"},
        "Henry": {"type": "isolated"}
    }
    
    # Add nodes
    for name, props in nodes.items():
        is_converter = props["type"] == "converter"
        network.add_node(name, is_converter=is_converter, metadata=props)
    
    # Trust relationships designed to test theory
    trust_edges = [
        # High trust cluster
        ("Alice", "Bob", 1.0),
        ("Bob", "Alice", 1.0),
        ("Alice", "Charlie", 0.9),
        ("Charlie", "Alice", 0.9),
        ("Bob", "David", 0.8),
        ("David", "Bob", 0.8),
        
        # Medium trust connections
        ("Charlie", "David", 0.7),
        ("David", "Charlie", 0.7),
        
        # Low trust connections
        ("Eve", "Frank", 0.3),
        ("Frank", "Eve", 0.2),
        
        # Converter connections
        ("ConvX", "Alice", 1.0),
        ("ConvX", "Bob", 0.9),
        ("ConvY", "Charlie", 1.0),
        ("ConvY", "David", 0.8),
        ("Alice", "ConvX", 0.9),
        ("Bob", "ConvX", 0.8),
        
        # Sparse connections to isolated nodes
        ("Grace", "Alice", 0.1),  # Very low trust
        ("Henry", "Bob", 0.1),
    ]
    
    for source, target, weight in trust_edges:
        network.add_edge(source, target, weight)
    
    # Balance distribution designed to test theory
    balance_scenarios = {
        # High balance nodes have tokens they can use and forward
        "Alice": {"Alice": 5000, "Bob": 1000, "Charlie": 500},
        "Bob": {"Bob": 4000, "Alice": 800, "David": 600},
        
        # Low balance nodes have minimal tokens for forwarding
        "Charlie": {"Charlie": 3000, "Alice": 50, "Bob": 30},  # Low foreign balances
        "David": {"David": 2500, "Bob": 40, "Charlie": 20},
        
        # High balance but low trust
        "Eve": {"Eve": 6000, "Frank": 2000},  # Has tokens but limited trust network
        "Frank": {"Frank": 5500, "Eve": 1500},
        
        # Converters typically have diverse token holdings
        "ConvX": {"Alice": 200, "Bob": 150, "Charlie": 100},
        "ConvY": {"Charlie": 180, "David": 120, "Alice": 80},
        
        # Isolated nodes with self-tokens but no network
        "Grace": {"Grace": 1000},  # Only self-tokens
        "Henry": {"Henry": 800}
    }
    
    for holder, tokens in balance_scenarios.items():
        for token, amount in tokens.items():
            network.set_balance(holder, token, amount)
    
    # Conversion rates
    rates = [
        ("ConvX", "Alice", 1.0),
        ("ConvX", "Bob", 0.95),
        ("ConvX", "Charlie", 0.90),
        ("ConvY", "Charlie", 1.0),
        ("ConvY", "David", 0.95),
        ("ConvY", "Alice", 0.85)
    ]
    
    for converter, token, rate in rates:
        network.set_conversion_rate(converter, token, rate)
    
    return network


def validate_payment_capacity_theory(network: TrustNetwork) -> dict:
    """Validate that payment capacity formula works as expected."""
    
    print("\n" + "="*80)
    print("PAYMENT CAPACITY THEORY VALIDATION")
    print("="*80)
    
    validation_results = {
        "theoretical_predictions": [],
        "actual_results": [],
        "validation_passed": True,
        "failures": []
    }
    
    # Test cases designed to validate theory
    test_cases = [
        {
            "name": "High Trust + High Balance",
            "source": "Alice",
            "target": "Bob", 
            "token": "Alice",
            "expected": "HIGH",  # Should work well
            "reason": "Both trust and balance support payment"
        },
        {
            "name": "High Trust + Low Balance",
            "source": "Charlie",
            "target": "David",
            "token": "Charlie", 
            "expected": "MEDIUM",  # Limited by balance
            "reason": "Trust exists but balance limits capacity"
        },
        {
            "name": "Low Trust + High Balance",
            "source": "Eve",
            "target": "Frank",
            "token": "Eve",
            "expected": "LOW",  # Limited by trust
            "reason": "Balance exists but trust limits acceptance"
        },
        {
            "name": "No Trust Path",
            "source": "Grace",
            "target": "Henry",
            "token": "Grace",
            "expected": "ZERO",  # Should fail
            "reason": "No viable trust path exists"
        },
        {
            "name": "Converter Reachability",
            "source": "Alice",
            "target": "ConvX",
            "token": "Alice",
            "expected": "HIGH",  # Should work well
            "reason": "Strong path to converter"
        }
    ]
    
    print(f"{'Test Case':<25} {'Expected':<10} {'Actual':<10} {'Capacity':<12} {'Status':<10}")
    print("-" * 80)
    
    for test in test_cases:
        capacity = network.estimate_payment_capacity(
            test["source"], test["target"], test["token"]
        )
        
        # Classify actual result
        if capacity == 0:
            actual_category = "ZERO"
        elif capacity < 100:
            actual_category = "LOW"
        elif capacity < 1000:
            actual_category = "MEDIUM"
        else:
            actual_category = "HIGH"
        
        # Check if prediction matches
        matches = (test["expected"] == actual_category)
        status = "✓ PASS" if matches else "✗ FAIL"
        
        print(f"{test['name']:<25} {test['expected']:<10} {actual_category:<10} {capacity:<12.1f} {status:<10}")
        
        validation_results["theoretical_predictions"].append(test["expected"])
        validation_results["actual_results"].append(actual_category)
        
        if not matches:
            validation_results["validation_passed"] = False
            validation_results["failures"].append({
                "test": test["name"],
                "expected": test["expected"],
                "actual": actual_category,
                "capacity": capacity,
                "reason": test["reason"]
            })
    
    return validation_results


def validate_algorithm_theory(network: TrustNetwork) -> dict:
    """Validate that balance-aware algorithms outperform trust-only algorithms."""
    
    print("\n" + "="*80)
    print("ALGORITHM THEORY VALIDATION")
    print("="*80)
    
    analyzer = TrustNetworkAnalyzer(network)
    comparison_results = analyzer.compare_algorithms_vs_payment_capacity()
    
    # Extract key metrics
    correlations = comparison_results["correlations"]
    classification = comparison_results["classification"]
    
    # Check if balance-aware algorithms outperform trust-only
    validation_results = {
        "balance_aware_wins": 0,
        "trust_only_wins": 0,
        "ties": 0,
        "overall_validation": False,
        "detailed_results": {}
    }
    
    print(f"{'Algorithm':<15} {'Type':<15} {'Avg Correlation':<15} {'Best Capacity':<15}")
    print("-" * 70)
    
    # Calculate average correlations for each algorithm
    algorithm_performance = {}
    
    for alg_name, alg_corrs in correlations.items():
        total_corr = 0
        count = 0
        
        for cap_type, corr_data in alg_corrs.items():
            if cap_type == "outgoing_capacity":  # Focus on most relevant capacity
                correlation = abs(corr_data["spearman_correlation"])
                total_corr += correlation
                count += 1
        
        avg_correlation = total_corr / count if count > 0 else 0
        algorithm_performance[alg_name] = avg_correlation
        
        # Determine algorithm type
        if alg_name in classification["balance_aware"]["algorithms"]:
            alg_type = "Balance-Aware"
        elif alg_name in classification["trust_only"]["algorithms"]:
            alg_type = "Trust-Only"
        else:
            alg_type = "Other"
        
        # Find best capacity correlation
        best_corr = max([abs(corr_data["spearman_correlation"]) for corr_data in alg_corrs.values()])
        
        print(f"{alg_name:<15} {alg_type:<15} {avg_correlation:<15.3f} {best_corr:<15.3f}")
        
        validation_results["detailed_results"][alg_name] = {
            "type": alg_type,
            "avg_correlation": avg_correlation,
            "best_correlation": best_corr
        }
    
    # Compare balance-aware vs trust-only
    balance_aware_scores = [
        score for alg, score in algorithm_performance.items()
        if alg in classification["balance_aware"]["algorithms"]
    ]
    
    trust_only_scores = [
        score for alg, score in algorithm_performance.items()
        if alg in classification["trust_only"]["algorithms"]
    ]
    
    if balance_aware_scores and trust_only_scores:
        avg_balance_aware = np.mean(balance_aware_scores)
        avg_trust_only = np.mean(trust_only_scores)
        
        print(f"\nAverage Performance:")
        print(f"Balance-Aware Algorithms: {avg_balance_aware:.3f}")
        print(f"Trust-Only Algorithms:    {avg_trust_only:.3f}")
        print(f"Advantage:                {avg_balance_aware - avg_trust_only:+.3f}")
        
        validation_results["overall_validation"] = avg_balance_aware > avg_trust_only
        
        if avg_balance_aware > avg_trust_only:
            print("✓ THEORY VALIDATED: Balance-aware algorithms outperform trust-only")
        else:
            print("✗ THEORY NOT VALIDATED: Trust-only algorithms perform better")
    
    return validation_results


def validate_bottleneck_theory(network: TrustNetwork) -> dict:
    """Validate bottleneck identification theory."""
    
    print("\n" + "="*80)
    print("BOTTLENECK THEORY VALIDATION")
    print("="*80)
    
    analyzer = TrustNetworkAnalyzer(network)
    bottleneck_analysis = analyzer.analyze_payment_bottlenecks()
    
    validation_results = {
        "trust_bottlenecks_found": [],
        "balance_bottlenecks_found": [],
        "expected_bottlenecks": [],
        "validation_passed": True
    }
    
    # Expected bottlenecks based on network design
    expected_bottlenecks = [
        {
            "type": "trust",
            "description": "Eve and Frank should have trust bottlenecks",
            "nodes": ["Eve", "Frank"]
        },
        {
            "type": "balance", 
            "description": "Charlie and David should have balance bottlenecks",
            "nodes": ["Charlie", "David"]
        },
        {
            "type": "isolation",
            "description": "Grace and Henry should be isolated",
            "nodes": ["Grace", "Henry"]
        }
    ]
    
    # Analyze trust-balance correlation
    correlation_analysis = analyzer.analyze_balance_vs_trust_correlation()
    
    print("Trust-Balance Correlation Analysis:")
    print(f"Correlation: {correlation_analysis['trust_balance_correlation']['spearman_correlation']:.3f}")
    print(f"Interpretation: {correlation_analysis['trust_balance_correlation']['interpretation']}")
    
    # Check payment enabling analysis
    enabling_analysis = correlation_analysis["payment_enabling_analysis"]
    
    print(f"\nPayment Enabling Analysis:")
    print(f"Trust-only blocks: {enabling_analysis['percentages']['trust_only_blocks']:.1f}%")
    print(f"Balance-only blocks: {enabling_analysis['percentages']['balance_only_blocks']:.1f}%")
    print(f"Both enable: {enabling_analysis['percentages']['both_enable']:.1f}%")
    
    # Validate that we can identify different types of bottlenecks
    trust_bottleneck_ratio = enabling_analysis["trust_bottleneck_ratio"]
    balance_bottleneck_ratio = enabling_analysis["balance_bottleneck_ratio"]
    
    if trust_bottleneck_ratio > 0.1:  # At least 10% trust bottlenecks
        print("✓ Trust bottlenecks successfully identified")
        validation_results["trust_bottlenecks_found"] = True
    else:
        print("✗ Trust bottlenecks not properly identified")
        validation_results["validation_passed"] = False
    
    if balance_bottleneck_ratio > 0.1:  # At least 10% balance bottlenecks
        print("✓ Balance bottlenecks successfully identified")
        validation_results["balance_bottlenecks_found"] = True
    else:
        print("✗ Balance bottlenecks not properly identified")
        validation_results["validation_passed"] = False
    
    return validation_results


def run_comprehensive_validation():
    """Run comprehensive validation of the theoretical framework."""
    
    print("="*80)
    print("TRUST NETWORK THEORETICAL FRAMEWORK VALIDATION")
    print("="*80)
    print("This validation tests the implementation against theoretical predictions:")
    print("1. Payment capacity formula correctness")
    print("2. Balance-aware algorithm superiority")
    print("3. Bottleneck identification accuracy")
    print("4. Trust vs balance interaction analysis")
    
    # Create validation network
    print("\nCreating validation network with designed bottlenecks...")
    network = create_validation_network()
    
    print(f"Network created: {len(network.nodes)} nodes, {len(network.edges)} edges")
    print(f"Converters: {len(network.converters)}")
    print(f"Total token supply: {sum(sum(tokens.values()) for tokens in network.balances.values())}")
    
    # Run all validations
    validation_results = {
        "payment_capacity": validate_payment_capacity_theory(network),
        "algorithm_comparison": validate_algorithm_theory(network),
        "bottleneck_identification": validate_bottleneck_theory(network)
    }
    
    # Overall validation summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    
    # Payment capacity validation
    payment_passed = validation_results["payment_capacity"]["validation_passed"]
    print(f"Payment Capacity Theory: {'✓ PASSED' if payment_passed else '✗ FAILED'}")
    if not payment_passed:
        print(f"  Failures: {len(validation_results['payment_capacity']['failures'])}")
        all_passed = False
    
    # Algorithm comparison validation
    algo_passed = validation_results["algorithm_comparison"]["overall_validation"]
    print(f"Algorithm Superiority Theory: {'✓ PASSED' if algo_passed else '✗ FAILED'}")
    if not algo_passed:
        all_passed = False
    
    # Bottleneck identification validation
    bottleneck_passed = validation_results["bottleneck_identification"]["validation_passed"]
    print(f"Bottleneck Identification Theory: {'✓ PASSED' if bottleneck_passed else '✗ FAILED'}")
    if not bottleneck_passed:
        all_passed = False
    
    print(f"\nOVERALL VALIDATION: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 THEORETICAL FRAMEWORK SUCCESSFULLY VALIDATED!")
        print("The implementation correctly captures the theoretical principles:")
        print("- Payment capacity formula works as expected")
        print("- Balance-aware algorithms outperform trust-only algorithms")
        print("- Different types of bottlenecks are properly identified")
        print("- Trust and balance interactions are correctly modeled")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED")
        print("Some aspects of the implementation may need adjustment.")
    
    # Generate detailed validation report
    print(f"\nGenerating detailed validation report...")
    
    analyzer = TrustNetworkAnalyzer(network)
    comprehensive_report = analyzer.generate_comprehensive_report()
    
    # Save validation results
    validation_report = {
        "validation_results": validation_results,
        "comprehensive_analysis": comprehensive_report,
        "network_data": network.to_dict()
    }
    
    with open("validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print("Validation report saved to: validation_report.json")
    
    # Network health assessment
    health_score = comprehensive_report["network_health"]["overall_score"]
    health_rating = comprehensive_report["network_health"]["health_rating"]
    
    print(f"\nValidation Network Health: {health_rating} ({health_score:.2f})")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)