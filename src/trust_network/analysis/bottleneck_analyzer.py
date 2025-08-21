"""
Bottleneck analysis for payment networks implementing theoretical framework insights.

This module provides comprehensive analysis of payment bottlenecks, identifying
where trust exists but balances are insufficient, or where trust gaps prevent payments.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class BottleneckAnalyzer:
    """Comprehensive bottleneck analysis for trust networks."""
    
    def __init__(self, network):
        """Initialize with a TrustNetwork instance."""
        self.network = network
        
    def analyze_all_bottlenecks(self) -> Dict[str, Any]:
        """
        Comprehensive bottleneck analysis covering all aspects.
        
        Returns:
            Dictionary with detailed bottleneck analysis results
        """
        # Build matrices if needed
        self.network._build_matrices()
        
        analysis = {
            'payment_bottlenecks': self._analyze_payment_bottlenecks(),
            'trust_bottlenecks': self._analyze_trust_bottlenecks(), 
            'balance_bottlenecks': self._analyze_balance_bottlenecks(),
            'structural_bottlenecks': self._analyze_structural_bottlenecks(),
            'converter_bottlenecks': self._analyze_converter_bottlenecks(),
            'token_distribution_issues': self._analyze_token_distribution(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_payment_bottlenecks(self) -> Dict[str, Any]:
        """Analyze bottlenecks in actual payment flows."""
        from ..core.payment_capacity import PaymentCapacityCalculator
        
        calculator = PaymentCapacityCalculator(
            self.network._W, self.network._B, 
            self.network.config.trust_network.parameters.tau
        )
        
        # Get basic payment bottleneck analysis
        basic_analysis = calculator.analyze_payment_bottlenecks()
        
        # Add more detailed analysis
        payment_success_rates = self._compute_payment_success_rates()
        capacity_distribution = self._analyze_capacity_distribution()
        
        return {
            'basic_bottlenecks': basic_analysis,
            'success_rates': payment_success_rates,
            'capacity_distribution': capacity_distribution,
            'network_liquidity_score': calculator.compute_network_liquidity_score()
        }
    
    def _analyze_trust_bottlenecks(self) -> Dict[str, Any]:
        """Analyze bottlenecks caused by insufficient trust relationships."""
        n = len(self.network.nodes)
        tau = self.network.config.trust_network.parameters.tau
        
        trust_issues = {
            'isolated_nodes': [],
            'low_trust_nodes': [],
            'trust_asymmetries': [],
            'missing_critical_trust': [],
            'trust_concentration': {}
        }
        
        # Find isolated nodes
        for node, info in self.network.nodes.items():
            idx = info.index
            
            incoming_trust = (self.network._W[:, idx] >= tau).sum()
            outgoing_trust = (self.network._W[idx, :] >= tau).sum()
            
            if incoming_trust == 0 and outgoing_trust == 0:
                trust_issues['isolated_nodes'].append({
                    'node': node,
                    'reason': 'No trust connections'
                })
            elif incoming_trust == 0:
                trust_issues['isolated_nodes'].append({
                    'node': node,
                    'reason': 'No incoming trust'
                })
            elif outgoing_trust == 0:
                trust_issues['isolated_nodes'].append({
                    'node': node,
                    'reason': 'No outgoing trust'
                })
        
        # Find nodes with low trust scores
        W_dense = self.network._W.toarray()
        
        for node, info in self.network.nodes.items():
            idx = info.index
            
            avg_incoming = W_dense[:, idx].mean()
            avg_outgoing = W_dense[idx, :].mean()
            
            if avg_incoming < tau / 2:
                trust_issues['low_trust_nodes'].append({
                    'node': node,
                    'avg_incoming_trust': float(avg_incoming),
                    'type': 'low_incoming'
                })
            
            if avg_outgoing < tau / 2:
                trust_issues['low_trust_nodes'].append({
                    'node': node,
                    'avg_outgoing_trust': float(avg_outgoing),
                    'type': 'low_outgoing'
                })
        
        # Find trust asymmetries
        for i, node_i in enumerate(self.network.nodes):
            for j, node_j in enumerate(self.network.nodes):
                if i < j:  # Avoid duplicate pairs
                    trust_ij = W_dense[i, j]
                    trust_ji = W_dense[j, i]
                    
                    if abs(trust_ij - trust_ji) > 0.5 and min(trust_ij, trust_ji) > 0:
                        trust_issues['trust_asymmetries'].append({
                            'node1': node_i,
                            'node2': node_j,
                            'trust_1_to_2': float(trust_ij),
                            'trust_2_to_1': float(trust_ji),
                            'asymmetry': float(abs(trust_ij - trust_ji))
                        })
        
        # Analyze trust concentration
        trust_concentrations = []
        for node, info in self.network.nodes.items():
            idx = info.index
            
            # How much of network's total trust involves this node?
            total_network_trust = W_dense.sum()
            node_trust = W_dense[:, idx].sum() + W_dense[idx, :].sum()
            
            concentration = node_trust / total_network_trust if total_network_trust > 0 else 0
            
            trust_concentrations.append((node, concentration))
        
        # Sort by concentration
        trust_concentrations.sort(key=lambda x: x[1], reverse=True)
        
        trust_issues['trust_concentration'] = {
            'top_nodes': trust_concentrations[:5],
            'gini_coefficient': self._compute_gini_coefficient([x[1] for x in trust_concentrations])
        }
        
        return trust_issues
    
    def _analyze_balance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze bottlenecks caused by insufficient token balances."""
        B_dense = self.network._B.toarray()
        n = len(self.network.nodes)
        
        balance_issues = {
            'nodes_without_self_tokens': [],
            'low_balance_nodes': [],
            'foreign_token_gaps': [],
            'balance_concentration': {},
            'token_starvation': []
        }
        
        # Nodes without self-tokens
        for node, info in self.network.nodes.items():
            idx = info.index
            
            self_balance = B_dense[idx, idx]
            if self_balance <= 0:
                balance_issues['nodes_without_self_tokens'].append({
                    'node': node,
                    'self_balance': float(self_balance)
                })
        
        # Nodes with low balances for forwarding
        for node, info in self.network.nodes.items():
            idx = info.index
            
            # Check balances of foreign tokens
            foreign_balances = []
            for token_idx in range(n):
                if token_idx != idx:
                    balance = B_dense[idx, token_idx]
                    if balance > 0:
                        foreign_balances.append(balance)
            
            if foreign_balances:
                avg_foreign = np.mean(foreign_balances)
                if avg_foreign < 10:  # Threshold for "low"
                    balance_issues['low_balance_nodes'].append({
                        'node': node,
                        'avg_foreign_balance': float(avg_foreign),
                        'num_foreign_tokens': len(foreign_balances)
                    })
        
        # Foreign token gaps (should have token but doesn't)
        for node, info in self.network.nodes.items():
            idx = info.index
            
            # Find tokens this node should have based on trust relationships
            should_have_tokens = []
            
            for token_idx in range(n):
                if token_idx != idx:
                    # If node trusts token issuer, should probably have some tokens
                    if self.network._W[idx, token_idx] >= self.network.config.trust_network.parameters.tau:
                        balance = B_dense[idx, token_idx]
                        if balance == 0:
                            token_node = self.network.idx_to_node[token_idx]
                            should_have_tokens.append(token_node)
            
            if should_have_tokens:
                balance_issues['foreign_token_gaps'].append({
                    'node': node,
                    'missing_tokens': should_have_tokens,
                    'count': len(should_have_tokens)
                })
        
        # Balance concentration analysis
        all_balances = B_dense.flatten()
        nonzero_balances = all_balances[all_balances > 0]
        
        if len(nonzero_balances) > 0:
            balance_issues['balance_concentration'] = {
                'gini_coefficient': self._compute_gini_coefficient(nonzero_balances),
                'top_1_percent': float(np.percentile(nonzero_balances, 99)),
                'median': float(np.median(nonzero_balances)),
                'mean': float(np.mean(nonzero_balances)),
                'std': float(np.std(nonzero_balances))
            }
        
        # Token starvation (tokens with very low total supply)
        token_supplies = B_dense.sum(axis=0)
        avg_supply = token_supplies.mean()
        
        for token_idx, supply in enumerate(token_supplies):
            if supply < avg_supply * 0.1 and supply > 0:  # Less than 10% of average
                token_node = self.network.idx_to_node[token_idx]
                balance_issues['token_starvation'].append({
                    'token': token_node,
                    'total_supply': float(supply),
                    'avg_supply': float(avg_supply),
                    'ratio': float(supply / avg_supply) if avg_supply > 0 else 0
                })
        
        return balance_issues
    
    def _analyze_structural_bottlenecks(self) -> Dict[str, Any]:
        """Analyze structural bottlenecks in network topology."""
        from scipy.sparse.csgraph import connected_components
        
        structural_issues = {
            'connectivity': {},
            'critical_nodes': [],
            'bridge_nodes': [],
            'clustering_issues': {}
        }
        
        # Connectivity analysis
        num_components, labels = connected_components(
            self.network._W, directed=True, connection='weak'
        )
        
        component_sizes = np.bincount(labels)
        largest_component = component_sizes.max()
        
        structural_issues['connectivity'] = {
            'num_components': int(num_components),
            'largest_component_size': int(largest_component),
            'connectivity_ratio': float(largest_component / len(self.network.nodes)),
            'is_connected': num_components == 1
        }
        
        # Find critical nodes (whose removal disconnects network)
        original_components = num_components
        
        for node, info in self.network.nodes.items():
            idx = info.index
            
            # Temporarily remove node by setting its row/column to 0
            W_temp = self.network._W.copy().tolil()
            W_temp[idx, :] = 0
            W_temp[:, idx] = 0
            W_temp = W_temp.tocsr()
            
            # Check new connectivity
            new_components, _ = connected_components(W_temp, directed=True, connection='weak')
            
            if new_components > original_components:
                structural_issues['critical_nodes'].append({
                    'node': node,
                    'components_after_removal': int(new_components),
                    'components_increase': int(new_components - original_components)
                })
        
        # Clustering analysis
        clustering_coeffs = self._compute_clustering_coefficients()
        
        avg_clustering = np.mean(list(clustering_coeffs.values()))
        low_clustering_nodes = [
            node for node, coeff in clustering_coeffs.items() 
            if coeff < avg_clustering * 0.5
        ]
        
        structural_issues['clustering_issues'] = {
            'avg_clustering': float(avg_clustering),
            'low_clustering_nodes': low_clustering_nodes,
            'clustering_distribution': {
                'min': float(min(clustering_coeffs.values())),
                'max': float(max(clustering_coeffs.values())),
                'std': float(np.std(list(clustering_coeffs.values())))
            }
        }
        
        return structural_issues
    
    def _analyze_converter_bottlenecks(self) -> Dict[str, Any]:
        """Analyze bottlenecks related to converter nodes."""
        if not self.network.converters:
            return {'error': 'No converters in network'}
        
        converter_issues = {
            'converter_reachability': {},
            'conversion_capacity': {},
            'rate_analysis': {},
            'converter_concentration': {}
        }
        
        # Converter reachability
        reachability_stats = []
        
        for conv in self.network.converters:
            reachable_nodes = 0
            total_capacity_to_conv = 0
            
            for node in self.network.nodes:
                if node != conv:
                    capacity = self.network.estimate_payment_capacity(node, conv, node)
                    if capacity > 0:
                        reachable_nodes += 1
                        total_capacity_to_conv += capacity
            
            reachability_stats.append({
                'converter': conv,
                'reachable_nodes': reachable_nodes,
                'reachability_ratio': reachable_nodes / (len(self.network.nodes) - 1),
                'total_capacity': total_capacity_to_conv,
                'avg_capacity': total_capacity_to_conv / reachable_nodes if reachable_nodes > 0 else 0
            })
        
        converter_issues['converter_reachability'] = reachability_stats
        
        # Conversion capacity analysis
        conversion_capacities = []
        
        for conv in self.network.converters:
            if conv in self.network.rates:
                total_convertible = 0
                
                for token, rate in self.network.rates[conv].items():
                    # How much of this token can realistically reach the converter?
                    reachable_amount = 0
                    
                    for holder in self.network.nodes:
                        if holder != conv:
                            capacity = self.network.estimate_payment_capacity(holder, conv, token)
                            reachable_amount += capacity
                    
                    convertible_value = reachable_amount * rate
                    total_convertible += convertible_value
                
                conversion_capacities.append({
                    'converter': conv,
                    'total_convertible_value': total_convertible,
                    'num_tokens_supported': len(self.network.rates[conv])
                })
        
        converter_issues['conversion_capacity'] = conversion_capacities
        
        # Rate analysis
        if self.network.rates:
            all_rates = []
            for conv_rates in self.network.rates.values():
                all_rates.extend(conv_rates.values())
            
            if all_rates:
                converter_issues['rate_analysis'] = {
                    'avg_rate': float(np.mean(all_rates)),
                    'min_rate': float(np.min(all_rates)),
                    'max_rate': float(np.max(all_rates)),
                    'rate_std': float(np.std(all_rates)),
                    'rate_range': float(np.max(all_rates) - np.min(all_rates))
                }
        
        return converter_issues
    
    def _analyze_token_distribution(self) -> Dict[str, Any]:
        """Analyze token distribution patterns."""
        B_dense = self.network._B.toarray()
        n = len(self.network.nodes)
        
        distribution_issues = {
            'supply_imbalances': [],
            'holder_concentration': {},
            'circulation_analysis': {}
        }
        
        # Supply imbalances
        token_supplies = B_dense.sum(axis=0)
        avg_supply = token_supplies.mean()
        
        for token_idx, supply in enumerate(token_supplies):
            token_node = self.network.idx_to_node[token_idx]
            
            if supply > avg_supply * 3:  # Much higher than average
                distribution_issues['supply_imbalances'].append({
                    'token': token_node,
                    'supply': float(supply),
                    'avg_supply': float(avg_supply),
                    'ratio': float(supply / avg_supply) if avg_supply > 0 else 0,
                    'type': 'oversupply'
                })
            elif supply < avg_supply * 0.3 and supply > 0:  # Much lower than average
                distribution_issues['supply_imbalances'].append({
                    'token': token_node,
                    'supply': float(supply),
                    'avg_supply': float(avg_supply),
                    'ratio': float(supply / avg_supply) if avg_supply > 0 else 0,
                    'type': 'undersupply'
                })
        
        # Holder concentration for each token
        holder_concentrations = []
        
        for token_idx in range(n):
            token_node = self.network.idx_to_node[token_idx]
            token_balances = B_dense[:, token_idx]
            
            if token_balances.sum() > 0:
                # Calculate Gini coefficient for this token's distribution
                nonzero_balances = token_balances[token_balances > 0]
                gini = self._compute_gini_coefficient(nonzero_balances)
                
                # Find top holders
                sorted_balances = sorted(enumerate(token_balances), key=lambda x: x[1], reverse=True)
                top_holders = []
                
                for holder_idx, balance in sorted_balances[:3]:
                    if balance > 0:
                        holder_node = self.network.idx_to_node[holder_idx]
                        percentage = balance / token_balances.sum() * 100
                        top_holders.append({
                            'holder': holder_node,
                            'balance': float(balance),
                            'percentage': float(percentage)
                        })
                
                holder_concentrations.append({
                    'token': token_node,
                    'gini_coefficient': float(gini),
                    'num_holders': int((token_balances > 0).sum()),
                    'top_holders': top_holders
                })
        
        distribution_issues['holder_concentration'] = holder_concentrations
        
        # Circulation analysis
        circulation_stats = {
            'tokens_in_circulation': 0,
            'dormant_tokens': 0,
            'self_held_ratio': 0
        }
        
        total_tokens = B_dense.sum()
        self_held_tokens = np.diag(B_dense).sum()
        
        circulation_stats['self_held_ratio'] = float(self_held_tokens / total_tokens) if total_tokens > 0 else 0
        circulation_stats['tokens_in_circulation'] = float(total_tokens - self_held_tokens)
        
        distribution_issues['circulation_analysis'] = circulation_stats
        
        return distribution_issues
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on bottleneck analysis."""
        recommendations = []
        
        # Analyze current state
        payment_bottlenecks = self.network.analyze_network_payment_bottlenecks()
        
        # Recommendation 1: Address nodes without self-tokens
        if payment_bottlenecks['no_self_tokens']:
            recommendations.append({
                'category': 'Balance Issues',
                'priority': 'High',
                'issue': f"{len(payment_bottlenecks['no_self_tokens'])} nodes have no self-issued tokens",
                'recommendation': "Issue initial token supply to nodes without self-tokens to enable payment initiation",
                'affected_nodes': ', '.join([str(n) for n in payment_bottlenecks['no_self_tokens'][:5]])
            })
        
        # Recommendation 2: Address trust gaps
        trust_analysis = self._analyze_trust_bottlenecks()
        if trust_analysis['isolated_nodes']:
            recommendations.append({
                'category': 'Trust Network',
                'priority': 'High', 
                'issue': f"{len(trust_analysis['isolated_nodes'])} nodes are trust-isolated",
                'recommendation': "Establish trust relationships for isolated nodes to improve network connectivity",
                'affected_nodes': ', '.join([n['node'] for n in trust_analysis['isolated_nodes'][:5]])
            })
        
        # Recommendation 3: Balance distribution
        balance_analysis = self._analyze_balance_bottlenecks()
        if balance_analysis['foreign_token_gaps']:
            recommendations.append({
                'category': 'Token Distribution',
                'priority': 'Medium',
                'issue': f"{len(balance_analysis['foreign_token_gaps'])} nodes lack foreign tokens for forwarding",
                'recommendation': "Distribute foreign tokens to trusted nodes to improve payment forwarding capacity",
                'affected_nodes': ', '.join([n['node'] for n in balance_analysis['foreign_token_gaps'][:5]])
            })
        
        # Recommendation 4: Converter accessibility
        if self.network.converters:
            converter_analysis = self._analyze_converter_bottlenecks()
            low_reachability = [c for c in converter_analysis['converter_reachability'] 
                              if c['reachability_ratio'] < 0.5]
            
            if low_reachability:
                recommendations.append({
                    'category': 'Converter Access',
                    'priority': 'Medium',
                    'issue': f"{len(low_reachability)} converters have low reachability",
                    'recommendation': "Improve trust paths to converters or add more conversion capacity",
                    'affected_nodes': ', '.join([c['converter'] for c in low_reachability])
                })
        
        # Recommendation 5: Structural improvements
        structural_analysis = self._analyze_structural_bottlenecks()
        if not structural_analysis['connectivity']['is_connected']:
            recommendations.append({
                'category': 'Network Structure',
                'priority': 'High',
                'issue': f"Network has {structural_analysis['connectivity']['num_components']} disconnected components",
                'recommendation': "Add trust relationships between components to improve overall connectivity",
                'affected_nodes': 'Multiple components'
            })
        
        return recommendations
    
    def _compute_payment_success_rates(self) -> Dict[str, float]:
        """Compute success rates for different types of payments."""
        n = len(self.network.nodes)
        
        if n < 2:
            return {}
        
        # Test payment success rates
        total_tests = 0
        successful_payments = 0
        self_token_successes = 0
        self_token_tests = 0
        
        node_list = list(self.network.nodes.keys())
        
        # Sample payment scenarios
        for source in node_list[:min(10, len(node_list))]:
            for target in node_list[:min(10, len(node_list))]:
                if source != target:
                    # Test with source's own token
                    capacity = self.network.estimate_payment_capacity(source, target, source)
                    
                    total_tests += 1
                    self_token_tests += 1
                    
                    if capacity > 0:
                        successful_payments += 1
                        self_token_successes += 1
        
        return {
            'overall_success_rate': successful_payments / total_tests if total_tests > 0 else 0,
            'self_token_success_rate': self_token_successes / self_token_tests if self_token_tests > 0 else 0,
            'total_tests': total_tests
        }
    
    def _analyze_capacity_distribution(self) -> Dict[str, float]:
        """Analyze distribution of payment capacities."""
        capacities = []
        
        node_list = list(self.network.nodes.keys())
        
        for source in node_list[:min(20, len(node_list))]:
            for target in node_list[:min(20, len(node_list))]:
                if source != target:
                    capacity = self.network.estimate_payment_capacity(source, target, source)
                    if capacity > 0:
                        capacities.append(capacity)
        
        if not capacities:
            return {}
        
        capacities = np.array(capacities)
        
        return {
            'mean_capacity': float(np.mean(capacities)),
            'median_capacity': float(np.median(capacities)),
            'std_capacity': float(np.std(capacities)),
            'min_capacity': float(np.min(capacities)),
            'max_capacity': float(np.max(capacities)),
            'num_viable_paths': len(capacities)
        }
    
    def _compute_clustering_coefficients(self) -> Dict[str, float]:
        """Compute clustering coefficient for each node."""
        W_dense = self.network._W.toarray()
        n = len(self.network.nodes)
        
        clustering_coeffs = {}
        
        for node, info in self.network.nodes.items():
            idx = info.index
            
            # Find neighbors (nodes this node trusts or that trust this node)
            neighbors = set()
            
            # Outgoing edges (nodes this node trusts)
            for j in range(n):
                if W_dense[idx, j] > 0:
                    neighbors.add(j)
            
            # Incoming edges (nodes that trust this node)
            for i in range(n):
                if W_dense[i, idx] > 0:
                    neighbors.add(i)
            
            neighbors.discard(idx)  # Remove self
            
            if len(neighbors) < 2:
                clustering_coeffs[node] = 0.0
                continue
            
            # Count edges between neighbors
            edges_between_neighbors = 0
            possible_edges = len(neighbors) * (len(neighbors) - 1)
            
            for i in neighbors:
                for j in neighbors:
                    if i != j and W_dense[i, j] > 0:
                        edges_between_neighbors += 1
            
            clustering_coeffs[node] = edges_between_neighbors / possible_edges if possible_edges > 0 else 0.0
        
        return clustering_coeffs
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for measuring inequality."""
        if len(values) == 0:
            return 0.0
        
        values = np.array(values)
        values = values[values >= 0]  # Remove negative values
        
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Compute Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * index - n - 1) @ sorted_values / (n * sorted_values.sum())
        
        return float(gini)