"""Main network analyzer class."""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

from .centrality import CentralityAnalyzer
from .paths import PathAnalyzer
from .communities import CommunityAnalyzer
from ..core.data_models import SecurityAnalysis, FlowAnalysis

logger = logging.getLogger(__name__)


class TrustNetworkAnalyzer:
    """Advanced analysis tools for trust networks."""
    
    def __init__(self, network):
        """Initialize analyzer with network."""
        self.network = network
        self.centrality_analyzer = CentralityAnalyzer(network)
        self.path_analyzer = PathAnalyzer(network)
        self.community_analyzer = CommunityAnalyzer(network)
    
    def analyze_sybil_resistance(self, sybil_nodes: set) -> SecurityAnalysis:
        """Analyze the network's resistance to a Sybil attack."""
        result = self.network.compute_trust_scores()
        
        # Calculate total score captured by Sybils
        sybil_scores = [
            result.scores[node].composite_score 
            for node in sybil_nodes 
            if node in result.scores
        ]
        
        total_sybil_score = sum(sybil_scores)
        avg_sybil_score = np.mean(sybil_scores) if sybil_scores else 0
        
        # Calculate theoretical bound
        n = len(self.network.nodes)
        m = len(sybil_nodes)
        alpha = self.network.config.trust_network.parameters.eigentrust.alpha
        
        theoretical_bound = (alpha * m / n) if n > 0 else 0
        
        # Analyze trust leakage from honest nodes
        honest_to_sybil_trust = 0.0
        for edge in self.network.edges:
            if edge.source not in sybil_nodes and edge.target in sybil_nodes:
                honest_to_sybil_trust += edge.weight
        
        return SecurityAnalysis(
            sybil_nodes=sybil_nodes,
            total_sybil_score=total_sybil_score,
            avg_sybil_score=avg_sybil_score,
            theoretical_bound=theoretical_bound,
            actual_vs_bound_ratio=total_sybil_score / theoretical_bound if theoretical_bound > 0 else float('inf'),
            honest_to_sybil_trust=honest_to_sybil_trust,
            resistance_effective=total_sybil_score < 2 * theoretical_bound
        )
    
    def analyze_token_flows(self) -> FlowAnalysis:
        """Analyze token flow patterns in the network."""
        result = self.network.compute_trust_scores()
        
        # Token velocity approximation
        velocities = {}
        for node, score in result.scores.items():
            if score.token_supply > 0:
                # Velocity based on flow centrality and supply
                velocity = score.flow_centrality * 100 / (score.token_supply + 1)
                velocities[node] = velocity
        
        # Concentration metrics (Gini coefficient)
        supplies = [score.token_supply for score in result.scores.values()]
        total_supply = sum(supplies)
        
        if total_supply > 0 and len(supplies) > 1:
            # Calculate Gini coefficient
            sorted_supplies = sorted(supplies)
            n = len(sorted_supplies)
            index = np.arange(1, n + 1)
            gini = (2 * index - n - 1).dot(sorted_supplies) / (n * sum(sorted_supplies))
        else:
            gini = 0.0
        
        # Flow concentration
        flow_scores = [score.flow_centrality for score in result.scores.values()]
        flow_concentration = {}
        
        if flow_scores:
            total_flow = sum(flow_scores)
            if total_flow > 0:
                # Top 10% nodes flow concentration
                sorted_flows = sorted(enumerate(flow_scores), key=lambda x: x[1], reverse=True)
                top_10_percent = max(1, len(sorted_flows) // 10)
                top_flow = sum(flow for _, flow in sorted_flows[:top_10_percent])
                flow_concentration['top_10_percent'] = top_flow / total_flow
        
        return FlowAnalysis(
            token_velocities=velocities,
            gini_coefficient=gini,
            total_supply=total_supply,
            avg_velocity=np.mean(list(velocities.values())) if velocities else 0.0,
            flow_concentration=flow_concentration
        )
    
    def compute_network_resilience(self) -> Dict[str, Any]:
        """Compute network resilience metrics."""
        # Build matrices if needed
        self.network._build_matrices()
        
        n = len(self.network.nodes)
        m = len(self.network.edges)
        
        if n == 0:
            return {"error": "Empty network"}
        
        # Basic resilience metrics
        edge_connectivity = self._compute_edge_connectivity()
        node_connectivity = self._compute_node_connectivity()
        
        # Clustering-based resilience
        clustering_resilience = self._compute_clustering_resilience()
        
        # Converter dependency
        converter_dependency = self._compute_converter_dependency()
        
        return {
            "edge_connectivity": edge_connectivity,
            "node_connectivity": node_connectivity,
            "clustering_resilience": clustering_resilience,
            "converter_dependency": converter_dependency,
            "redundancy_ratio": m / (n * (n - 1)) if n > 1 else 0
        }
    
    def _compute_edge_connectivity(self) -> int:
        """Compute minimum edge connectivity."""
        # Simplified: return minimum out-degree
        if self.network._W is None:
            return 0
        
        out_degrees = np.asarray((self.network._W > 0).sum(axis=1)).ravel()
        return int(out_degrees.min()) if len(out_degrees) > 0 else 0
    
    def _compute_node_connectivity(self) -> int:
        """Compute minimum node connectivity."""
        # Simplified: return minimum number of neighbors
        if self.network._W is None:
            return 0
        
        neighbor_counts = []
        for i in range(self.network._W.shape[0]):
            row = self.network._W.getrow(i)
            col = self.network._W.getcol(i)
            neighbors = set(row.indices) | set(col.indices)
            neighbors.discard(i)  # Remove self
            neighbor_counts.append(len(neighbors))
        
        return min(neighbor_counts) if neighbor_counts else 0
    
    def _compute_clustering_resilience(self) -> float:
        """Compute resilience based on clustering structure."""
        result = self.network.compute_trust_scores()
        clustering_scores = [
            score.clustering_coefficient 
            for score in result.scores.values()
        ]
        return np.mean(clustering_scores) if clustering_scores else 0.0
    
    def _compute_converter_dependency(self) -> Dict[str, float]:
        """Compute dependency on converter nodes."""
        result = self.network.compute_trust_scores()
        
        converter_scores = []
        total_scores = []
        
        for node, score in result.scores.items():
            total_scores.append(score.composite_score)
            if node in self.network.converters:
                converter_scores.append(score.composite_score)
        
        total_score = sum(total_scores)
        converter_score = sum(converter_scores)
        
        return {
            "converter_score_ratio": converter_score / total_score if total_score > 0 else 0,
            "num_converters": len(self.network.converters),
            "converter_centralization": len(self.network.converters) / len(self.network.nodes) if self.network.nodes else 0
        }
    
    def generate_report(self, include_plots: bool = False) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive network analysis report")
        
        # Basic network statistics
        stats = self.network._compute_network_statistics()
        
        # Trust scores
        results = self.network.compute_trust_scores()
        
        # Centrality analysis
        centrality_metrics = self.centrality_analyzer.compute_all_centralities()
        
        # Path analysis
        path_metrics = self.path_analyzer.analyze_path_distribution()
        
        # Community structure
        communities = self.community_analyzer.detect_communities()
        
        # Flow analysis
        flow_analysis = self.analyze_token_flows()
        
        # Resilience analysis
        resilience = self.compute_network_resilience()
        
        report = {
            "network_statistics": {
                "nodes": stats.num_nodes,
                "edges": stats.num_edges,
                "converters": stats.num_converters,
                "density": stats.density,
                "clustering": stats.avg_clustering,
                "diameter": stats.diameter
            },
            "trust_scores": {
                "computation_time": results.computation_time,
                "top_nodes": results.get_ranking(top_n=10),
                "score_distribution": self._compute_score_distribution(results)
            },
            "centrality_analysis": centrality_metrics,
            "path_analysis": path_metrics,
            "community_structure": {
                "num_communities": communities.num_communities,
                "modularity": communities.modularity,
                "largest_community": max(communities.community_sizes.values()) if communities.community_sizes else 0
            },
            "flow_analysis": {
                "gini_coefficient": flow_analysis.gini_coefficient,
                "total_supply": flow_analysis.total_supply,
                "avg_velocity": flow_analysis.avg_velocity
            },
            "resilience": resilience
        }
        
        return report
    
    def _compute_score_distribution(self, results) -> Dict[str, float]:
        """Compute trust score distribution statistics."""
        scores = [score.composite_score for score in results.scores.values()]
        
        if not scores:
            return {}
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
   
    def analyze_payment_bottlenecks(self) -> Dict[str, Any]:
        """
        Comprehensive payment bottleneck analysis using the new BottleneckAnalyzer.
        
        Returns:
            Dictionary with detailed bottleneck analysis
        """
        from .bottleneck_analyzer import BottleneckAnalyzer
        
        bottleneck_analyzer = BottleneckAnalyzer(self.network)
        return bottleneck_analyzer.analyze_all_bottlenecks()

    def compare_algorithms_vs_payment_capacity(self) -> Dict[str, Any]:
        """
        Compare all algorithms against actual payment capacity using AlgorithmComparator.
        
        Returns:
            Dictionary with comprehensive algorithm comparison results
        """
        from .algorithm_comparator import AlgorithmComparator
        
        comparator = AlgorithmComparator(self.network)
        return comparator.compare_all_algorithms()

    def analyze_balance_vs_trust_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between trust relationships and balance distributions.
        
        Returns:
            Analysis of how trust and balances interact in the network
        """
        if self.network._W is None or self.network._B is None:
            self.network._build_matrices()
        
        W_dense = self.network._W.toarray()
        B_dense = self.network._B.toarray()
        n = len(self.network.nodes)
        
        correlation_analysis = {
            'trust_balance_correlation': {},
            'balance_distribution_by_trust': {},
            'trust_asymmetry_vs_balance': {},
            'payment_enabling_analysis': {}
        }
        
        # 1. Correlation between trust and balance holding
        trust_balance_pairs = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    trust_ij = W_dense[i, j]
                    balance_ij = B_dense[i, j]  # i holds j's tokens
                    
                    trust_balance_pairs.append((trust_ij, balance_ij))
        
        if trust_balance_pairs:
            trusts, balances = zip(*trust_balance_pairs)
            
            from scipy.stats import spearmanr, pearsonr
            spearman_corr, spearman_p = spearmanr(trusts, balances)
            pearson_corr, pearson_p = pearsonr(trusts, balances)
            
            correlation_analysis['trust_balance_correlation'] = {
                'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                'spearman_p_value': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                'pearson_p_value': float(pearson_p) if not np.isnan(pearson_p) else 1.0,
                'interpretation': self._interpret_trust_balance_correlation(spearman_corr)
            }
        
        # 2. Balance distribution analysis by trust level
        high_trust_balances = []
        medium_trust_balances = []
        low_trust_balances = []
        no_trust_balances = []
        
        tau = self.network.config.trust_network.parameters.tau
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    trust = W_dense[i, j]
                    balance = B_dense[i, j]
                    
                    if trust >= tau:
                        high_trust_balances.append(balance)
                    elif trust >= tau / 2:
                        medium_trust_balances.append(balance)
                    elif trust > 0:
                        low_trust_balances.append(balance)
                    else:
                        no_trust_balances.append(balance)
        
        correlation_analysis['balance_distribution_by_trust'] = {
            'high_trust': {
                'mean_balance': float(np.mean(high_trust_balances)) if high_trust_balances else 0,
                'median_balance': float(np.median(high_trust_balances)) if high_trust_balances else 0,
                'count': len(high_trust_balances)
            },
            'medium_trust': {
                'mean_balance': float(np.mean(medium_trust_balances)) if medium_trust_balances else 0,
                'median_balance': float(np.median(medium_trust_balances)) if medium_trust_balances else 0,
                'count': len(medium_trust_balances)
            },
            'low_trust': {
                'mean_balance': float(np.mean(low_trust_balances)) if low_trust_balances else 0,
                'median_balance': float(np.median(low_trust_balances)) if low_trust_balances else 0,
                'count': len(low_trust_balances)
            },
            'no_trust': {
                'mean_balance': float(np.mean(no_trust_balances)) if no_trust_balances else 0,
                'median_balance': float(np.median(no_trust_balances)) if no_trust_balances else 0,
                'count': len(no_trust_balances)
            }
        }
        
        # 3. Trust asymmetry vs balance asymmetry
        trust_asymmetries = []
        balance_asymmetries = []
        
        for i in range(n):
            for j in range(n):
                if i < j:  # Avoid duplicates
                    trust_ij = W_dense[i, j]
                    trust_ji = W_dense[j, i]
                    balance_ij = B_dense[i, j]  # i holds j's tokens
                    balance_ji = B_dense[j, i]  # j holds i's tokens
                    
                    trust_asymmetry = abs(trust_ij - trust_ji)
                    balance_asymmetry = abs(balance_ij - balance_ji)
                    
                    trust_asymmetries.append(trust_asymmetry)
                    balance_asymmetries.append(balance_asymmetry)
        
        if trust_asymmetries and balance_asymmetries:
            asym_corr, asym_p = spearmanr(trust_asymmetries, balance_asymmetries)
            
            correlation_analysis['trust_asymmetry_vs_balance'] = {
                'correlation': float(asym_corr) if not np.isnan(asym_corr) else 0.0,
                'p_value': float(asym_p) if not np.isnan(asym_p) else 1.0,
                'avg_trust_asymmetry': float(np.mean(trust_asymmetries)),
                'avg_balance_asymmetry': float(np.mean(balance_asymmetries))
            }
        
        # 4. Payment enabling analysis
        payment_enabling_stats = {
            'trust_only_blocks': 0,    # Trust exists but no balance
            'balance_only_blocks': 0,  # Balance exists but no trust
            'both_enable': 0,         # Both trust and balance exist
            'neither_exists': 0       # Neither trust nor balance
        }
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    trust_exists = W_dense[i, j] >= tau
                    balance_exists = B_dense[i, j] > 0
                    
                    if trust_exists and balance_exists:
                        payment_enabling_stats['both_enable'] += 1
                    elif trust_exists and not balance_exists:
                        payment_enabling_stats['trust_only_blocks'] += 1
                    elif not trust_exists and balance_exists:
                        payment_enabling_stats['balance_only_blocks'] += 1
                    else:
                        payment_enabling_stats['neither_exists'] += 1
        
        total_pairs = sum(payment_enabling_stats.values())
        
        correlation_analysis['payment_enabling_analysis'] = {
            'counts': payment_enabling_stats,
            'percentages': {
                key: float(count / total_pairs * 100) if total_pairs > 0 else 0 
                for key, count in payment_enabling_stats.items()
            },
            'trust_bottleneck_ratio': float(payment_enabling_stats['trust_only_blocks'] / total_pairs) if total_pairs > 0 else 0,
            'balance_bottleneck_ratio': float(payment_enabling_stats['balance_only_blocks'] / total_pairs) if total_pairs > 0 else 0
        }
        
        return correlation_analysis

    def _interpret_trust_balance_correlation(self, correlation: float) -> str:
        """Interpret the trust-balance correlation value."""
        if abs(correlation) < 0.1:
            return "Very weak correlation - trust and balance holdings are largely independent"
        elif abs(correlation) < 0.3:
            return "Weak correlation - some relationship between trust and balance holdings"
        elif abs(correlation) < 0.5:
            return "Moderate correlation - noticeable relationship between trust and balance holdings"
        elif abs(correlation) < 0.7:
            return "Strong correlation - trust relationships strongly influence balance holdings"
        else:
            return "Very strong correlation - trust and balance holdings are highly related"

    def analyze_converter_accessibility(self) -> Dict[str, Any]:
        """
        Analyze accessibility of converter nodes from different perspectives.
        
        Returns:
            Dictionary with converter accessibility analysis
        """
        if not self.network.converters:
            return {'error': 'No converters in network'}
        
        accessibility_analysis = {
            'converter_reachability': {},
            'payment_paths_to_converters': {},
            'converter_capacity_analysis': {},
            'bottleneck_identification': {}
        }
        
        # For each converter
        for converter in self.network.converters:
            converter_stats = {
                'reachable_nodes': 0,
                'total_capacity_received': 0,
                'avg_path_length': 0,
                'payment_paths': []
            }
            
            path_lengths = []
            
            # Check reachability from each node
            for node in self.network.nodes:
                if node != converter:
                    # Try payment with node's own token
                    capacity = self.network.estimate_payment_capacity(node, converter, node)
                    
                    if capacity > 0:
                        converter_stats['reachable_nodes'] += 1
                        converter_stats['total_capacity_received'] += capacity
                        
                        # Get path details
                        path_info = self.network.find_payment_path_with_capacity(node, converter, node)
                        if path_info:
                            path_lengths.append(path_info['hops'])
                            converter_stats['payment_paths'].append({
                                'source': node,
                                'capacity': capacity,
                                'hops': path_info['hops'],
                                'path': path_info['path']
                            })
            
            # Calculate statistics
            total_nodes = len(self.network.nodes) - 1  # Exclude the converter itself
            converter_stats['reachability_ratio'] = converter_stats['reachable_nodes'] / total_nodes if total_nodes > 0 else 0
            converter_stats['avg_capacity'] = converter_stats['total_capacity_received'] / converter_stats['reachable_nodes'] if converter_stats['reachable_nodes'] > 0 else 0
            converter_stats['avg_path_length'] = np.mean(path_lengths) if path_lengths else 0
            
            # Sort paths by capacity
            converter_stats['payment_paths'].sort(key=lambda x: x['capacity'], reverse=True)
            
            accessibility_analysis['converter_reachability'][converter] = converter_stats
        
        # Overall converter accessibility
        total_reachable = sum(stats['reachable_nodes'] for stats in accessibility_analysis['converter_reachability'].values())
        total_possible = len(self.network.converters) * (len(self.network.nodes) - 1)
        
        accessibility_analysis['overall_stats'] = {
            'total_reachable_paths': total_reachable,
            'total_possible_paths': total_possible,
            'overall_reachability_ratio': total_reachable / total_possible if total_possible > 0 else 0,
            'avg_converters_reachable_per_node': total_reachable / (len(self.network.nodes) - len(self.network.converters)) if len(self.network.nodes) > len(self.network.converters) else 0
        }
        
        return accessibility_analysis

    def analyze_network_health_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive network health metrics.
        
        Returns:
            Dictionary with various network health indicators
        """
        health_metrics = {
            'connectivity_health': {},
            'liquidity_health': {},
            'distribution_health': {},
            'resilience_health': {},
            'overall_score': 0.0
        }
        
        # 1. Connectivity Health
        trust_result = self.network.compute_trust_scores()
        payment_bottlenecks = self.network.analyze_network_payment_bottlenecks()
        
        # Basic connectivity
        isolated_nodes = len(payment_bottlenecks.get('trust_isolated', []))
        total_nodes = len(self.network.nodes)
        connectivity_ratio = 1 - (isolated_nodes / total_nodes) if total_nodes > 0 else 0
        
        # Payment success rate
        successful_payments = 0
        total_attempts = 0
        
        node_list = list(self.network.nodes.keys())[:min(20, len(self.network.nodes))]  # Sample for performance
        
        for source in node_list:
            for target in node_list:
                if source != target:
                    total_attempts += 1
                    capacity = self.network.estimate_payment_capacity(source, target, source)
                    if capacity > 0:
                        successful_payments += 1
        
        payment_success_rate = successful_payments / total_attempts if total_attempts > 0 else 0
        
        health_metrics['connectivity_health'] = {
            'connectivity_ratio': float(connectivity_ratio),
            'payment_success_rate': float(payment_success_rate),
            'isolated_nodes_count': isolated_nodes,
            'health_score': float((connectivity_ratio + payment_success_rate) / 2)
        }
        
        # 2. Liquidity Health
        network_liquidity = self.network.compute_network_liquidity_score()
        
        # Balance distribution metrics
        B_dense = self.network._B.toarray()
        all_balances = B_dense.flatten()
        nonzero_balances = all_balances[all_balances > 0]
        
        balance_gini = self._compute_gini_coefficient(nonzero_balances) if len(nonzero_balances) > 0 else 0
        balance_distribution_score = 1 - balance_gini  # Lower Gini = better distribution
        
        # Foreign token availability
        self_tokens = np.diag(B_dense).sum()
        foreign_tokens = B_dense.sum() - self_tokens
        foreign_token_ratio = foreign_tokens / B_dense.sum() if B_dense.sum() > 0 else 0
        
        health_metrics['liquidity_health'] = {
            'network_liquidity_score': float(network_liquidity),
            'balance_distribution_score': float(balance_distribution_score),
            'foreign_token_ratio': float(foreign_token_ratio),
            'balance_gini_coefficient': float(balance_gini),
            'health_score': float((network_liquidity + balance_distribution_score + foreign_token_ratio) / 3)
        }
        
        # 3. Distribution Health
        # Trust distribution
        W_dense = self.network._W.toarray()
        trust_values = W_dense[W_dense > 0]
        trust_gini = self._compute_gini_coefficient(trust_values) if len(trust_values) > 0 else 0
        trust_distribution_score = 1 - trust_gini
        
        # Converter distribution
        if self.network.converters:
            converter_accessibility = self.analyze_converter_accessibility()
            overall_reachability = converter_accessibility['overall_stats']['overall_reachability_ratio']
        else:
            overall_reachability = 0
        
        health_metrics['distribution_health'] = {
            'trust_distribution_score': float(trust_distribution_score),
            'trust_gini_coefficient': float(trust_gini),
            'converter_accessibility': float(overall_reachability),
            'health_score': float((trust_distribution_score + overall_reachability) / 2)
        }
        
        # 4. Resilience Health
        # Critical node analysis
        bottleneck_analysis = self.analyze_payment_bottlenecks()
        
        critical_issues = 0
        if bottleneck_analysis['structural_bottlenecks']['critical_nodes']:
            critical_issues += len(bottleneck_analysis['structural_bottlenecks']['critical_nodes'])
        
        resilience_score = max(0, 1 - (critical_issues / total_nodes)) if total_nodes > 0 else 0
        
        # Network redundancy
        edge_density = trust_result.network_stats.density
        redundancy_score = min(1.0, edge_density * 10)  # Scale density to [0,1]
        
        health_metrics['resilience_health'] = {
            'resilience_score': float(resilience_score),
            'redundancy_score': float(redundancy_score),
            'critical_nodes_count': critical_issues,
            'network_density': float(edge_density),
            'health_score': float((resilience_score + redundancy_score) / 2)
        }
        
        # 5. Overall Health Score
        component_scores = [
            health_metrics['connectivity_health']['health_score'],
            health_metrics['liquidity_health']['health_score'],
            health_metrics['distribution_health']['health_score'],
            health_metrics['resilience_health']['health_score']
        ]
        
        overall_score = np.mean(component_scores)
        health_metrics['overall_score'] = float(overall_score)
        
        # Health rating
        if overall_score >= 0.8:
            health_rating = "Excellent"
        elif overall_score >= 0.6:
            health_rating = "Good"
        elif overall_score >= 0.4:
            health_rating = "Fair"
        elif overall_score >= 0.2:
            health_rating = "Poor"
        else:
            health_rating = "Critical"
        
        health_metrics['health_rating'] = health_rating
        health_metrics['improvement_priority'] = min(
            health_metrics.keys(), 
            key=lambda k: health_metrics[k]['health_score'] if isinstance(health_metrics[k], dict) and 'health_score' in health_metrics[k] else 1.0
        )
        
        return health_metrics

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

    def generate_comprehensive_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report combining all analysis tools.
        
        Args:
            include_recommendations: Whether to include actionable recommendations
            
        Returns:
            Dictionary with comprehensive network analysis
        """
        logger.info("Generating comprehensive network analysis report...")
        
        comprehensive_report = {
            'network_overview': {},
            'trust_scores_analysis': {},
            'payment_capacity_analysis': {},
            'algorithm_comparison': {},
            'bottleneck_analysis': {},
            'network_health': {},
            'balance_trust_correlation': {},
            'converter_analysis': {},
            'recommendations': [],
            'executive_summary': {}
        }
        
        # 1. Network Overview
        stats = self.network._compute_network_statistics()
        comprehensive_report['network_overview'] = {
            'basic_stats': {
                'nodes': stats.num_nodes,
                'edges': stats.num_edges,
                'converters': stats.num_converters,
                'density': stats.density,
                'clustering': stats.avg_clustering
            },
            'network_type': self._classify_network_type(stats),
            'scale': self._classify_network_scale(stats.num_nodes)
        }
        
        # 2. Trust Scores Analysis
        trust_result = self.network.compute_trust_scores()
        comprehensive_report['trust_scores_analysis'] = {
            'computation_time': trust_result.computation_time,
            'top_nodes': trust_result.get_ranking(top_n=10),
            'score_distribution': self._analyze_score_distribution(trust_result),
            'algorithm_used': trust_result.algorithm_used
        }
        
        # 3. Payment Capacity Analysis
        comprehensive_report['payment_capacity_analysis'] = self.analyze_payment_bottlenecks()
        
        # 4. Algorithm Comparison
        comprehensive_report['algorithm_comparison'] = self.compare_algorithms_vs_payment_capacity()
        
        # 5. Bottleneck Analysis (already included in payment_capacity_analysis)
        
        # 6. Network Health
        comprehensive_report['network_health'] = self.analyze_network_health_metrics()
        
        # 7. Balance-Trust Correlation
        comprehensive_report['balance_trust_correlation'] = self.analyze_balance_vs_trust_correlation()
        
        # 8. Converter Analysis
        if self.network.converters:
            comprehensive_report['converter_analysis'] = self.analyze_converter_accessibility()
        
        # 9. Recommendations
        if include_recommendations:
            comprehensive_report['recommendations'] = self._generate_comprehensive_recommendations(comprehensive_report)
        
        # 10. Executive Summary
        comprehensive_report['executive_summary'] = self._generate_executive_summary(comprehensive_report)
        
        logger.info("Comprehensive analysis report generated successfully")
        
        return comprehensive_report

    def _classify_network_type(self, stats) -> str:
        """Classify the type of network based on structure."""
        if stats.density > 0.5:
            return "Dense Network"
        elif stats.avg_clustering > 0.3:
            return "Clustered Network" 
        elif stats.density < 0.1:
            return "Sparse Network"
        else:
            return "Moderate Network"

    def _classify_network_scale(self, num_nodes: int) -> str:
        """Classify network scale."""
        if num_nodes < 10:
            return "Small"
        elif num_nodes < 100:
            return "Medium"
        elif num_nodes < 1000:
            return "Large"
        else:
            return "Very Large"

    def _analyze_score_distribution(self, trust_result) -> Dict[str, float]:
        """Analyze the distribution of trust scores."""
        scores = [score.composite_score for score in trust_result.scores.values()]
        
        if not scores:
            return {}
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'coefficient_of_variation': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        }

    def _generate_comprehensive_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        # Based on network health
        health = report['network_health']
        overall_health = health['overall_score']
        
        if overall_health < 0.6:
            recommendations.append({
                'category': 'Network Health',
                'priority': 'High',
                'issue': f"Overall network health is {health['health_rating'].lower()} ({overall_health:.2f})",
                'recommendation': f"Focus on improving {health['improvement_priority']} as the primary concern",
                'impact': 'High - affects overall network functionality'
            })
        
        # Based on algorithm comparison
        algo_comparison = report['algorithm_comparison']
        if 'summary' in algo_comparison:
            summary = algo_comparison['summary']
            if summary['balance_aware_advantage']['win_rate'] > 0.7:
                recommendations.append({
                    'category': 'Algorithm Selection',
                    'priority': 'Medium',
                    'issue': 'Balance-aware algorithms significantly outperform trust-only algorithms',
                    'recommendation': f"Use {summary['best_algorithm']} for payment-related applications",
                    'impact': 'Medium - improves prediction accuracy'
                })
        
        # Based on payment bottlenecks
        bottlenecks = report['payment_capacity_analysis']
        if 'payment_bottlenecks' in bottlenecks:
            payment_issues = bottlenecks['payment_bottlenecks']['basic_bottlenecks']
            
            if payment_issues.get('no_self_tokens'):
                recommendations.append({
                    'category': 'Token Distribution',
                    'priority': 'High',
                    'issue': f"{len(payment_issues['no_self_tokens'])} nodes cannot initiate payments",
                    'recommendation': 'Issue initial token supplies to nodes without self-tokens',
                    'impact': 'High - enables basic payment functionality'
                })
        
        # Based on converter analysis
        if 'converter_analysis' in report and report['converter_analysis']:
            conv_analysis = report['converter_analysis']
            if 'overall_stats' in conv_analysis:
                reachability = conv_analysis['overall_stats']['overall_reachability_ratio']
                if reachability < 0.5:
                    recommendations.append({
                        'category': 'Converter Accessibility',
                        'priority': 'Medium',
                        'issue': f"Only {reachability:.1%} of possible converter paths are viable",
                        'recommendation': 'Improve trust paths to converters or add more conversion capacity',
                        'impact': 'Medium - improves fiat conversion access'
                    })
        
        return recommendations

    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the comprehensive analysis."""
        
        # Key metrics
        network_size = report['network_overview']['basic_stats']['nodes']
        health_score = report['network_health']['overall_score']
        health_rating = report['network_health']['health_rating']
        
        # Payment functionality
        payment_success = 0
        if 'payment_bottlenecks' in report['payment_capacity_analysis']:
            success_rates = report['payment_capacity_analysis']['payment_bottlenecks'].get('success_rates', {})
            payment_success = success_rates.get('overall_success_rate', 0)
        
        # Algorithm performance
        best_algorithm = "N/A"
        if 'algorithm_comparison' in report and 'summary' in report['algorithm_comparison']:
            best_algorithm = report['algorithm_comparison']['summary'].get('best_algorithm', 'N/A')
        
        # Critical issues count
        critical_issues = len(report.get('recommendations', []))
        high_priority_issues = len([r for r in report.get('recommendations', []) if r.get('priority') == 'High'])
        
        return {
            'network_size': network_size,
            'health_score': float(health_score),
            'health_rating': health_rating,
            'payment_success_rate': float(payment_success),
            'best_algorithm': best_algorithm,
            'total_issues_identified': critical_issues,
            'high_priority_issues': high_priority_issues,
            'key_insight': self._generate_key_insight(report),
            'primary_recommendation': self._get_primary_recommendation(report)
        }

    def _generate_key_insight(self, report: Dict[str, Any]) -> str:
        """Generate the key insight from the analysis."""
        health_score = report['network_health']['overall_score']
        
        # Check algorithm comparison results
        algo_comparison = report.get('algorithm_comparison', {})
        balance_advantage = False
        
        if 'summary' in algo_comparison:
            summary = algo_comparison['summary']
            balance_advantage = summary.get('balance_aware_advantage', {}).get('win_rate', 0) > 0.6
        
        if health_score >= 0.8:
            if balance_advantage:
                return "Network is highly functional with balance-aware algorithms providing superior payment capacity prediction."
            else:
                return "Network is highly functional with strong structural properties."
        elif health_score >= 0.6:
            if balance_advantage:
                return "Network functions well but would benefit from balance-aware algorithms for payment optimization."
            else:
                return "Network has good basic functionality but may need structural improvements."
        else:
            return "Network has significant limitations requiring immediate attention to payment infrastructure."

    def _get_primary_recommendation(self, report: Dict[str, Any]) -> str:
        """Get the primary recommendation based on analysis."""
        recommendations = report.get('recommendations', [])
        
        # Find highest priority recommendation
        high_priority = [r for r in recommendations if r.get('priority') == 'High']
        
        if high_priority:
            return high_priority[0]['recommendation']
        elif recommendations:
            return recommendations[0]['recommendation']
        else:
            return "Network appears to be functioning well - consider regular monitoring and optimization."