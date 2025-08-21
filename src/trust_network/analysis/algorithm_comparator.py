"""
Algorithm comparison tool implementing theoretical framework insights.

This module compares algorithms that use balances vs those that don't,
analyzing correlation with actual payment capacity to validate the theory.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr, pearsonr
import logging

logger = logging.getLogger(__name__)


class AlgorithmComparator:
    """Compare different trust algorithms and their correlation with payment capacity."""
    
    def __init__(self, network):
        """Initialize with a TrustNetwork instance."""
        self.network = network
        
    def compare_all_algorithms(self) -> Dict[str, Any]:
        """
        Comprehensive comparison of all algorithms against actual payment capacity.
        
        Returns:
            Dictionary with detailed comparison results
        """
        # Build matrices if needed
        self.network._build_matrices()
        
        logger.info("Computing all algorithm scores for comparison...")
        
        # Compute all algorithm scores
        algorithm_scores = self._compute_all_algorithm_scores()
        
        # Compute actual payment capacities
        logger.info("Computing actual payment capacities...")
        payment_capacities = self._compute_actual_payment_capacities()
        
        # Perform correlation analysis
        logger.info("Analyzing correlations...")
        correlations = self._analyze_correlations(algorithm_scores, payment_capacities)
        
        # Algorithm classification analysis
        classification_analysis = self._classify_algorithms(algorithm_scores, correlations)
        
        # Performance analysis
        performance_analysis = self._analyze_algorithm_performance(algorithm_scores)
        
        # Generate insights
        insights = self._generate_insights(correlations, classification_analysis)
        
        return {
            'algorithm_scores': algorithm_scores,
            'payment_capacities': payment_capacities,
            'correlations': correlations,
            'classification': classification_analysis,
            'performance': performance_analysis,
            'insights': insights,
            'summary': self._generate_summary(correlations, classification_analysis)
        }
    
    def _compute_all_algorithm_scores(self) -> Dict[str, np.ndarray]:
        """Compute scores for all available algorithms."""
        from ..algorithms.social import get_social_algorithm
        from ..algorithms.liquidity import get_liquidity_algorithm
        
        n = len(self.network.nodes)
        scores = {}
        
        # Social algorithms (pure trust - no balance usage)
        social_algorithms = ['eigentrust', 'appleseed', 'pagerank']
        
        for alg_name in social_algorithms:
            try:
                logger.debug(f"Computing {alg_name} scores...")
                alg = get_social_algorithm(alg_name, self.network.config.trust_network.parameters)
                
                scores[alg_name] = alg.compute(
                    self.network._W,
                    converters=list(self.network.converters),
                    node_to_idx=self.network.node_to_idx
                )
                
                logger.debug(f"{alg_name} score range: [{scores[alg_name].min():.4f}, {scores[alg_name].max():.4f}]")
                
            except Exception as e:
                logger.warning(f"Failed to compute {alg_name}: {e}")
                scores[alg_name] = np.zeros(n)
        
        # Liquidity algorithms (balance-aware)
        liquidity_algorithms = ['conductance', 'flow_centrality', 'hybrid']
        
        for alg_name in liquidity_algorithms:
            try:
                logger.debug(f"Computing {alg_name} scores...")
                alg = get_liquidity_algorithm(alg_name, self.network.config.trust_network.parameters)
                
                scores[alg_name] = alg.compute(
                    trust_matrix=self.network._W,
                    balance_matrix=self.network._B,
                    rate_matrix=self.network._R,
                    converters=list(self.network.converters),
                    node_to_idx=self.network.node_to_idx,
                    tau=self.network.config.trust_network.parameters.tau
                )
                
                logger.debug(f"{alg_name} score range: [{scores[alg_name].min():.4f}, {scores[alg_name].max():.4f}]")
                
            except Exception as e:
                logger.warning(f"Failed to compute {alg_name}: {e}")
                scores[alg_name] = np.zeros(n)
        
        # Composite scores using current network configuration
        try:
            logger.debug("Computing composite scores...")
            trust_result = self.network.compute_trust_scores()
            scores['composite'] = np.array([
                trust_result.scores[node].composite_score 
                for node in self.network.nodes.keys()
            ])
            
        except Exception as e:
            logger.warning(f"Failed to compute composite scores: {e}")
            scores['composite'] = np.zeros(n)
        
        return scores
    
    def _compute_actual_payment_capacities(self) -> Dict[str, np.ndarray]:
        """
        Compute actual payment capacities for each node.
        
        Returns capacities in different scenarios:
        - outgoing_capacity: Total capacity for outgoing payments
        - incoming_capacity: Total capacity for incoming payments  
        - self_token_capacity: Capacity using node's own tokens
        - converter_capacity: Capacity to reach converters
        """
        n = len(self.network.nodes)
        
        capacities = {
            'outgoing_capacity': np.zeros(n),
            'incoming_capacity': np.zeros(n),
            'self_token_capacity': np.zeros(n),
            'converter_capacity': np.zeros(n)
        }
        
        node_list = list(self.network.nodes.keys())
        
        # Compute outgoing and incoming capacities
        for i, source in enumerate(node_list):
            total_outgoing = 0
            total_incoming = 0
            self_token_total = 0
            converter_total = 0
            
            for target in node_list:
                if source != target:
                    # Outgoing capacity (source pays target)
                    outgoing = self.network.estimate_payment_capacity(source, target, source)
                    total_outgoing += outgoing
                    self_token_total += outgoing
                    
                    # Incoming capacity (target pays source)
                    incoming = self.network.estimate_payment_capacity(target, source, target)
                    total_incoming += incoming
            
            # Converter capacity
            for converter in self.network.converters:
                if converter != source:
                    conv_capacity = self.network.estimate_payment_capacity(source, converter, source)
                    converter_total += conv_capacity
            
            capacities['outgoing_capacity'][i] = total_outgoing
            capacities['incoming_capacity'][i] = total_incoming  
            capacities['self_token_capacity'][i] = self_token_total
            capacities['converter_capacity'][i] = converter_total
        
        logger.debug(f"Payment capacity ranges computed:")
        for cap_type, cap_values in capacities.items():
            logger.debug(f"  {cap_type}: [{cap_values.min():.2f}, {cap_values.max():.2f}]")
        
        return capacities
    
    def _analyze_correlations(self, algorithm_scores: Dict[str, np.ndarray], 
                            payment_capacities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between algorithm scores and payment capacities."""
        correlations = {}
        
        # For each algorithm
        for alg_name, alg_scores in algorithm_scores.items():
            correlations[alg_name] = {}
            
            # Against each capacity type
            for cap_type, cap_values in payment_capacities.items():
                # Spearman correlation (rank-based, robust to outliers)
                spearman_corr, spearman_p = spearmanr(alg_scores, cap_values)
                
                # Pearson correlation (linear relationship)
                pearson_corr, pearson_p = pearsonr(alg_scores, cap_values)
                
                correlations[alg_name][cap_type] = {
                    'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                    'spearman_p_value': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                    'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                    'pearson_p_value': float(pearson_p) if not np.isnan(pearson_p) else 1.0,
                    'significant': spearman_p < 0.05 if not np.isnan(spearman_p) else False
                }
        
        return correlations
    
    def _classify_algorithms(self, algorithm_scores: Dict[str, np.ndarray], 
                           correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Classify algorithms by their properties and performance."""
        
        # Algorithm classification by balance usage
        balance_aware = ['conductance', 'flow_centrality', 'hybrid', 'composite']
        trust_only = ['eigentrust', 'appleseed', 'pagerank']
        
        classification = {
            'balance_aware': {
                'algorithms': balance_aware,
                'description': 'Algorithms that incorporate token balance information'
            },
            'trust_only': {
                'algorithms': trust_only,
                'description': 'Algorithms based purely on trust relationships'
            },
            'performance_comparison': {}
        }
        
        # Compare performance between categories
        for cap_type in ['outgoing_capacity', 'self_token_capacity', 'converter_capacity']:
            balance_aware_corrs = []
            trust_only_corrs = []
            
            for alg in balance_aware:
                if alg in correlations and cap_type in correlations[alg]:
                    corr = correlations[alg][cap_type]['spearman_correlation']
                    balance_aware_corrs.append(abs(corr))
            
            for alg in trust_only:
                if alg in correlations and cap_type in correlations[alg]:
                    corr = correlations[alg][cap_type]['spearman_correlation']
                    trust_only_corrs.append(abs(corr))
            
            if balance_aware_corrs and trust_only_corrs:
                avg_balance_aware = np.mean(balance_aware_corrs)
                avg_trust_only = np.mean(trust_only_corrs)
                
                classification['performance_comparison'][cap_type] = {
                    'balance_aware_avg_correlation': float(avg_balance_aware),
                    'trust_only_avg_correlation': float(avg_trust_only),
                    'balance_aware_advantage': float(avg_balance_aware - avg_trust_only),
                    'balance_aware_better': avg_balance_aware > avg_trust_only
                }
        
        return classification
    
    def _analyze_algorithm_performance(self, algorithm_scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze performance characteristics of each algorithm."""
        performance = {}
        
        for alg_name, scores in algorithm_scores.items():
            if len(scores) == 0:
                continue
                
            performance[alg_name] = {
                'score_statistics': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'range': float(np.max(scores) - np.min(scores)),
                    'coefficient_of_variation': float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                },
                'distribution_properties': {
                    'median': float(np.median(scores)),
                    'q25': float(np.percentile(scores, 25)),
                    'q75': float(np.percentile(scores, 75)),
                    'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                    'skewness': float(self._compute_skewness(scores)),
                    'num_zeros': int(np.sum(scores == 0)),
                    'num_nonzeros': int(np.sum(scores > 0))
                }
            }
        
        return performance
    
    def _generate_insights(self, correlations: Dict[str, Any], 
                         classification: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate insights from the analysis."""
        insights = []
        
        # Insight 1: Balance-aware vs Trust-only comparison
        if 'performance_comparison' in classification:
            for cap_type, comparison in classification['performance_comparison'].items():
                if comparison['balance_aware_better']:
                    advantage = comparison['balance_aware_advantage']
                    insights.append({
                        'category': 'Algorithm Performance',
                        'insight': f"Balance-aware algorithms outperform trust-only algorithms for {cap_type}",
                        'detail': f"Average correlation advantage: {advantage:.3f}",
                        'implication': "Incorporating balance information improves payment capacity prediction"
                    })
        
        # Insight 2: Best performing algorithm overall
        best_alg_scores = {}
        for alg_name, alg_corrs in correlations.items():
            total_corr = 0
            count = 0
            for cap_type, corr_data in alg_corrs.items():
                total_corr += abs(corr_data['spearman_correlation'])
                count += 1
            
            if count > 0:
                best_alg_scores[alg_name] = total_corr / count
        
        if best_alg_scores:
            best_alg = max(best_alg_scores.keys(), key=lambda k: best_alg_scores[k])
            best_score = best_alg_scores[best_alg]
            
            insights.append({
                'category': 'Best Algorithm',
                'insight': f"{best_alg} shows highest correlation with payment capacity",
                'detail': f"Average correlation: {best_score:.3f}",
                'implication': f"Use {best_alg} for applications requiring payment capacity prediction"
            })
        
        # Insight 3: Trust-only algorithm performance
        trust_only_performance = []
        for alg in ['eigentrust', 'appleseed', 'pagerank']:
            if alg in correlations:
                avg_corr = np.mean([
                    abs(corr_data['spearman_correlation']) 
                    for corr_data in correlations[alg].values()
                ])
                trust_only_performance.append((alg, avg_corr))
        
        if trust_only_performance:
            trust_only_performance.sort(key=lambda x: x[1], reverse=True)
            best_trust_alg, best_trust_score = trust_only_performance[0]
            
            insights.append({
                'category': 'Trust-Only Algorithms',
                'insight': f"{best_trust_alg} performs best among pure trust algorithms",
                'detail': f"Average correlation: {best_trust_score:.3f}",
                'implication': "For reputation-only applications, this is the recommended algorithm"
            })
        
        # Insight 4: Correlation significance
        significant_correlations = []
        for alg_name, alg_corrs in correlations.items():
            for cap_type, corr_data in alg_corrs.items():
                if corr_data['significant'] and abs(corr_data['spearman_correlation']) > 0.5:
                    significant_correlations.append((alg_name, cap_type, corr_data['spearman_correlation']))
        
        if significant_correlations:
            insights.append({
                'category': 'Statistical Significance',
                'insight': f"Found {len(significant_correlations)} strong significant correlations (|r| > 0.5, p < 0.05)",
                'detail': "Strong statistical evidence for algorithm-capacity relationships",
                'implication': "Results are statistically reliable for practical applications"
            })
        
        # Insight 5: Algorithm diversity
        score_correlations = []
        alg_names = list(correlations.keys())
        
        for i, alg1 in enumerate(alg_names):
            for alg2 in alg_names[i+1:]:
                if alg1 in correlations and alg2 in correlations:
                    # Find a common capacity type
                    common_cap_types = set(correlations[alg1].keys()) & set(correlations[alg2].keys())
                    if common_cap_types:
                        cap_type = list(common_cap_types)[0]
                        corr1 = correlations[alg1][cap_type]['spearman_correlation']
                        corr2 = correlations[alg2][cap_type]['spearman_correlation']
                        
                        # Simple correlation between the correlations (rough measure of similarity)
                        similarity = abs(corr1 - corr2)
                        score_correlations.append((alg1, alg2, similarity))
        
        if score_correlations:
            avg_diversity = np.mean([sim for _, _, sim in score_correlations])
            insights.append({
                'category': 'Algorithm Diversity',
                'insight': f"Algorithms show {'high' if avg_diversity > 0.3 else 'low'} diversity in performance",
                'detail': f"Average performance difference: {avg_diversity:.3f}",
                'implication': "Algorithm choice significantly impacts results" if avg_diversity > 0.3 else "Algorithms are relatively similar"
            })
        
        return insights
    
    def _generate_summary(self, correlations: Dict[str, Any], 
                        classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the comparison."""
        
        # Find best overall algorithm
        best_overall = None
        best_score = -1
        
        for alg_name, alg_corrs in correlations.items():
            avg_corr = np.mean([
                abs(corr_data['spearman_correlation']) 
                for corr_data in alg_corrs.values()
            ])
            
            if avg_corr > best_score:
                best_score = avg_corr
                best_overall = alg_name
        
        # Count significant correlations
        total_correlations = 0
        significant_correlations = 0
        
        for alg_corrs in correlations.values():
            for corr_data in alg_corrs.values():
                total_correlations += 1
                if corr_data['significant']:
                    significant_correlations += 1
        
        # Balance-aware vs trust-only summary
        balance_wins = 0
        total_comparisons = 0
        
        if 'performance_comparison' in classification:
            for comparison in classification['performance_comparison'].values():
                total_comparisons += 1
                if comparison['balance_aware_better']:
                    balance_wins += 1
        
        return {
            'best_algorithm': best_overall,
            'best_algorithm_score': float(best_score),
            'total_correlations_tested': total_correlations,
            'significant_correlations': significant_correlations,
            'significance_rate': float(significant_correlations / total_correlations) if total_correlations > 0 else 0,
            'balance_aware_advantage': {
                'wins': balance_wins,
                'total_comparisons': total_comparisons,
                'win_rate': float(balance_wins / total_comparisons) if total_comparisons > 0 else 0
            },
            'key_finding': self._generate_key_finding(correlations, classification)
        }
    
    def _generate_key_finding(self, correlations: Dict[str, Any], 
                            classification: Dict[str, Any]) -> str:
        """Generate the key finding from the analysis."""
        
        # Check if balance-aware algorithms consistently outperform
        balance_wins = 0
        total_comparisons = 0
        
        if 'performance_comparison' in classification:
            for comparison in classification['performance_comparison'].values():
                total_comparisons += 1
                if comparison['balance_aware_better']:
                    balance_wins += 1
        
        balance_win_rate = balance_wins / total_comparisons if total_comparisons > 0 else 0
        
        if balance_win_rate >= 0.8:
            return "Balance-aware algorithms consistently outperform trust-only algorithms for payment capacity prediction, validating the theoretical framework."
        elif balance_win_rate >= 0.6:
            return "Balance-aware algorithms generally outperform trust-only algorithms, supporting the importance of balance information."
        elif balance_win_rate >= 0.4:
            return "Mixed results between balance-aware and trust-only algorithms, suggesting context-dependent performance."
        else:
            return "Trust-only algorithms perform surprisingly well, possibly indicating structural network properties dominate balance effects."
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        
        return float(skewness)
    
    def export_comparison_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Export comparison results to JSON file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        
        for key, value in results.items():
            if key == 'algorithm_scores':
                export_data[key] = {alg: scores.tolist() for alg, scores in value.items()}
            elif key == 'payment_capacities':
                export_data[key] = {cap_type: caps.tolist() for cap_type, caps in value.items()}
            else:
                export_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Comparison results exported to {filepath}")