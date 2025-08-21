"""
Hybrid liquidity algorithm with complete balance integration.

Implements the theoretical hybrid formula:
L(v) = Σ_{i=1}^4 w_i * f̂_i(v)

Components:
1. Structural Cohesion: f̂_1(v) = 1 - φ_B(N_k(v)) (balance-weighted conductance)
2. Balance-Weighted Flow: f̂_2(v) = FC_B(v) (flow centrality)
3. Effective Conversion Rate: f̂_3(v) = R̂(v) (with funded path checking)
4. Token Supply: f̂_4(v) = S(v) (total token supply)
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any
import logging

from ..base import LiquidityAlgorithm
from .conductance import ConductanceAlgorithm
from .flow_centrality import FlowCentralityAlgorithm

logger = logging.getLogger(__name__)

class HybridLiquidityAlgorithm(LiquidityAlgorithm):
    """
    Hybrid algorithm combining balance-weighted conductance, flow centrality, 
    effective rates, and token supplies with proper balance integration.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize component algorithms
        self.conductance_alg = ConductanceAlgorithm(config)
        self.flow_alg = FlowCentralityAlgorithm(config)
        
        # Component weights
        self.w_conductance = getattr(config, 'w_conductance', 0.25)
        self.w_flow = getattr(config, 'w_flow', 0.25) 
        self.w_rates = getattr(config, 'w_rates', 0.30)
        self.w_supplies = getattr(config, 'w_supplies', 0.20)
        
        # Validate weights sum to 1
        total_weight = self.w_conductance + self.w_flow + self.w_rates + self.w_supplies
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Component weights sum to {total_weight}, normalizing")
            self.w_conductance /= total_weight
            self.w_flow /= total_weight
            self.w_rates /= total_weight
            self.w_supplies /= total_weight
        
        # Parameters for funded path checking
        self.max_path_length = getattr(config, 'max_path_length', 6)
        self.min_path_capacity = getattr(config, 'min_path_capacity', 1.0)
    
    def compute(self, trust_matrix: sp.csr_matrix,
                balance_matrix: sp.csr_matrix,
                rate_matrix: np.ndarray,
                converters: List[str],
                node_to_idx: Dict[str, int],
                tau: float,
                **kwargs) -> np.ndarray:
        """Compute hybrid liquidity scores with complete balance integration."""
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        logger.debug(f"Computing hybrid liquidity for {n} nodes")
        
        # Component 1: Balance-weighted conductance (structural cohesion)
        logger.debug("Computing balance-weighted conductance...")
        structure_scores = self._compute_structural_cohesion(trust_matrix, balance_matrix)
        
        # Component 2: Balance-weighted flow centrality
        logger.debug("Computing balance-weighted flow centrality...")
        flow_scores = self._compute_balance_weighted_flow(trust_matrix, balance_matrix)
        
        # Component 3: Effective conversion rates with funded path checking
        logger.debug("Computing effective rates with funded paths...")
        rate_scores = self._compute_effective_rates_with_funded_paths(
            trust_matrix, balance_matrix, rate_matrix, converters, node_to_idx, tau
        )
        
        # Component 4: Token supply scores
        logger.debug("Computing token supply scores...")
        supply_scores = self._compute_token_supply_scores(balance_matrix)
        
        # Normalize all components to [0, 1]
        structure_norm = self._normalize_with_enhancement(structure_scores)
        flow_norm = self._normalize_with_enhancement(flow_scores)
        rate_norm = self._normalize_with_enhancement(rate_scores)
        supply_norm = self._normalize_with_enhancement(supply_scores)
        
        # Log component statistics
        logger.debug(f"Component ranges - Structure: [{structure_norm.min():.3f}, {structure_norm.max():.3f}]")
        logger.debug(f"Component ranges - Flow: [{flow_norm.min():.3f}, {flow_norm.max():.3f}]")
        logger.debug(f"Component ranges - Rates: [{rate_norm.min():.3f}, {rate_norm.max():.3f}]")
        logger.debug(f"Component ranges - Supply: [{supply_norm.min():.3f}, {supply_norm.max():.3f}]")
        
        # Weighted combination implementing L(v) = Σ w_i * f̂_i(v)
        liquidity_scores = (
            self.w_conductance * structure_norm +     # Structural cohesion
            self.w_flow * flow_norm +                 # Flow importance
            self.w_rates * rate_norm +                # Conversion capability
            self.w_supplies * supply_norm             # Token availability
        )
        
        logger.debug(f"Final liquidity scores range: [{liquidity_scores.min():.3f}, {liquidity_scores.max():.3f}]")
        
        return liquidity_scores
    
    def _compute_structural_cohesion(self, trust_matrix: sp.csr_matrix,
                                   balance_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute structural cohesion: f̂_1(v) = 1 - φ_B(N_k(v))
        
        Lower conductance = higher cohesion = better liquidity position
        """
        n = trust_matrix.shape[0]
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        
        conductances = np.zeros(n)
        
        for i in range(n):
            # Use conductance algorithm's balance-weighted computation
            conductance = self.conductance_alg._compute_balance_weighted_conductance(
                W_dense, B_dense, i, k_hops=2
            )
            conductances[i] = conductance
        
        # Invert conductance: lower conductance = higher structural score
        structure_scores = 1.0 - conductances
        
        return structure_scores
    
    def _compute_balance_weighted_flow(self, trust_matrix: sp.csr_matrix,
                                     balance_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute balance-weighted flow centrality: f̂_2(v) = FC_B(v)
        """
        return self.flow_alg._compute_balance_weighted_flow_centrality(
            trust_matrix, balance_matrix
        )
    
    def _compute_effective_rates_with_funded_paths(self, trust_matrix: sp.csr_matrix,
                                                 balance_matrix: sp.csr_matrix,
                                                 rate_matrix: np.ndarray,
                                                 converters: List[str],
                                                 node_to_idx: Dict[str, int],
                                                 tau: float) -> np.ndarray:
        """
        Compute effective conversion rates: f̂_3(v) = R̂(v)
        
        Implements: R̂(v) = max_{c∈K} {R(c, T(v)) * 1[funded path c → v]}
        """
        n = trust_matrix.shape[0]
        effective_rates = np.zeros(n)
        
        if len(converters) == 0 or rate_matrix.size == 0:
            return effective_rates
        
        converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
        
        # For each converter
        for i, conv_idx in enumerate(converter_indices):
            if i >= rate_matrix.shape[0]:
                continue
            
            # For each token
            for token_idx in range(n):
                if token_idx >= rate_matrix.shape[1]:
                    continue
                
                rate = rate_matrix[i, token_idx]
                if rate <= 0:
                    continue
                
                # Check if funded path exists from token issuer to converter
                # This means converter can receive the token for conversion
                funded_path_exists = self._check_funded_path(
                    trust_matrix, balance_matrix, token_idx, conv_idx, token_idx, tau
                )
                
                if funded_path_exists:
                    effective_rates[token_idx] = max(effective_rates[token_idx], rate)
        
        return effective_rates
    
    def _check_funded_path(self, trust_matrix: sp.csr_matrix, balance_matrix: sp.csr_matrix,
                          source: int, target: int, token_issuer: int, tau: float) -> bool:
        """
        Check if funded path exists for token flow.
        
        A funded path requires:
        1. Trust path exists from source to target
        2. Each node on path has sufficient balance of the token
        3. Each next node trusts the token issuer (accepts the token)
        """
        # Find trust path
        path = self._find_trust_path(trust_matrix, source, target, tau)
        if not path:
            return False
        
        # Check path length
        if len(path) > self.max_path_length:
            return False
        
        # Check funding along path
        min_capacity = float('inf')
        
        for i in range(len(path)):
            current = path[i]
            
            # Check if current node has the token
            balance = balance_matrix[current, token_issuer]
            if balance < self.min_path_capacity:
                return False
            
            min_capacity = min(min_capacity, balance)
            
            # Check if next node accepts the token (trusts token issuer)
            if i < len(path) - 1:
                next_node = path[i + 1]
                if trust_matrix[next_node, token_issuer] < tau:
                    return False
        
        return min_capacity >= self.min_path_capacity
    
    def _find_trust_path(self, trust_matrix: sp.csr_matrix, source: int, 
                        target: int, tau: float) -> List[int]:
        """Find trust path using BFS."""
        if source == target:
            return [source]
        
        from collections import deque
        
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            current, path = queue.popleft()
            
            # Find nodes that trust current (can receive from current)
            for next_node in range(trust_matrix.shape[0]):
                if (trust_matrix[next_node, current] >= tau and 
                    next_node not in visited):
                    
                    new_path = path + [next_node]
                    
                    if next_node == target:
                        return new_path
                    
                    if len(new_path) < self.max_path_length:
                        visited.add(next_node)
                        queue.append((next_node, new_path))
        
        return []
    
    def _compute_token_supply_scores(self, balance_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute token supply scores: f̂_4(v) = S(v)
        
        S(v) = total supply of token T(v) = Σ_i B[i,v]
        """
        # Column sums give total supply for each token
        token_supplies = np.asarray(balance_matrix.sum(axis=0)).ravel()
        return token_supplies
    
    def _normalize_with_enhancement(self, scores: np.ndarray) -> np.ndarray:
        """
        Enhanced normalization that preserves relative differences and handles edge cases.
        """
        scores = np.array(scores, dtype=np.float64)
        
        if len(scores) == 0:
            return scores
        
        # Handle all-zero case
        if np.all(scores == 0):
            return scores
        
        # Handle negative values
        if np.any(scores < 0):
            scores = scores - scores.min()
        
        # Min-max normalization
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-10:
            # All values are essentially the same
            return np.full_like(scores, 0.5)
        
        normalized = (scores - min_val) / (max_val - min_val)
        
        # Apply slight enhancement to prevent complete flattening
        # This maintains ranking while ensuring proper [0,1] range
        normalized = normalized * 0.9 + 0.05
        
        return normalized
    
    def compute_component_analysis(self, trust_matrix: sp.csr_matrix,
                                 balance_matrix: sp.csr_matrix,
                                 rate_matrix: np.ndarray,
                                 converters: List[str],
                                 node_to_idx: Dict[str, int],
                                 tau: float) -> Dict[str, np.ndarray]:
        """
        Compute and return all components separately for analysis.
        
        Returns:
            Dictionary with all four components and their normalized versions
        """
        n = trust_matrix.shape[0]
        
        if n == 0:
            return {}
        
        # Compute all components
        structure_raw = self._compute_structural_cohesion(trust_matrix, balance_matrix)
        flow_raw = self._compute_balance_weighted_flow(trust_matrix, balance_matrix)
        rates_raw = self._compute_effective_rates_with_funded_paths(
            trust_matrix, balance_matrix, rate_matrix, converters, node_to_idx, tau
        )
        supply_raw = self._compute_token_supply_scores(balance_matrix)
        
        # Normalized versions
        structure_norm = self._normalize_with_enhancement(structure_raw)
        flow_norm = self._normalize_with_enhancement(flow_raw)
        rates_norm = self._normalize_with_enhancement(rates_raw)
        supply_norm = self._normalize_with_enhancement(supply_raw)
        
        return {
            'structure_raw': structure_raw,
            'flow_raw': flow_raw,
            'rates_raw': rates_raw,
            'supply_raw': supply_raw,
            'structure_normalized': structure_norm,
            'flow_normalized': flow_norm,
            'rates_normalized': rates_norm,
            'supply_normalized': supply_norm,
            'weights': {
                'conductance': self.w_conductance,
                'flow': self.w_flow,
                'rates': self.w_rates,
                'supplies': self.w_supplies
            }
        }
    
    def analyze_component_correlations(self, trust_matrix: sp.csr_matrix,
                                     balance_matrix: sp.csr_matrix,
                                     rate_matrix: np.ndarray,
                                     converters: List[str],
                                     node_to_idx: Dict[str, int],
                                     tau: float) -> Dict[str, float]:
        """
        Analyze correlations between different components.
        
        Returns:
            Dictionary with pairwise correlations between components
        """
        components = self.compute_component_analysis(
            trust_matrix, balance_matrix, rate_matrix, converters, node_to_idx, tau
        )
        
        if not components:
            return {}
        
        from scipy.stats import spearmanr
        
        # Extract normalized components
        structure = components['structure_normalized']
        flow = components['flow_normalized']
        rates = components['rates_normalized']
        supply = components['supply_normalized']
        
        correlations = {}
        
        # Compute all pairwise correlations
        pairs = [
            ('structure_flow', structure, flow),
            ('structure_rates', structure, rates),
            ('structure_supply', structure, supply),
            ('flow_rates', flow, rates),
            ('flow_supply', flow, supply),
            ('rates_supply', rates, supply)
        ]
        
        for name, x, y in pairs:
            if len(x) > 1 and len(y) > 1:
                corr, p_value = spearmanr(x, y)
                correlations[name] = {
                    'correlation': float(corr) if not np.isnan(corr) else 0.0,
                    'p_value': float(p_value) if not np.isnan(p_value) else 1.0
                }
            else:
                correlations[name] = {'correlation': 0.0, 'p_value': 1.0}
        
        return correlations
    
    def optimize_component_weights(self, trust_matrix: sp.csr_matrix,
                                 balance_matrix: sp.csr_matrix,
                                 rate_matrix: np.ndarray,
                                 converters: List[str],
                                 node_to_idx: Dict[str, int],
                                 tau: float,
                                 target_scores: np.ndarray) -> Dict[str, float]:
        """
        Optimize component weights to best match target scores.
        
        Args:
            target_scores: Ground truth or desired scores for optimization
            
        Returns:
            Optimized weights dictionary
        """
        components = self.compute_component_analysis(
            trust_matrix, balance_matrix, rate_matrix, converters, node_to_idx, tau
        )
        
        if not components:
            return {'conductance': 0.25, 'flow': 0.25, 'rates': 0.30, 'supplies': 0.20}
        
        # Extract normalized components
        X = np.column_stack([
            components['structure_normalized'],
            components['flow_normalized'], 
            components['rates_normalized'],
            components['supply_normalized']
        ])
        
        # Solve least squares with constraint that weights sum to 1
        from scipy.optimize import minimize
        
        def objective(weights):
            predicted = X @ weights
            return np.mean((predicted - target_scores) ** 2)
        
        def constraint(weights):
            return weights.sum() - 1.0
        
        # Initial guess
        x0 = np.array([0.25, 0.25, 0.30, 0.20])
        
        # Bounds: all weights between 0 and 1
        bounds = [(0, 1) for _ in range(4)]
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': constraint}
        
        result = minimize(objective, x0, bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            return {
                'conductance': float(optimal_weights[0]),
                'flow': float(optimal_weights[1]),
                'rates': float(optimal_weights[2]),
                'supplies': float(optimal_weights[3])
            }
        else:
            logger.warning("Weight optimization failed, returning default weights")
            return {'conductance': 0.25, 'flow': 0.25, 'rates': 0.30, 'supplies': 0.20}