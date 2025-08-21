"""
Balance-weighted flow centrality algorithm implementing the theoretical framework.

Implements balance-weighted random walks where transition probabilities are:
P_B(i → j) = W[j,i] * g(B[i,j]) / Σ_k W[k,i] * g(B[i,k])

Where g(b) is the balance weighting function that prefers edges where 
the walker holds tokens of the target node.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any
import logging

from ..base import LiquidityAlgorithm

logger = logging.getLogger(__name__)


class FlowCentralityAlgorithm(LiquidityAlgorithm):
    """Compute liquidity scores using balance-weighted flow centrality via random walks."""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_walks = getattr(config, 'num_walks', 10000)
        self.walk_length = getattr(config, 'walk_length', 20)
        # Balance weighting parameters
        self.balance_boost_factor = getattr(config, 'balance_boost_factor', 2.0)
        self.min_transition_prob = getattr(config, 'min_transition_prob', 0.001)
    
    def compute(self, trust_matrix: sp.csr_matrix,
                balance_matrix: sp.csr_matrix,
                rate_matrix: np.ndarray,
                converters: List[str],
                node_to_idx: Dict[str, int],
                tau: float,
                **kwargs) -> np.ndarray:
        """Compute balance-weighted flow centrality-based liquidity scores."""
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        logger.debug(f"Computing balance-weighted flow centrality for {n} nodes")
        
        # Compute balance-weighted flow centrality
        flow_centrality = self._compute_balance_weighted_flow_centrality(
            trust_matrix, balance_matrix
        )
        
        # Compute effective rates
        effective_rates = self._compute_effective_rates(
            trust_matrix, rate_matrix, converters, node_to_idx, tau
        )
        
        # Token supplies
        token_supplies = np.asarray(balance_matrix.sum(axis=0)).ravel()
        
        # Compute balance-weighted stationary distribution
        stationary_dist = self._compute_balance_weighted_stationary_distribution(
            trust_matrix, balance_matrix
        )
        
        # Normalize components
        flow_norm = self._normalize_array(flow_centrality)
        rates_norm = self._normalize_array(effective_rates)
        supplies_norm = self._normalize_array(token_supplies)
        stationary_norm = self._normalize_array(stationary_dist)
        
        # Weighted combination emphasizing flow properties
        liquidity_scores = (
            0.4 * flow_norm +           # Random walk centrality
            0.2 * stationary_norm +     # Stationary importance  
            0.25 * rates_norm +         # Conversion capability
            0.15 * supplies_norm        # Token availability
        )
        
        return liquidity_scores
    
    def _compute_balance_weighted_flow_centrality(self, trust_matrix: sp.csr_matrix,
                                                balance_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute flow centrality using balance-weighted random walks.
        
        Key insight: Walks prefer edges where the walker holds tokens of the target node,
        as this represents higher payment capacity.
        """
        n = trust_matrix.shape[0]
        visit_counts = np.zeros(n, dtype=np.float64)
        
        if n == 0:
            return visit_counts
        
        # Convert to dense for easier access during walks
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        
        # Compute average balance for normalization
        avg_balance = B_dense.mean() if B_dense.sum() > 0 else 1.0
        
        logger.debug(f"Starting {self.num_walks} balance-weighted random walks")
        
        for walk_num in range(self.num_walks):
            # Start at random node
            current = np.random.randint(n)
            
            for step in range(self.walk_length):
                # Count visit
                visit_counts[current] += 1.0
                
                # Get balance-weighted transition probabilities
                neighbors, transition_probs = self._get_balance_weighted_transitions(
                    current, W_dense, B_dense, avg_balance
                )
                
                if len(neighbors) == 0:
                    # Dead end - restart randomly
                    current = np.random.randint(n)
                    continue
                
                # Sample next node based on balance-weighted probabilities
                next_node = np.random.choice(neighbors, p=transition_probs)
                current = next_node
        
        # Normalize by total visits
        total_visits = visit_counts.sum()
        if total_visits > 0:
            visit_counts = visit_counts / total_visits
        
        logger.debug(f"Completed random walks, total visits: {total_visits}")
        
        return visit_counts
    
    def _get_balance_weighted_transitions(self, current: int, W: np.ndarray, 
                                        B: np.ndarray, avg_balance: float) -> Tuple[List[int], np.ndarray]:
        """
        Get balance-weighted transition probabilities from current node.
        
        Implements: P_B(current → j) = W[j,current] * g(B[current,j]) / normalization
        
        Where g(b) gives higher weight when current node holds more tokens of target j.
        """
        n = W.shape[0]
        neighbors = []
        weights = []
        
        # Find all nodes that trust current node (can receive from current)
        for target in range(n):
            trust_weight = W[target, current]  # target trusts current
            
            if trust_weight > 0:
                # Balance weighting: how much of target's tokens does current hold?
                balance = B[current, target]
                
                # Balance weighting function g(b)
                if balance > 0:
                    # Higher balance → higher probability
                    balance_weight = 1.0 + self.balance_boost_factor * np.sqrt(balance / avg_balance)
                else:
                    # Small weight for zero balance (maintains connectivity)
                    balance_weight = self.min_transition_prob
                
                # Combined weight
                total_weight = trust_weight * balance_weight
                
                neighbors.append(target)
                weights.append(total_weight)
        
        if len(neighbors) == 0:
            return [], np.array([])
        
        # Normalize to probabilities
        weights = np.array(weights)
        total_weight = weights.sum()
        
        if total_weight > 0:
            transition_probs = weights / total_weight
        else:
            # Uniform if all weights are zero
            transition_probs = np.ones(len(neighbors)) / len(neighbors)
        
        return neighbors, transition_probs
    
    def _compute_balance_weighted_stationary_distribution(self, trust_matrix: sp.csr_matrix,
                                                        balance_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute stationary distribution of balance-weighted Markov chain.
        
        This gives the long-term probability of being at each node under
        balance-weighted random walks.
        """
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        # Build balance-weighted transition matrix
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        avg_balance = B_dense.mean() if B_dense.sum() > 0 else 1.0
        
        P_B = np.zeros((n, n))
        
        for i in range(n):
            neighbors, transition_probs = self._get_balance_weighted_transitions(
                i, W_dense, B_dense, avg_balance
            )
            
            for j, prob in zip(neighbors, transition_probs):
                P_B[i, j] = prob
        
        # Handle rows with no outgoing edges (add self-loops)
        row_sums = P_B.sum(axis=1)
        zero_rows = (row_sums == 0)
        P_B[zero_rows, zero_rows] = 1.0
        
        # Compute stationary distribution using power iteration
        stationary = np.ones(n) / n  # Start uniform
        
        for iteration in range(1000):  # Max iterations
            new_stationary = P_B.T @ stationary
            
            # Check convergence
            if np.linalg.norm(new_stationary - stationary, ord=1) < 1e-9:
                break
            
            stationary = new_stationary
        
        # Normalize
        if stationary.sum() > 0:
            stationary = stationary / stationary.sum()
        
        return stationary
    
    def _compute_effective_rates(self, trust_matrix: sp.csr_matrix,
                               rate_matrix: np.ndarray,
                               converters: List[str],
                               node_to_idx: Dict[str, int],
                               tau: float) -> np.ndarray:
        """Compute effective conversion rates considering trust paths."""
        n = trust_matrix.shape[0]
        effective_rates = np.zeros(n)
        
        if len(converters) == 0 or rate_matrix.size == 0:
            return effective_rates
        
        converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
        
        for i, conv_idx in enumerate(converter_indices):
            if i >= rate_matrix.shape[0]:
                continue
            
            # Simple trust propagation to find reachable tokens
            reachable = self._find_reachable_tokens(trust_matrix, conv_idx, tau)
            
            # Apply rates for reachable tokens
            for j in range(n):
                if reachable[j] and j < rate_matrix.shape[1]:
                    rate = rate_matrix[i, j]
                    effective_rates[j] = max(effective_rates[j], rate)
        
        return effective_rates
    
    def _find_reachable_tokens(self, trust_matrix: sp.csr_matrix, 
                             converter: int, tau: float) -> np.ndarray:
        """Find which tokens can reach the converter through trust paths."""
        n = trust_matrix.shape[0]
        reachable = np.zeros(n, dtype=bool)
        
        # BFS from converter backwards
        visited = {converter}
        queue = [converter]
        reachable[converter] = True
        
        while queue:
            current = queue.pop(0)
            
            # Find nodes that trust current (can send to current)
            col = trust_matrix.getcol(current)
            for idx in range(col.nnz):
                source = col.indices[idx]
                trust_weight = col.data[idx]
                
                if source not in visited and trust_weight >= tau:
                    reachable[source] = True
                    visited.add(source)
                    queue.append(source)
        
        return reachable
    
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr = np.array(arr)
        if len(arr) == 0:
            return arr
        
        min_val = arr.min()
        max_val = arr.max()
        
        if max_val - min_val < 1e-10:
            return np.ones_like(arr) * 0.5
        
        return (arr - min_val) / (max_val - min_val)
    
    def analyze_walk_statistics(self, trust_matrix: sp.csr_matrix,
                               balance_matrix: sp.csr_matrix, 
                               num_sample_walks: int = 1000) -> Dict[str, any]:
        """
        Analyze statistics of balance-weighted random walks.
        
        Returns detailed statistics about walk behavior.
        """
        n = trust_matrix.shape[0]
        if n == 0:
            return {}
        
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        avg_balance = B_dense.mean() if B_dense.sum() > 0 else 1.0
        
        # Track walk statistics
        walk_lengths = []
        transitions_used = []
        balance_weighted_transitions = 0
        zero_balance_transitions = 0
        
        for _ in range(num_sample_walks):
            current = np.random.randint(n)
            walk_length = 0
            
            for step in range(self.walk_length):
                neighbors, transition_probs = self._get_balance_weighted_transitions(
                    current, W_dense, B_dense, avg_balance
                )
                
                if len(neighbors) == 0:
                    break
                
                # Check if this transition used balance weighting
                next_node = np.random.choice(neighbors, p=transition_probs)
                balance = B_dense[current, next_node]
                
                if balance > 0:
                    balance_weighted_transitions += 1
                else:
                    zero_balance_transitions += 1
                
                transitions_used.extend(neighbors)
                current = next_node
                walk_length += 1
            
            walk_lengths.append(walk_length)
        
        total_transitions = balance_weighted_transitions + zero_balance_transitions
        
        return {
            'avg_walk_length': np.mean(walk_lengths) if walk_lengths else 0,
            'std_walk_length': np.std(walk_lengths) if walk_lengths else 0,
            'balance_weighted_ratio': balance_weighted_transitions / total_transitions if total_transitions > 0 else 0,
            'unique_transitions_used': len(set(transitions_used)),
            'total_possible_transitions': np.sum(W_dense > 0),
            'transition_coverage': len(set(transitions_used)) / np.sum(W_dense > 0) if np.sum(W_dense > 0) > 0 else 0
        }