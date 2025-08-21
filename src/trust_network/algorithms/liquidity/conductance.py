"""
Balance-weighted conductance algorithm implementing the theoretical framework.

Implements balance-weighted conductance:
φ_B(S) = cut_B(S, S̄) / min(vol_B(S), vol_B(S̄))

Where:
- cut_B(S, S̄) = Σ_{u∈S, v∈S̄} W[u,v] * f(B[u,v])
- vol_B(S) = Σ_{u∈S} Σ_v W[u,v] * f(B[u,v])
- f(b) = 1 + log(1 + b) is the balance weighting function
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Set, Tuple
import logging

from ..base import LiquidityAlgorithm

logger = logging.getLogger(__name__)


class ConductanceAlgorithm(LiquidityAlgorithm):
    """Compute liquidity scores based on balance-weighted graph conductance."""
    
    def __init__(self, config):
        super().__init__(config)
        self.k_hops = getattr(config, 'k_hops', 2)
        # Balance weighting parameters
        self.balance_weight_scale = getattr(config, 'balance_weight_scale', 1.0)
        self.min_balance_weight = getattr(config, 'min_balance_weight', 0.01)
    
    def compute(self, trust_matrix: sp.csr_matrix,
                balance_matrix: sp.csr_matrix,
                rate_matrix: np.ndarray,
                converters: List[str],
                node_to_idx: Dict[str, int],
                tau: float,
                **kwargs) -> np.ndarray:
        """Compute balance-weighted conductance-based liquidity scores."""
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        logger.debug(f"Computing balance-weighted conductance for {n} nodes")
        
        # Convert sparse matrices to dense for easier computation
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        
        # Compute balance-weighted conductance for each node
        conductances = np.zeros(n)
        
        for i in range(n):
            conductances[i] = self._compute_balance_weighted_conductance(
                W_dense, B_dense, i, self.k_hops
            )
        
        # Compute additional components
        effective_rates = self._compute_effective_rates(
            trust_matrix, rate_matrix, converters, node_to_idx, tau
        )
        
        token_supplies = np.asarray(balance_matrix.sum(axis=0)).ravel()
        
        # Structural liquidity: inverse of conductance (lower conductance = higher liquidity)
        structure_scores = 1 - conductances
        
        # Normalize components
        structure_norm = self._normalize_array(structure_scores)
        rates_norm = self._normalize_array(effective_rates)
        supplies_norm = self._normalize_array(token_supplies)
        
        # Weighted combination emphasizing structural properties
        liquidity_scores = (
            0.5 * structure_norm +      # Structural conductance
            0.3 * rates_norm +          # Conversion capability
            0.2 * supplies_norm         # Token availability
        )
        
        return liquidity_scores
    
    def _compute_balance_weighted_conductance(self, W: np.ndarray, B: np.ndarray, 
                                            node_idx: int, k_hops: int) -> float:
        """
        Compute balance-weighted conductance around a node using k-hop neighborhood.
        
        Implements: φ_B(S) = cut_B(S, S̄) / min(vol_B(S), vol_B(S̄))
        
        Args:
            W: Trust adjacency matrix (dense)
            B: Balance matrix (dense)
            node_idx: Center node for neighborhood
            k_hops: Radius of neighborhood
            
        Returns:
            Balance-weighted conductance value
        """
        n = W.shape[0]
        
        # Find k-hop neighborhood using BFS
        neighborhood = self._get_k_hop_neighborhood(W, node_idx, k_hops)
        
        if len(neighborhood) >= n - 1:  # Almost whole graph
            return 0.0
        
        # Compute balance-weighted cut and volumes
        cut_B = 0.0
        vol_B_S = 0.0
        
        # For each node in neighborhood S
        for u in neighborhood:
            for v in range(n):
                if W[u, v] > 0:
                    # Balance weighting function: f(b) = 1 + log(1 + b)
                    balance_weight = self._balance_weighting_function(B[u, v])
                    edge_weight = W[u, v] * balance_weight
                    
                    # Add to volume of S
                    vol_B_S += edge_weight
                    
                    # Add to cut if v is not in S
                    if v not in neighborhood:
                        cut_B += edge_weight
        
        # Compute total volume for complement
        total_vol_B = 0.0
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    balance_weight = self._balance_weighting_function(B[i, j])
                    total_vol_B += W[i, j] * balance_weight
        
        vol_B_S_bar = total_vol_B - vol_B_S
        
        # Conductance formula
        min_vol = min(vol_B_S, vol_B_S_bar)
        if min_vol <= 0:
            return 1.0  # Maximum conductance (worst case)
        
        conductance = cut_B / min_vol
        
        # Ensure conductance is in [0, 1]
        return min(1.0, conductance)
    
    def _balance_weighting_function(self, balance: float) -> float:
        """
        Balance weighting function: f(b) = 1 + log(1 + b)
        
        This gives higher weight to edges where the source holds more tokens
        of the target, indicating higher liquidity potential.
        """
        if balance < 0:
            balance = 0
        
        # Scale balance for numerical stability
        scaled_balance = balance * self.balance_weight_scale
        
        # f(b) = 1 + log(1 + b)
        weight = 1.0 + np.log(1.0 + scaled_balance)
        
        # Ensure minimum weight
        return max(weight, self.min_balance_weight)
    
    def _get_k_hop_neighborhood(self, W: np.ndarray, center: int, k: int) -> Set[int]:
        """
        Get k-hop neighborhood of a node using BFS.
        
        Args:
            W: Trust adjacency matrix
            center: Center node
            k: Number of hops
            
        Returns:
            Set of nodes in k-hop neighborhood
        """
        neighborhood = {center}
        frontier = {center}
        
        for hop in range(k):
            new_frontier = set()
            
            for u in frontier:
                # Find neighbors: nodes that trust u or u trusts
                for v in range(W.shape[0]):
                    if (W[v, u] > 0 or W[u, v] > 0) and v not in neighborhood:
                        new_frontier.add(v)
            
            if not new_frontier:
                break
            
            frontier = new_frontier
            neighborhood.update(frontier)
        
        return neighborhood
    
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
            
            # Compute trust from all nodes to converter
            trust_to_converter = self._compute_trust_to_node(trust_matrix, conv_idx, tau)
            
            # For each token, check if converter can receive it
            for j in range(n):
                if j < rate_matrix.shape[1] and trust_to_converter[j] >= tau:
                    rate = rate_matrix[i, j]
                    effective_rates[j] = max(effective_rates[j], rate)
        
        return effective_rates
    
    def _compute_trust_to_node(self, trust_matrix: sp.csr_matrix, 
                             target: int, tau: float) -> np.ndarray:
        """
        Compute trust from all nodes to a target node.
        
        Uses BFS to find which nodes can reach the target through trust paths.
        """
        n = trust_matrix.shape[0]
        trust_scores = np.zeros(n)
        
        # BFS from target backwards to find who can reach target
        visited = {target}
        queue = [(target, 1.0)]  # (node, trust_value)
        trust_scores[target] = 1.0
        
        while queue:
            current, current_trust = queue.pop(0)
            
            # Find nodes that trust current (can send to current)
            col = trust_matrix.getcol(current)
            for idx in range(col.nnz):
                source = col.indices[idx]
                trust_weight = col.data[idx]
                
                if source not in visited and trust_weight >= tau:
                    # Trust propagates backwards
                    new_trust = current_trust * trust_weight
                    
                    if new_trust >= tau:
                        trust_scores[source] = max(trust_scores[source], new_trust)
                        visited.add(source)
                        queue.append((source, new_trust))
        
        return trust_scores
    
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range with min-max normalization."""
        arr = np.array(arr)
        if len(arr) == 0:
            return arr
        
        min_val = arr.min()
        max_val = arr.max()
        
        if max_val - min_val < 1e-10:
            return np.ones_like(arr) * 0.5  # All equal, return middle value
        
        # Min-max normalization
        normalized = (arr - min_val) / (max_val - min_val)
        
        return normalized
    
    def compute_network_conductance_distribution(self, trust_matrix: sp.csr_matrix,
                                               balance_matrix: sp.csr_matrix) -> Dict[str, float]:
        """
        Compute conductance distribution statistics for the entire network.
        
        Returns:
            Dictionary with conductance statistics
        """
        n = trust_matrix.shape[0]
        if n == 0:
            return {}
        
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        
        conductances = []
        for i in range(n):
            cond = self._compute_balance_weighted_conductance(W_dense, B_dense, i, self.k_hops)
            conductances.append(cond)
        
        conductances = np.array(conductances)
        
        return {
            'mean_conductance': float(np.mean(conductances)),
            'std_conductance': float(np.std(conductances)),
            'min_conductance': float(np.min(conductances)),
            'max_conductance': float(np.max(conductances)),
            'median_conductance': float(np.median(conductances)),
            'low_conductance_nodes': int(np.sum(conductances < 0.1)),  # Well-connected nodes
            'high_conductance_nodes': int(np.sum(conductances > 0.8)),  # Poorly connected nodes
        }
    
    def find_best_conductance_nodes(self, trust_matrix: sp.csr_matrix,
                                  balance_matrix: sp.csr_matrix, 
                                  top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find nodes with best (lowest) conductance values.
        
        Low conductance indicates good network position for liquidity.
        """
        n = trust_matrix.shape[0]
        if n == 0:
            return []
        
        W_dense = trust_matrix.toarray()
        B_dense = balance_matrix.toarray()
        
        node_conductances = []
        for i in range(n):
            cond = self._compute_balance_weighted_conductance(W_dense, B_dense, i, self.k_hops)
            node_conductances.append((i, cond))
        
        # Sort by conductance (ascending - lower is better)
        node_conductances.sort(key=lambda x: x[1])
        
        return node_conductances[:top_k]