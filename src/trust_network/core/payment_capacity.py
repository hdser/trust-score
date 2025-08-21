"""
Payment capacity calculation implementing the theoretical framework.

This module implements the core payment capacity formula:
Capacity(s → t, T(token)) = min{B[v_i, token] · 1[W[v_{i+1}, token] ≥ τ]}

Key principles:
1. Trust creates acceptance: W[i,j] = 1 means i accepts tokens from j
2. Balances enable flow: B[i,j] = amount of token T(j) held by node i
3. Payment requires both trust path AND sufficient balances
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PaymentCapacityCalculator:
    """Calculate payment capacities with proper token flow logic."""
    
    def __init__(self, trust_matrix: sp.csr_matrix, balance_matrix: sp.csr_matrix, 
                 tau: float = 0.5):
        """
        Initialize payment capacity calculator.
        
        Args:
            trust_matrix: W[i,j] = trust from i to j (i accepts tokens from j)
            balance_matrix: B[i,j] = amount of token T(j) held by node i
            tau: Trust acceptance threshold
        """
        self.W = trust_matrix
        self.B = balance_matrix
        self.tau = tau
        self.n = trust_matrix.shape[0]
    
    def compute_payment_capacity(self, source: int, target: int, 
                               token_issuer: int) -> Tuple[float, List[int], str]:
        """
        Compute payment capacity for specific token flow.
        
        Implementation of theoretical formula:
        Capacity(s → t, T(token)) = min{B[v_i, token] · 1[W[v_{i+1}, token] ≥ τ]}
        
        Args:
            source: Payment source node
            target: Payment target node  
            token_issuer: Node that issued the token being sent
            
        Returns:
            (capacity, path, failure_reason)
        """
        # Find trust path from source to target
        path_indices = self._find_trust_path(source, target)
        
        if not path_indices:
            return 0.0, [], "No trust path exists"
        
        # Check token acceptance along path
        for i in range(len(path_indices) - 1):
            current = path_indices[i]
            next_node = path_indices[i + 1]
            
            # Critical: next_node must trust token_issuer to accept the token
            if self.W[next_node, token_issuer] < self.tau:
                return 0.0, path_indices, f"Node {next_node} doesn't accept tokens from {token_issuer}"
        
        # Calculate capacity along path
        capacity = float('inf')
        bottleneck_node = -1
        
        # Source must have the token to send
        source_balance = self.B[source, token_issuer]
        if source_balance <= 0:
            return 0.0, path_indices, f"Source {source} has no tokens from {token_issuer}"
        
        capacity = min(capacity, source_balance)
        if capacity == source_balance:
            bottleneck_node = source
        
        # Check intermediate nodes' balances for forwarding
        for i in range(1, len(path_indices) - 1):  # Skip source and target
            current = path_indices[i]
            
            # Intermediate must have token to forward
            balance = self.B[current, token_issuer]
            if balance <= 0:
                return 0.0, path_indices, f"Intermediate {current} has no tokens from {token_issuer} to forward"
            
            if balance < capacity:
                capacity = balance
                bottleneck_node = current
        
        return capacity, path_indices, f"Success (bottleneck at node {bottleneck_node})"
    
    def _find_trust_path(self, source: int, target: int) -> Optional[List[int]]:
        """
        Find trust path using BFS on the trust graph.
        
        Trust path exists if there's a sequence of nodes where each trusts the previous.
        """
        if source == target:
            return [source]
        
        visited = {source}
        queue = deque([(source, [source])])
        
        while queue:
            current, path = queue.popleft()
            
            # Check all nodes that trust current (can receive from current)
            for next_node in range(self.n):
                if (self.W[next_node, current] >= self.tau and 
                    next_node not in visited):
                    
                    new_path = path + [next_node]
                    
                    if next_node == target:
                        return new_path
                    
                    visited.add(next_node)
                    queue.append((next_node, new_path))
        
        return None
    
    def compute_all_payment_capacities(self, token_issuer: int) -> np.ndarray:
        """
        Compute payment capacities from all nodes to all nodes for specific token.
        
        Returns:
            n x n matrix where entry [i,j] is capacity from i to j using token_issuer's token
        """
        capacity_matrix = np.zeros((self.n, self.n))
        
        for source in range(self.n):
            for target in range(self.n):
                if source != target:
                    capacity, _, _ = self.compute_payment_capacity(source, target, token_issuer)
                    capacity_matrix[source, target] = capacity
        
        return capacity_matrix
    
    def analyze_payment_bottlenecks(self) -> Dict[str, List]:
        """
        Comprehensive bottleneck analysis for payment network.
        
        Returns:
            Dictionary with different types of bottlenecks identified
        """
        bottlenecks = {
            'no_self_tokens': [],       # Nodes with no self-issued tokens
            'trust_isolated': [],       # Nodes with no trust connections
            'balance_bottlenecks': [],  # Nodes blocking payments due to low balances
            'trust_gaps': [],          # Missing trust relationships blocking payments
            'token_concentration': {}   # Tokens concentrated in few nodes
        }
        
        # 1. Check for nodes with no self-tokens
        for node in range(self.n):
            if self.B[node, node] <= 0:
                bottlenecks['no_self_tokens'].append(node)
        
        # 2. Check for trust-isolated nodes
        for node in range(self.n):
            incoming_trust = (self.W[node, :] >= self.tau).sum()
            outgoing_trust = (self.W[:, node] >= self.tau).sum()
            
            if incoming_trust == 0 and outgoing_trust == 0:
                bottlenecks['trust_isolated'].append(node)
        
        # 3. Analyze balance bottlenecks
        for token_issuer in range(self.n):
            # Find nodes that could forward this token but have low balances
            potential_forwarders = []
            
            for node in range(self.n):
                if node != token_issuer:
                    # Check if node is on any payment path for this token
                    is_on_path = False
                    for source in range(self.n):
                        for target in range(self.n):
                            if source != target and source != node and target != node:
                                path = self._find_trust_path(source, target)
                                if path and node in path:
                                    is_on_path = True
                                    break
                        if is_on_path:
                            break
                    
                    if is_on_path and self.B[node, token_issuer] < self.B[token_issuer, token_issuer] * 0.1:
                        potential_forwarders.append({
                            'node': node,
                            'token_issuer': token_issuer,
                            'balance': self.B[node, token_issuer],
                            'issuer_balance': self.B[token_issuer, token_issuer]
                        })
            
            if potential_forwarders:
                bottlenecks['balance_bottlenecks'].extend(potential_forwarders)
        
        # 4. Find trust gaps blocking payments
        for source in range(self.n):
            for target in range(self.n):
                if source != target:
                    # Try payment with source's token
                    capacity, path, reason = self.compute_payment_capacity(source, target, source)
                    
                    if capacity == 0 and "doesn't accept tokens" in reason:
                        bottlenecks['trust_gaps'].append({
                            'source': source,
                            'target': target,
                            'path': path,
                            'reason': reason
                        })
        
        # 5. Analyze token concentration
        for token_issuer in range(self.n):
            token_holders = []
            total_supply = self.B[:, token_issuer].sum()
            
            if total_supply > 0:
                for holder in range(self.n):
                    balance = self.B[holder, token_issuer]
                    if balance > 0:
                        token_holders.append({
                            'holder': holder,
                            'balance': balance,
                            'percentage': balance / total_supply
                        })
                
                # Sort by balance
                token_holders.sort(key=lambda x: x['balance'], reverse=True)
                
                # Check if top 20% of holders control 80% of tokens (Pareto principle)
                top_20_percent = max(1, len(token_holders) // 5)
                top_balance = sum(h['balance'] for h in token_holders[:top_20_percent])
                concentration_ratio = top_balance / total_supply if total_supply > 0 else 0
                
                bottlenecks['token_concentration'][token_issuer] = {
                    'holders': token_holders,
                    'concentration_ratio': concentration_ratio,
                    'is_concentrated': concentration_ratio > 0.8
                }
        
        return bottlenecks
    
    def compute_network_liquidity_score(self) -> float:
        """
        Compute overall network liquidity score based on payment capacities.
        
        Returns:
            Score between 0 and 1 indicating network payment liquidity
        """
        total_possible_payments = self.n * (self.n - 1)
        successful_payments = 0
        total_capacity = 0
        
        for source in range(self.n):
            for target in range(self.n):
                if source != target:
                    # Try payment with source's own token
                    capacity, _, _ = self.compute_payment_capacity(source, target, source)
                    
                    if capacity > 0:
                        successful_payments += 1
                        total_capacity += capacity
        
        # Connectivity component
        connectivity = successful_payments / total_possible_payments if total_possible_payments > 0 else 0
        
        # Average capacity component (normalized)
        avg_capacity = total_capacity / successful_payments if successful_payments > 0 else 0
        max_possible_capacity = self.B.diagonal().mean()  # Average self-token balance
        normalized_capacity = min(1.0, avg_capacity / max_possible_capacity) if max_possible_capacity > 0 else 0
        
        # Combined score
        liquidity_score = 0.7 * connectivity + 0.3 * normalized_capacity
        
        return liquidity_score
    
    def get_payment_paths_for_token(self, token_issuer: int, min_capacity: float = 0) -> List[Dict]:
        """
        Get all viable payment paths for a specific token.
        
        Args:
            token_issuer: The token to analyze
            min_capacity: Minimum capacity threshold
            
        Returns:
            List of viable payment paths with capacities
        """
        viable_paths = []
        
        for source in range(self.n):
            for target in range(self.n):
                if source != target:
                    capacity, path, reason = self.compute_payment_capacity(source, target, token_issuer)
                    
                    if capacity >= min_capacity:
                        viable_paths.append({
                            'source': source,
                            'target': target,
                            'token_issuer': token_issuer,
                            'capacity': capacity,
                            'path': path,
                            'hops': len(path) - 1,
                            'reason': reason
                        })
        
        # Sort by capacity descending
        viable_paths.sort(key=lambda x: x['capacity'], reverse=True)
        
        return viable_paths