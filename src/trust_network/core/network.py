"""Main TrustNetwork class - orchestrates trust score computation."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from collections import defaultdict
import time
import logging
import hashlib

from .data_models import (
    NodeInfo, TrustEdge, TokenBalance, ConversionRate, TrustPath,
    TrustScoreResult, NodeTrustScore, NetworkStatistics
)
from ..algorithms.social import get_social_algorithm
from ..algorithms.liquidity import get_liquidity_algorithm
from ..utils.matrix_ops import MatrixBuilder
from ..utils.cache import TrustCache
from ..config.settings import NetworkConfig

logger = logging.getLogger(__name__)


class TrustNetwork:
    """
    Main class for managing and analyzing a decentralized token network.
    
    This class orchestrates trust score computation by coordinating
    various algorithms and managing network data.
    """
    
    def __init__(self, config: NetworkConfig):
        """Initialize the trust network with given configuration."""
        self.config = config
        
        # Core data structures
        self.nodes: Dict[str, NodeInfo] = {}
        self.edges: List[TrustEdge] = []
        self.balances: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.converters: Set[str] = set()
        self.rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Index mappings
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
        # Matrix builder and cache
        self.matrix_builder = MatrixBuilder()
        self.cache = TrustCache(
            enabled=config.trust_network.performance.enable_caching,
            max_size=config.trust_network.performance.cache_size
        )
        
        # Matrices (lazily computed)
        self._matrices_built = False
        self._W: Optional[sp.csr_matrix] = None  # Trust adjacency
        self._B: Optional[sp.csr_matrix] = None  # Balance matrix
        self._R: Optional[np.ndarray] = None     # Rate matrix
        self._network_hash: Optional[str] = None
        
        logger.info(f"TrustNetwork initialized with {config.trust_network.algorithms.social} + {config.trust_network.algorithms.liquidity}")
    
    def add_node(self, label: str, is_converter: bool = False, 
                 token_symbol: Optional[str] = None, **metadata) -> None:
        """Add a node to the network."""
        if label in self.nodes:
            logger.warning(f"Node {label} already exists, updating metadata")
            self.nodes[label].metadata.update(metadata)
            self.nodes[label].last_updated = time.time()
            return
        
        idx = len(self.nodes)
        self.nodes[label] = NodeInfo(
            label=label,
            index=idx,
            is_converter=is_converter,
            token_symbol=token_symbol or f"T_{label}",
            metadata=metadata
        )
        
        self.node_to_idx[label] = idx
        self.idx_to_node[idx] = label
        
        if is_converter:
            self.converters.add(label)
        
        self._invalidate_caches()
        logger.debug(f"Added node {label} (converter={is_converter})")
    
    def add_edge(self, source: str, target: str, weight: float, **metadata) -> None:
        """Add a trust edge to the network."""
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        # Check for existing edge and update
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                logger.debug(f"Updating edge {source}->{target}: {edge.weight} -> {weight}")
                edge.weight = weight
                edge.metadata.update(metadata)
                self._invalidate_caches()
                return
        
        # Add new edge
        edge = TrustEdge(source, target, weight, metadata=metadata)
        self.edges.append(edge)
        
        self._invalidate_caches()
        logger.debug(f"Added edge {source}->{target} with weight {weight}")
    
    def set_balance(self, holder: str, token: str, amount: float) -> None:
        """Set the balance of a specific token for a holder."""
        if holder not in self.nodes:
            self.add_node(holder)
        if token not in self.nodes:
            self.add_node(token)
        
        if amount > 0:
            self.balances[holder][token] = amount
        elif token in self.balances[holder]:
            del self.balances[holder][token]
        
        self._invalidate_caches()
        logger.debug(f"Set balance: {holder} holds {amount} of {token}")
    
    def set_conversion_rate(self, converter: str, token: str, rate: float) -> None:
        """Set the conversion rate for a token at a converter."""
        if converter not in self.converters:
            raise ValueError(f"{converter} is not registered as a converter")
        
        if token not in self.nodes:
            self.add_node(token)
        
        self.rates[converter][token] = rate
        self._invalidate_caches()
        logger.debug(f"Set rate: {converter} converts {token} at {rate}")
    
    def load_data(self, data: Dict[str, Any]) -> None:
        """Load network data from dictionary."""
        # Load nodes
        for node_data in data.get("nodes", []):
            self.add_node(
                label=node_data["label"],
                is_converter=node_data.get("is_converter", False),
                token_symbol=node_data.get("token_symbol"),
                **node_data.get("metadata", {})
            )
        
        # Load edges
        for edge_data in data.get("edges", []):
            self.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                weight=edge_data["weight"],
                **edge_data.get("metadata", {})
            )
        
        # Load balances
        for balance_data in data.get("balances", []):
            self.set_balance(
                holder=balance_data["holder"],
                token=balance_data["token"],
                amount=balance_data["amount"]
            )
        
        # Load rates
        for rate_data in data.get("rates", []):
            self.set_conversion_rate(
                converter=rate_data["converter"],
                token=rate_data["token"],
                rate=rate_data["rate"]
            )
        
        logger.info(f"Loaded data: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _build_matrices(self) -> None:
        """Build sparse matrix representations of the network."""
        if self._matrices_built and self._network_hash == self._compute_network_hash():
            return
        
        n = len(self.nodes)
        if n == 0:
            raise ValueError("Cannot build matrices for empty network")
        
        # Build trust adjacency matrix W
        self._W = self.matrix_builder.build_trust_matrix(
            self.edges, self.node_to_idx, n
        )
        
        # Build balance matrix B
        self._B = self.matrix_builder.build_balance_matrix(
            self.balances, self.node_to_idx, n
        )
        
        # Build rate matrix R
        self._R = self.matrix_builder.build_rate_matrix(
            self.rates, self.converters, self.node_to_idx, n
        )
        
        self._matrices_built = True
        self._network_hash = self._compute_network_hash()
        
        logger.info(f"Built matrices: W={self._W.shape}, B={self._B.shape}, R={self._R.shape}")
    
    def compute_trust_scores(self, force_recompute: bool = False) -> TrustScoreResult:
        """Compute comprehensive trust scores for all nodes."""
        # Check cache
        cache_key = self._compute_network_hash()
        if not force_recompute:
            cached_result = self.cache.get_scores(cache_key)
            if cached_result is not None:
                logger.info("Returning cached trust scores")
                return cached_result
        
        start_time = time.time()
        
        # Build matrices if needed
        self._build_matrices()
        
        n = len(self.nodes)
        if n == 0:
            raise ValueError("Cannot compute scores for empty network")
        
        logger.info(f"Computing trust scores for {n} nodes")
        
        # Get algorithm instances
        social_alg = get_social_algorithm(
            self.config.trust_network.algorithms.social,
            self.config.trust_network.parameters
        )
        
        liquidity_alg = get_liquidity_algorithm(
            self.config.trust_network.algorithms.liquidity,
            self.config.trust_network.parameters
        )
        
        # Compute social scores
        logger.debug(f"Computing social scores using {social_alg.__class__.__name__}")
        social_raw = social_alg.compute(
            self._W,
            converters=list(self.converters),
            node_to_idx=self.node_to_idx
        )
        
        # Compute liquidity scores
        logger.debug(f"Computing liquidity scores using {liquidity_alg.__class__.__name__}")
        liquidity_raw = liquidity_alg.compute(
            trust_matrix=self._W,
            balance_matrix=self._B,
            rate_matrix=self._R,
            converters=list(self.converters),
            node_to_idx=self.node_to_idx,
            tau=self.config.trust_network.parameters.tau
        )
        
        # FIXED: Proper normalization that preserves differences
        social_scores = self._normalize_scores_properly(social_raw)
        liquidity_scores = self._normalize_scores_properly(liquidity_raw)
        
        # Compute composite scores
        alpha = self.config.trust_network.weights.alpha
        beta = self.config.trust_network.weights.beta
        composite_scores = alpha * social_scores + beta * liquidity_scores
        
        # Don't over-normalize composite scores
        # composite_scores already properly weighted
        
        # Build individual node scores
        scores = self._build_node_scores(
            composite_scores, social_scores, liquidity_scores,
            social_raw, liquidity_raw
        )
        
        # Compute network statistics
        network_stats = self._compute_network_statistics()
        
        # Build result
        result = TrustScoreResult(
            scores=scores,
            computation_time=time.time() - start_time,
            convergence_iterations=getattr(social_alg, 'iterations_used', 0),
            network_stats=network_stats,
            algorithm_used=f"{self.config.trust_network.algorithms.social}+{self.config.trust_network.algorithms.liquidity}"
        )
        
        # Cache result
        self.cache.set_scores(cache_key, result)
        
        logger.info(f"Trust score computation completed in {result.computation_time:.3f} seconds")
        return result
    
    def _normalize_scores_properly(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range while preserving relative differences."""
        if len(scores) == 0:
            return scores
            
        min_score = scores.min()
        max_score = scores.max()
        
        # If all scores are the same, return them as-is (avoid division by zero)
        if max_score - min_score < 1e-10:
            return scores / scores.max() if scores.max() > 0 else scores
        
        # Min-max normalization preserves relative differences
        normalized = (scores - min_score) / (max_score - min_score)
        
        # Ensure no score is exactly 0 (add small epsilon to maintain ranking)
        normalized = normalized * 0.9 + 0.1
        
        return normalized

    def find_trust_path(self, source: str, target: str, 
                       max_hops: Optional[int] = None) -> Optional[TrustPath]:
        """Find the most trustworthy path between two nodes."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # Check cache
        cache_key = (source, target)
        cached_path = self.cache.get_path(cache_key)
        if cached_path is not None and time.time() - cached_path.created_at < 3600:
            return cached_path
        
        # Build matrices if needed
        self._build_matrices()
        
        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target]
        
        # Compute transitive trust using Dijkstra
        trust_value, path_indices = self._compute_shortest_trust_path(source_idx, target_idx)
        
        if trust_value < self.config.trust_network.parameters.tau or not path_indices:
            return None
        
        # Convert to node labels
        path_nodes = [self.idx_to_node[idx] for idx in path_indices]
        
        # Compute bottleneck capacity
        min_capacity = self._compute_path_capacity(path_indices)
        
        # Build path object
        path = TrustPath(
            nodes=path_nodes,
            trust_value=trust_value,
            hop_count=len(path_nodes) - 1,
            bottleneck_capacity=min_capacity
        )
        
        # Cache result
        self.cache.set_path(cache_key, path)
        
        return path
    
    def get_ranking(self, top_n: Optional[int] = None, 
                   sort_by: str = "composite") -> List[Tuple[str, float]]:
        """Get nodes ranked by trust score."""
        result = self.compute_trust_scores()
        return result.get_ranking(sort_by=sort_by, top_n=top_n)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        max_score = scores.max()
        if max_score > 0:
            return scores / max_score
        return scores
    
    def _build_node_scores(self, composite: np.ndarray, social: np.ndarray, 
                          liquidity: np.ndarray, social_raw: np.ndarray,
                          liquidity_raw: np.ndarray) -> Dict[str, NodeTrustScore]:
        """Build NodeTrustScore objects for all nodes."""
        scores = {}
        
        # Compute additional metrics
        in_degrees = np.asarray((self._W > 0).sum(axis=0)).ravel()
        out_degrees = np.asarray((self._W > 0).sum(axis=1)).ravel()
        token_supplies = np.asarray(self._B.sum(axis=0)).ravel()
        
        for node, info in self.nodes.items():
            idx = info.index
            
            scores[node] = NodeTrustScore(
                node=node,
                composite_score=float(composite[idx]),
                social_score=float(social[idx]),
                liquidity_score=float(liquidity[idx]),
                eigentrust=float(social_raw[idx]) if self.config.trust_network.algorithms.social == "eigentrust" else 0.0,
                pagerank=float(social_raw[idx]) if self.config.trust_network.algorithms.social == "pagerank" else 0.0,
                appleseed=float(social_raw[idx]) if self.config.trust_network.algorithms.social == "appleseed" else 0.0,
                in_degree=int(in_degrees[idx]),
                out_degree=int(out_degrees[idx]),
                token_supply=float(token_supplies[idx])
            )
        
        return scores
    
    def _compute_shortest_trust_path(self, source_idx: int, target_idx: int) -> Tuple[float, List[int]]:
        """Compute shortest trust path using Dijkstra's algorithm."""
        n = self._W.shape[0]
        
        # Build cost matrix: c(i,j) = -ln(W(i,j))
        cost_data = []
        cost_indices = []
        cost_indptr = [0]
        
        for i in range(n):
            row_start = len(cost_data)
            row = self._W.getrow(i)
            
            for j_idx, j in enumerate(row.indices):
                weight = row.data[j_idx]
                if weight > 0:
                    cost = -np.log(weight)
                    cost_data.append(cost)
                    cost_indices.append(j)
            
            cost_indptr.append(len(cost_data))
        
        cost_matrix = sp.csr_matrix(
            (cost_data, cost_indices, cost_indptr),
            shape=(n, n),
            dtype=np.float64
        )
        
        # Run Dijkstra's algorithm
        distances, predecessors = dijkstra(
            csgraph=cost_matrix,
            directed=True,
            indices=source_idx,
            return_predecessors=True
        )
        
        # Check if path exists
        if predecessors[target_idx] == -9999:
            return 0.0, []
        
        # Reconstruct path
        path_indices = []
        current = target_idx
        
        while current != source_idx and current != -9999:
            path_indices.append(current)
            current = predecessors[current]
        
        if current == -9999:
            return 0.0, []
        
        path_indices.append(source_idx)
        path_indices.reverse()
        
        # Convert distance back to trust value
        trust_value = np.exp(-distances[target_idx])
        
        return trust_value, path_indices
    
    def _compute_path_capacity(self, path_indices: List[int]) -> float:
        """Compute bottleneck capacity along a path."""
        if len(path_indices) < 2:
            return 0.0
        
        min_capacity = float('inf')
        for i in range(len(path_indices) - 1):
            u_idx = path_indices[i]
            v_idx = path_indices[i + 1]
            weight = self._W[u_idx, v_idx]
            min_capacity = min(min_capacity, weight)
        
        return min_capacity
    
    def _compute_network_statistics(self) -> NetworkStatistics:
        """Compute comprehensive network statistics."""
        n = len(self.nodes)
        m = len(self.edges)
        
        if n == 0:
            return NetworkStatistics(0, 0, 0, 0, 0.0, 0.0, -1, 0, 0.0)
        
        # Basic counts
        num_converters = len(self.converters)
        num_tokens = n
        total_supply = float(self._B.sum()) if self._B is not None else 0.0
        
        # Connected components
        num_components, component_labels = connected_components(
            csgraph=self._W,
            directed=True,
            connection='weak'
        )
        
        # Largest component size
        if num_components > 0:
            component_sizes = np.bincount(component_labels)
            largest_component_size = int(component_sizes.max())
        else:
            largest_component_size = 0
        
        # Network density
        possible_edges = n * (n - 1)
        density = m / possible_edges if possible_edges > 0 else 0.0
        
        # Average clustering (simplified)
        avg_clustering = 0.0
        if n > 2:
            clustering_sum = 0.0
            for i in range(n):
                neighbors = set(self._W.getrow(i).indices)
                k = len(neighbors)
                
                if k >= 2:
                    edges_between = 0
                    for u in neighbors:
                        for v in neighbors:
                            if u != v and self._W[u, v] > 0:
                                edges_between += 1
                    
                    possible = k * (k - 1)
                    clustering_sum += edges_between / possible if possible > 0 else 0
            
            avg_clustering = clustering_sum / n
        
        # Estimate diameter (for small networks)
        diameter = -1
        if n < 1000:
            try:
                all_distances = dijkstra(csgraph=self._W, directed=True)
                finite_distances = all_distances[np.isfinite(all_distances)]
                if len(finite_distances) > 0:
                    diameter = int(finite_distances.max())
            except:
                pass
        
        return NetworkStatistics(
            num_nodes=n,
            num_edges=m,
            num_converters=num_converters,
            num_tokens=num_tokens,
            total_supply=total_supply,
            avg_clustering=avg_clustering,
            diameter=diameter,
            largest_component_size=largest_component_size,
            density=density
        )
    
    def _compute_network_hash(self) -> str:
        """Compute a hash of the network state for cache invalidation."""
        # Create a deterministic representation of the network
        node_data = str(sorted([(n.label, n.is_converter) for n in self.nodes.values()]))
        edge_data = str(sorted([(e.source, e.target, e.weight) for e in self.edges]))
        balance_data = str(sorted([
            (holder, token, amount) 
            for holder, tokens in self.balances.items()
            for token, amount in tokens.items()
        ]))
        rate_data = str(sorted([
            (conv, token, rate)
            for conv, rates in self.rates.items()
            for token, rate in rates.items()
        ]))
        
        combined = node_data + edge_data + balance_data + rate_data
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _invalidate_caches(self) -> None:
        """Invalidate all caches when network changes."""
        self._matrices_built = False
        self._W = None
        self._B = None
        self._R = None
        self._network_hash = None
        self.cache.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export network to dictionary format."""
        return {
            "nodes": [
                {
                    "label": node.label,
                    "is_converter": node.is_converter,
                    "token_symbol": node.token_symbol,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ],
            "balances": [
                {
                    "holder": holder,
                    "token": token,
                    "amount": amount
                }
                for holder, tokens in self.balances.items()
                for token, amount in tokens.items()
            ],
            "rates": [
                {
                    "converter": converter,
                    "token": token,
                    "rate": rate
                }
                for converter, rates in self.rates.items()
                for token, rate in rates.items()
            ]
        }
    
    def save(self, filepath: str, format: str = "json") -> None:
        """Save network to file."""
        if format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Network saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: NetworkConfig, format: str = "json") -> 'TrustNetwork':
        """Load network from file."""
        if format == "json":
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            network = cls(config)
            network.load_data(data)
            return network
            
        elif format == "pickle":
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
    def estimate_payment_capacity(self, source: str, target: str, token: str) -> float:
        """Estimate the payment capacity from source to target for a specific token."""
        # Find trust path
        path = self.find_trust_path(source, target)
        
        if path is None or not path.is_valid:
            return 0.0
        
        # Check if target accepts the token
        token_idx = self.node_to_idx.get(token)
        target_idx = self.node_to_idx.get(target)
        
        if token_idx is None or target_idx is None:
            return 0.0
        
        # For now, return a simple capacity based on path trust
        # You can make this more sophisticated later
        return float(path.trust_value * 100)  # Simple approximation
    

    def estimate_payment_capacity(self, source: str, target: str, token: str) -> float:
        """
        Estimate payment capacity using the theoretical framework.
        
        Implements: Capacity(s → t, T(token)) = min{B[v_i, token] · 1[W[v_{i+1}, token] ≥ τ]}
        """
        if source not in self.nodes or target not in self.nodes or token not in self.nodes:
            return 0.0
        
        # Import payment capacity calculator
        from .payment_capacity import PaymentCapacityCalculator
        
        # Build matrices if needed
        self._build_matrices()
        
        # Get indices
        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target] 
        token_idx = self.node_to_idx[token]
        
        # Create calculator
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        # Compute capacity
        capacity, path, reason = calculator.compute_payment_capacity(source_idx, target_idx, token_idx)
        
        logger.debug(f"Payment capacity {source}→{target} (token {token}): {capacity} ({reason})")
        
        return capacity

    def find_payment_path_with_capacity(self, source: str, target: str, token: str) -> Optional[Dict]:
        """
        Find payment path with detailed capacity analysis.
        
        Returns:
            Dictionary with path details or None if no viable path
        """
        if source not in self.nodes or target not in self.nodes or token not in self.nodes:
            return None
        
        from .payment_capacity import PaymentCapacityCalculator
        
        self._build_matrices()
        
        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target]
        token_idx = self.node_to_idx[token]
        
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        capacity, path_indices, reason = calculator.compute_payment_capacity(source_idx, target_idx, token_idx)
        
        if capacity <= 0:
            return None
        
        # Convert path indices to node labels
        path_nodes = [self.idx_to_node[idx] for idx in path_indices]
        
        # Analyze bottlenecks along path
        bottlenecks = []
        for i in range(len(path_indices) - 1):
            current_idx = path_indices[i]
            next_idx = path_indices[i + 1]
            
            # Check balance at current node
            balance = self._B[current_idx, token_idx]
            trust = self._W[next_idx, token_idx]
            
            bottlenecks.append({
                'node': self.idx_to_node[current_idx],
                'next_node': self.idx_to_node[next_idx],
                'balance': float(balance),
                'trust_to_token': float(trust),
                'is_bottleneck': balance <= capacity * 1.1  # Within 10% of bottleneck
            })
        
        return {
            'source': source,
            'target': target,
            'token': token,
            'capacity': capacity,
            'path': path_nodes,
            'hops': len(path_nodes) - 1,
            'bottlenecks': bottlenecks,
            'reason': reason
        }

    def analyze_network_payment_bottlenecks(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of payment bottlenecks in the network.
        
        Returns:
            Dictionary with detailed bottleneck analysis
        """
        from .payment_capacity import PaymentCapacityCalculator
        
        self._build_matrices()
        
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        return calculator.analyze_payment_bottlenecks()

    def compute_network_liquidity_score(self) -> float:
        """
        Compute overall network liquidity score based on payment capacities.
        
        Returns:
            Score between 0 and 1 indicating network payment liquidity
        """
        from .payment_capacity import PaymentCapacityCalculator
        
        self._build_matrices()
        
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        return calculator.compute_network_liquidity_score()

    def get_payment_matrix_for_token(self, token: str) -> np.ndarray:
        """
        Get payment capacity matrix for a specific token.
        
        Returns:
            n x n matrix where entry [i,j] is payment capacity from node i to node j using token
        """
        if token not in self.nodes:
            return np.array([])
        
        from .payment_capacity import PaymentCapacityCalculator
        
        self._build_matrices()
        
        token_idx = self.node_to_idx[token]
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        return calculator.compute_all_payment_capacities(token_idx)

    def find_best_payment_paths(self, min_capacity: float = 1.0, top_k: int = 20) -> List[Dict]:
        """
        Find the best payment paths in the network.
        
        Args:
            min_capacity: Minimum capacity threshold
            top_k: Number of top paths to return
            
        Returns:
            List of best payment paths sorted by capacity
        """
        from .payment_capacity import PaymentCapacityCalculator
        
        self._build_matrices()
        
        calculator = PaymentCapacityCalculator(self._W, self._B, self.config.trust_network.parameters.tau)
        
        all_paths = []
        
        # Check paths for each token
        for token, token_info in self.nodes.items():
            token_idx = token_info.index
            viable_paths = calculator.get_payment_paths_for_token(token_idx, min_capacity)
            
            # Convert indices to node labels
            for path_info in viable_paths:
                path_info['source_label'] = self.idx_to_node[path_info['source']]
                path_info['target_label'] = self.idx_to_node[path_info['target']]
                path_info['token_label'] = token
                path_info['path_labels'] = [self.idx_to_node[idx] for idx in path_info['path']]
                
            all_paths.extend(viable_paths)
        
        # Sort by capacity and return top k
        all_paths.sort(key=lambda x: x['capacity'], reverse=True)
        
        return all_paths[:top_k]

    def validate_payment_logic(self) -> Dict[str, Any]:
        """
        Validate that the payment logic is working correctly.
        
        Returns:
            Dictionary with validation results
        """
        self._build_matrices()
        
        validation_results = {
            'trust_matrix_properties': {},
            'balance_matrix_properties': {},
            'payment_capacity_tests': {},
            'logical_consistency': {}
        }
        
        n = len(self.nodes)
        
        # Check trust matrix properties
        validation_results['trust_matrix_properties'] = {
            'shape': self._W.shape,
            'nnz': self._W.nnz,
            'density': self._W.nnz / (n * n) if n > 0 else 0,
            'has_self_loops': np.diag(self._W.toarray()).sum() > 0,
            'max_trust': self._W.max(),
            'min_positive_trust': self._W.data[self._W.data > 0].min() if self._W.nnz > 0 else 0
        }
        
        # Check balance matrix properties  
        validation_results['balance_matrix_properties'] = {
            'shape': self._B.shape,
            'nnz': self._B.nnz,
            'total_supply': self._B.sum(),
            'diagonal_sum': np.diag(self._B.toarray()).sum(),
            'max_balance': self._B.max(),
            'nodes_with_self_tokens': (np.diag(self._B.toarray()) > 0).sum()
        }
        
        # Test payment capacity logic
        test_results = []
        node_labels = list(self.nodes.keys())
        
        if len(node_labels) >= 2:
            # Test a few random payment scenarios
            for _ in range(min(5, len(node_labels))):
                source = np.random.choice(node_labels)
                target = np.random.choice([n for n in node_labels if n != source])
                token = source  # Use source's own token
                
                capacity = self.estimate_payment_capacity(source, target, token)
                path_info = self.find_payment_path_with_capacity(source, target, token)
                
                test_results.append({
                    'source': source,
                    'target': target, 
                    'token': token,
                    'capacity': capacity,
                    'path_exists': path_info is not None,
                    'path_length': len(path_info['path']) if path_info else 0
                })
        
        validation_results['payment_capacity_tests'] = test_results
        
        # Check logical consistency
        consistency_checks = {
            'nodes_with_no_self_tokens': 0,
            'isolated_nodes': 0,
            'converter_reachability': 0
        }
        
        for node, info in self.nodes.items():
            idx = info.index
            
            # Check self-tokens
            if self._B[idx, idx] == 0:
                consistency_checks['nodes_with_no_self_tokens'] += 1
            
            # Check isolation
            in_trust = (self._W[:, idx] > 0).sum()
            out_trust = (self._W[idx, :] > 0).sum()
            if in_trust == 0 and out_trust == 0:
                consistency_checks['isolated_nodes'] += 1
        
        # Check converter reachability
        if self.converters:
            reachable_converters = 0
            for node in self.nodes:
                for conv in self.converters:
                    if self.estimate_payment_capacity(node, conv, node) > 0:
                        reachable_converters += 1
                        break
            consistency_checks['converter_reachability'] = reachable_converters / len(self.nodes)
        
        validation_results['logical_consistency'] = consistency_checks
        
        return validation_results