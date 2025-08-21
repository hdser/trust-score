"""Community detection utilities."""

import numpy as np
from typing import Dict, Set, List, Any
from collections import defaultdict
import logging

from ..core.data_models import CommunityStructure

logger = logging.getLogger(__name__)


class CommunityAnalyzer:
    """Community detection and analysis."""
    
    def __init__(self, network):
        self.network = network
    
    def detect_communities(self, method: str = "modularity") -> CommunityStructure:
        """Detect communities in the network."""
        if method == "modularity":
            return self._detect_communities_modularity()
        elif method == "connected_components":
            return self._detect_connected_components()
        else:
            raise ValueError(f"Unknown community detection method: {method}")
    
    def _detect_communities_modularity(self) -> CommunityStructure:
        """Detect communities using modularity optimization (simplified)."""
        if self.network._W is None:
            self.network._build_matrices()
        
        n = len(self.network.nodes)
        
        if n == 0:
            return CommunityStructure({}, 0.0, 0, {})
        
        # Start with each node in its own community
        communities = {node: i for i, node in enumerate(self.network.nodes)}
        
        # Simplified modularity optimization using greedy approach
        best_modularity = self._compute_modularity(communities)
        improved = True
        
        while improved:
            improved = False
            
            for node in self.network.nodes:
                current_community = communities[node]
                best_community = current_community
                best_delta = 0
                
                # Try moving node to neighbor communities
                neighbor_communities = set()
                node_idx = self.network.node_to_idx[node]
                
                # Get neighbors
                row = self.network._W.getrow(node_idx)
                for neighbor_idx in row.indices:
                    neighbor_node = self.network.idx_to_node[neighbor_idx]
                    neighbor_communities.add(communities[neighbor_node])
                
                # Try each neighbor community
                for community_id in neighbor_communities:
                    if community_id == current_community:
                        continue
                    
                    # Temporarily move node
                    communities[node] = community_id
                    new_modularity = self._compute_modularity(communities)
                    delta = new_modularity - best_modularity
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_community = community_id
                    
                    # Restore original community
                    communities[node] = current_community
                
                # Make the best move
                if best_delta > 0:
                    communities[node] = best_community
                    best_modularity += best_delta
                    improved = True
        
        # Relabel communities to be consecutive
        community_mapping = {}
        next_id = 0
        final_communities = {}
        
        for node, comm_id in communities.items():
            if comm_id not in community_mapping:
                community_mapping[comm_id] = next_id
                next_id += 1
            final_communities[node] = community_mapping[comm_id]
        
        # Compute community sizes
        community_sizes = defaultdict(int)
        for comm_id in final_communities.values():
            community_sizes[comm_id] += 1
        
        return CommunityStructure(
            communities=final_communities,
            modularity=best_modularity,
            num_communities=len(community_sizes),
            community_sizes=dict(community_sizes)
        )
    
    def _detect_connected_components(self) -> CommunityStructure:
        """Detect communities as connected components."""
        from scipy.sparse.csgraph import connected_components
        
        if self.network._W is None:
            self.network._build_matrices()
        
        num_components, labels = connected_components(
            csgraph=self.network._W,
            directed=True,
            connection='weak'
        )
        
        # Convert to community structure
        communities = {}
        for node, info in self.network.nodes.items():
            communities[node] = int(labels[info.index])
        
        # Compute community sizes
        community_sizes = defaultdict(int)
        for comm_id in communities.values():
            community_sizes[comm_id] += 1
        
        # Modularity for connected components
        modularity = self._compute_modularity(communities)
        
        return CommunityStructure(
            communities=communities,
            modularity=modularity,
            num_communities=num_components,
            community_sizes=dict(community_sizes)
        )
    
    def _compute_modularity(self, communities: Dict[str, int]) -> float:
        """Compute modularity of a community assignment."""
        if self.network._W is None:
            return 0.0
        
        # Total edge weight
        m = self.network._W.sum()
        if m == 0:
            return 0.0
        
        modularity = 0.0
        
        # Degree vectors
        out_degree = np.asarray(self.network._W.sum(axis=1)).ravel()
        in_degree = np.asarray(self.network._W.sum(axis=0)).ravel()
        
        for i, node_i in enumerate(self.network.nodes):
            for j, node_j in enumerate(self.network.nodes):
                if communities[node_i] == communities[node_j]:
                    A_ij = self.network._W[i, j]
                    expected = (out_degree[i] * in_degree[j]) / m
                    modularity += (A_ij - expected) / m
        
        return modularity
    
    def analyze_community_structure(self, communities: CommunityStructure) -> Dict[str, Any]:
        """Analyze properties of detected communities."""
        analysis = {
            "num_communities": communities.num_communities,
            "modularity": communities.modularity,
            "community_sizes": communities.community_sizes,
            "size_distribution": self._analyze_size_distribution(communities.community_sizes),
            "inter_community_edges": self._count_inter_community_edges(communities.communities),
            "community_centralization": self._compute_community_centralization(communities.communities)
        }
        
        return analysis
    
    def _analyze_size_distribution(self, community_sizes: Dict[int, int]) -> Dict[str, float]:
        """Analyze the distribution of community sizes."""
        sizes = list(community_sizes.values())
        
        if not sizes:
            return {}
        
        return {
            "mean_size": float(np.mean(sizes)),
            "std_size": float(np.std(sizes)),
            "min_size": int(np.min(sizes)),
            "max_size": int(np.max(sizes)),
            "median_size": float(np.median(sizes))
        }
    
    def _count_inter_community_edges(self, communities: Dict[str, int]) -> int:
        """Count edges between different communities."""
        inter_edges = 0
        
        for edge in self.network.edges:
            source_comm = communities.get(edge.source)
            target_comm = communities.get(edge.target)
            
            if source_comm is not None and target_comm is not None:
                if source_comm != target_comm:
                    inter_edges += 1
        
        return inter_edges
    
    def _compute_community_centralization(self, communities: Dict[str, int]) -> float:
        """Compute how centralized the community structure is."""
        community_sizes = defaultdict(int)
        for comm_id in communities.values():
            community_sizes[comm_id] += 1
        
        if not community_sizes:
            return 0.0
        
        sizes = list(community_sizes.values())
        max_size = max(sizes)
        total_nodes = sum(sizes)
        
        # Centralization based on largest community
        return max_size / total_nodes if total_nodes > 0 else 0.0