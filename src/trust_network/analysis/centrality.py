"""Centrality analysis utilities."""

import numpy as np
from scipy.sparse.csgraph import dijkstra
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CentralityAnalyzer:
    """Centrality metrics computation."""
    
    def __init__(self, network):
        self.network = network
    
    def compute_all_centralities(self) -> Dict[str, Dict[str, float]]:
        """Compute various centrality metrics."""
        if self.network._W is None:
            self.network._build_matrices()
        
        metrics = {
            "degree": self.compute_degree_centrality(),
            "closeness": self.compute_closeness_centrality(),
            "betweenness": self.compute_betweenness_centrality(),
            "eigenvector": self.compute_eigenvector_centrality()
        }
        
        return metrics
    
    def compute_degree_centrality(self) -> Dict[str, float]:
        """Compute degree centrality for all nodes."""
        n = len(self.network.nodes)
        
        if n <= 1:
            return {node: 0.0 for node in self.network.nodes}
        
        # In-degree and out-degree
        in_degree = np.asarray((self.network._W > 0).sum(axis=0)).ravel()
        out_degree = np.asarray((self.network._W > 0).sum(axis=1)).ravel()
        
        # Total degree centrality
        degree_centrality = {}
        for node, info in self.network.nodes.items():
            idx = info.index
            total_degree = in_degree[idx] + out_degree[idx]
            # Normalize by maximum possible degree
            degree_centrality[node] = total_degree / (2 * (n - 1))
        
        return degree_centrality
    
    def compute_closeness_centrality(self) -> Dict[str, float]:
        """Compute closeness centrality for all nodes."""
        n = len(self.network.nodes)
        closeness = {}
        
        if n <= 1:
            return {node: 0.0 for node in self.network.nodes}
        
        # Compute all-pairs shortest paths
        try:
            distances = dijkstra(
                csgraph=self.network._W,
                directed=True,
                return_predecessors=False
            )
            
            for node, info in self.network.nodes.items():
                idx = info.index
                node_distances = distances[idx]
                
                # Only consider finite distances
                finite_distances = node_distances[np.isfinite(node_distances)]
                
                if len(finite_distances) > 1:
                    # Closeness = (n-1) / sum of distances
                    closeness[node] = (len(finite_distances) - 1) / finite_distances.sum()
                else:
                    closeness[node] = 0.0
        
        except Exception as e:
            logger.warning(f"Failed to compute closeness centrality: {e}")
            closeness = {node: 0.0 for node in self.network.nodes}
        
        return closeness
    
    def compute_betweenness_centrality(self) -> Dict[str, float]:
        """Compute approximated betweenness centrality."""
        n = len(self.network.nodes)
        betweenness = {node: 0.0 for node in self.network.nodes}
        
        if n <= 2:
            return betweenness
        
        # Simplified betweenness using random sampling
        num_samples = min(100, n * (n - 1) // 2)
        
        try:
            # Sample node pairs
            nodes = list(self.network.nodes.keys())
            
            for _ in range(num_samples):
                # Random source and target
                source = np.random.choice(nodes)
                target = np.random.choice(nodes)
                
                if source == target:
                    continue
                
                # Find shortest path
                path = self.network.find_trust_path(source, target)
                
                if path and path.is_valid and len(path.nodes) > 2:
                    # Add to betweenness of intermediate nodes
                    for intermediate in path.nodes[1:-1]:
                        betweenness[intermediate] += 1.0
            
            # Normalize
            if num_samples > 0:
                for node in betweenness:
                    betweenness[node] /= num_samples
        
        except Exception as e:
            logger.warning(f"Failed to compute betweenness centrality: {e}")
        
        return betweenness
    
    def compute_eigenvector_centrality(self) -> Dict[str, float]:
        """Compute eigenvector centrality."""
        try:
            from scipy.sparse.linalg import eigs
            
            # Use the largest eigenvalue's eigenvector
            eigenvalues, eigenvectors = eigs(
                self.network._W.T,  # Transpose for left eigenvector
                k=1,
                which='LM'
            )
            
            # Get the principal eigenvector
            principal_eigenvector = np.real(eigenvectors[:, 0])
            
            # Ensure non-negative values
            if principal_eigenvector.sum() < 0:
                principal_eigenvector = -principal_eigenvector
            
            # Normalize
            norm = np.linalg.norm(principal_eigenvector)
            if norm > 0:
                principal_eigenvector = principal_eigenvector / norm
            
            # Convert to dictionary
            eigenvector_centrality = {}
            for node, info in self.network.nodes.items():
                idx = info.index
                eigenvector_centrality[node] = float(principal_eigenvector[idx])
        
        except Exception as e:
            logger.warning(f"Failed to compute eigenvector centrality: {e}")
            eigenvector_centrality = {node: 0.0 for node in self.network.nodes}
        
        return eigenvector_centrality