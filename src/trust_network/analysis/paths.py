"""Path analysis utilities."""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PathAnalyzer:
    """Path-related analysis."""
    
    def __init__(self, network):
        self.network = network
    
    def analyze_path_distribution(self, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze distribution of path lengths and trust values."""
        nodes = list(self.network.nodes.keys())
        
        if len(nodes) < 2:
            return {"error": "Need at least 2 nodes for path analysis"}
        
        path_lengths = []
        trust_values = []
        successful_paths = 0
        
        # Sample random node pairs
        for _ in range(sample_size):
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)
            
            if source == target:
                continue
            
            path = self.network.find_trust_path(source, target)
            
            if path and path.is_valid:
                path_lengths.append(path.hop_count)
                trust_values.append(path.trust_value)
                successful_paths += 1
        
        if successful_paths == 0:
            return {
                "sample_size": sample_size,
                "successful_paths": 0,
                "connectivity_rate": 0.0
            }
        
        return {
            "sample_size": sample_size,
            "successful_paths": successful_paths,
            "connectivity_rate": successful_paths / sample_size,
            "path_length_stats": {
                "mean": float(np.mean(path_lengths)),
                "std": float(np.std(path_lengths)),
                "min": int(np.min(path_lengths)),
                "max": int(np.max(path_lengths)),
                "median": float(np.median(path_lengths))
            },
            "trust_value_stats": {
                "mean": float(np.mean(trust_values)),
                "std": float(np.std(trust_values)),
                "min": float(np.min(trust_values)),
                "max": float(np.max(trust_values)),
                "median": float(np.median(trust_values))
            }
        }
    
    def find_critical_paths(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most critical paths in the network."""
        converters = list(self.network.converters)
        regular_nodes = [n for n in self.network.nodes if n not in self.network.converters]
        
        critical_paths = []
        
        # Find paths from regular nodes to converters
        for node in regular_nodes[:20]:  # Limit to avoid too much computation
            best_path = None
            best_score = 0
            
            for converter in converters:
                path = self.network.find_trust_path(node, converter)
                
                if path and path.is_valid:
                    # Score based on trust value and inverse path length
                    score = path.trust_value / (path.hop_count + 1)
                    
                    if score > best_score:
                        best_score = score
                        best_path = path
            
            if best_path:
                critical_paths.append({
                    "source": best_path.nodes[0],
                    "target": best_path.nodes[-1],
                    "path": " -> ".join(best_path.nodes),
                    "trust_value": best_path.trust_value,
                    "hop_count": best_path.hop_count,
                    "score": best_score,
                    "bottleneck_capacity": best_path.bottleneck_capacity
                })
        
        # Sort by score and return top k
        critical_paths.sort(key=lambda x: x["score"], reverse=True)
        return critical_paths[:top_k]
    
    def analyze_payment_capacity(self, token: str, sample_size: int = 50) -> Dict[str, Any]:
        """Analyze payment capacity for a specific token."""
        if token not in self.network.nodes:
            return {"error": f"Token {token} not found"}
        
        nodes = [n for n in self.network.nodes if n != token]
        capacities = []
        
        for _ in range(sample_size):
            if len(nodes) < 2:
                break
            
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)
            
            if source == target:
                continue
            
            capacity = self.network.estimate_payment_capacity(source, target, token)
            if capacity > 0:
                capacities.append(capacity)
        
        if not capacities:
            return {
                "token": token,
                "sample_size": sample_size,
                "feasible_payments": 0
            }
        
        return {
            "token": token,
            "sample_size": sample_size,
            "feasible_payments": len(capacities),
            "capacity_stats": {
                "mean": float(np.mean(capacities)),
                "std": float(np.std(capacities)),
                "min": float(np.min(capacities)),
                "max": float(np.max(capacities)),
                "median": float(np.median(capacities))
            }
        }