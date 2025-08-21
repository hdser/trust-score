"""Liquidity algorithms."""

from .conductance import ConductanceAlgorithm
from .flow_centrality import FlowCentralityAlgorithm
from .hybrid import HybridLiquidityAlgorithm

_ALGORITHMS = {
    "conductance": ConductanceAlgorithm,
    "flow_centrality": FlowCentralityAlgorithm,
    "pagerank": FlowCentralityAlgorithm,  # Alias for flow centrality
    "hybrid": HybridLiquidityAlgorithm
}


def get_liquidity_algorithm(algorithm_name: str, config):
    """Get liquidity algorithm instance."""
    if algorithm_name not in _ALGORITHMS:
        raise ValueError(f"Unknown liquidity algorithm: {algorithm_name}. Available: {list(_ALGORITHMS.keys())}")
    
    alg_class = _ALGORITHMS[algorithm_name]
    return alg_class(config)


__all__ = ["ConductanceAlgorithm", "FlowCentralityAlgorithm", "HybridLiquidityAlgorithm", "get_liquidity_algorithm"]