"""Social trust algorithms."""

from .eigentrust import EigenTrustAlgorithm
from .appleseed import AppleseedAlgorithm
from .pagerank import PageRankAlgorithm

_ALGORITHMS = {
    "eigentrust": EigenTrustAlgorithm,
    "appleseed": AppleseedAlgorithm,
    "pagerank": PageRankAlgorithm
}


def get_social_algorithm(algorithm_name: str, config):
    """Get social trust algorithm instance."""
    if algorithm_name not in _ALGORITHMS:
        raise ValueError(f"Unknown social algorithm: {algorithm_name}. Available: {list(_ALGORITHMS.keys())}")
    
    alg_class = _ALGORITHMS[algorithm_name]
    alg_config = getattr(config, algorithm_name)
    return alg_class(alg_config)


__all__ = ["EigenTrustAlgorithm", "AppleseedAlgorithm", "PageRankAlgorithm", "get_social_algorithm"]