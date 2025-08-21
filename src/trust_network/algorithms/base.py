"""Base classes for trust algorithms."""

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any


class TrustAlgorithm(ABC):
    """Base class for trust scoring algorithms."""
    
    def __init__(self, config: Any):
        """Initialize algorithm with configuration."""
        self.config = config
        self.iterations_used = 0
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> np.ndarray:
        """Compute trust scores."""
        pass


class SocialTrustAlgorithm(TrustAlgorithm):
    """Base class for social trust algorithms."""
    
    @abstractmethod
    def compute(self, trust_matrix: sp.csr_matrix, 
                converters: List[str] = None,
                node_to_idx: Dict[str, int] = None,
                **kwargs) -> np.ndarray:
        """
        Compute social trust scores.
        
        Args:
            trust_matrix: Trust adjacency matrix
            converters: List of converter node labels
            node_to_idx: Mapping from node labels to indices
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Array of social trust scores
        """
        pass


class LiquidityAlgorithm(TrustAlgorithm):
    """Base class for liquidity algorithms."""
    
    @abstractmethod
    def compute(self, trust_matrix: sp.csr_matrix,
                balance_matrix: sp.csr_matrix,
                rate_matrix: np.ndarray,
                converters: List[str],
                node_to_idx: Dict[str, int],
                tau: float,
                **kwargs) -> np.ndarray:
        """
        Compute liquidity scores.
        
        Args:
            trust_matrix: Trust adjacency matrix
            balance_matrix: Token balance matrix
            rate_matrix: Conversion rate matrix
            converters: List of converter node labels
            node_to_idx: Mapping from node labels to indices
            tau: Trust acceptance threshold
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Array of liquidity scores
        """
        pass


class IterativeAlgorithm(TrustAlgorithm):
    """Base class for iterative algorithms."""
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.max_iterations = getattr(config, 'iterations', 100)
        self.tolerance = getattr(config, 'tolerance', 1e-9)
    
    def check_convergence(self, current: np.ndarray, previous: np.ndarray) -> bool:
        """Check if algorithm has converged."""
        return np.linalg.norm(current - previous, ord=1) < self.tolerance
    
    def log_convergence(self, iteration: int, error: float) -> None:
        """Log convergence information."""
        import logging
        logger = logging.getLogger(__name__)
        if iteration % 10 == 0 or error < self.tolerance:
            logger.debug(f"Iteration {iteration}: error = {error:.2e}")