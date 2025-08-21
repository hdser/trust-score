"""EigenTrust algorithm implementation."""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional
import logging

from ..base import SocialTrustAlgorithm, IterativeAlgorithm

logger = logging.getLogger(__name__)


class EigenTrustAlgorithm(SocialTrustAlgorithm, IterativeAlgorithm):
    """EigenTrust algorithm for computing social trust scores."""
    
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha  # Teleportation probability
        self.max_iterations = config.iterations
        self.tolerance = 1e-9
        self.iterations_used = 0
    
    def compute(self, trust_matrix: sp.csr_matrix, 
                converters: List[str] = None,
                node_to_idx: Dict[str, int] = None,
                **kwargs) -> np.ndarray:
        """Compute EigenTrust scores."""
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        logger.debug(f"Computing EigenTrust for {n} nodes")
        
        # Remove self-loops efficiently for EigenTrust computation
        W_no_self = self._remove_self_loops_efficiently(trust_matrix)
        
        # Build column-stochastic matrix C
        C = self._build_column_stochastic_matrix(W_no_self)
        
        # Initialize pre-trust vector
        pre_trust = self._build_pre_trust_vector(n, converters, node_to_idx)
        
        # Power iteration
        trust = pre_trust.copy()
        
        for iteration in range(self.max_iterations):
            trust_prev = trust.copy()
            
            # EigenTrust update: t = (1-α)C^T t + α p
            trust = (1 - self.alpha) * (C.T @ trust_prev) + self.alpha * pre_trust
            
            # Check convergence
            if np.linalg.norm(trust - trust_prev, ord=1) < self.tolerance:
                self.iterations_used = iteration + 1
                logger.debug(f"EigenTrust converged at iteration {iteration + 1}")
                break
        else:
            self.iterations_used = self.max_iterations
            logger.warning(f"EigenTrust did not converge in {self.max_iterations} iterations")
        
        return trust
    
    def _remove_self_loops_efficiently(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Remove self-loops efficiently without sparse structure warnings."""
        # Convert to COO format for efficient diagonal modification
        coo = trust_matrix.tocoo()
        
        # Find non-diagonal entries
        mask = coo.row != coo.col
        
        # Create new matrix without diagonal entries
        W_no_self = sp.csr_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])),
            shape=trust_matrix.shape
        )
        
        return W_no_self
    
    def _build_column_stochastic_matrix(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Build column-stochastic matrix from trust matrix."""
        # Calculate column sums
        col_sums = np.asarray(trust_matrix.sum(axis=0)).ravel()
        
        # Handle zero columns (dangling nodes) - distribute weight equally
        zero_cols = (col_sums == 0)
        col_sums[zero_cols] = 1.0
        
        # Create diagonal matrix for normalization
        D_inv = sp.diags(1.0 / col_sums, format='csr')
        
        # C = W * D^(-1)
        C = trust_matrix @ D_inv
        
        # For zero columns, distribute incoming trust equally
        if np.any(zero_cols):
            n = trust_matrix.shape[0]
            for j in np.where(zero_cols)[0]:
                C[:, j] = 1.0 / n
        
        return C
    
    def _build_pre_trust_vector(self, n: int, converters: List[str] = None, 
                               node_to_idx: Dict[str, int] = None) -> np.ndarray:
        """Build pre-trust vector for EigenTrust."""
        pre_trust = np.ones(n) / n
        
        # Give extra weight to converters for Sybil resistance
        if converters and node_to_idx:
            converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
            if converter_indices:
                # Start with uniform, then boost converters
                pre_trust = np.ones(n) * 0.1  # Base trust for all nodes
                pre_trust[converter_indices] = 0.9 / len(converter_indices)  # High trust for converters
                
                # Normalize to ensure it's a probability distribution
                pre_trust = pre_trust / pre_trust.sum()
        
        return pre_trust