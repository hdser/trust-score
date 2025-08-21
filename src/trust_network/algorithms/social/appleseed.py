"""Appleseed algorithm implementation."""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional
import logging

from ..base import SocialTrustAlgorithm, IterativeAlgorithm

logger = logging.getLogger(__name__)


class AppleseedAlgorithm(SocialTrustAlgorithm, IterativeAlgorithm):
    """Appleseed algorithm for computing social trust scores."""
    
    def __init__(self, config):
        super().__init__(config)
        self.energy_decay = config.energy  # Energy retention factor
        self.max_iterations = max(config.iterations, 200) 
    
    def compute(self, trust_matrix: sp.csr_matrix, 
                converters: List[str] = None,
                node_to_idx: Dict[str, int] = None,
                **kwargs) -> np.ndarray:
        """Compute Appleseed trust scores."""
        n = trust_matrix.shape[0]
        
        if n == 0:
            return np.array([])
        
        logger.debug(f"Computing Appleseed for {n} nodes")
        
        # Build row-stochastic transition matrix
        P = self._build_row_stochastic_matrix(trust_matrix)
        
        # Initialize seed energy
        energy = self._initialize_energy(n, converters, node_to_idx)
        
        # Trust accumulator
        trust = np.zeros(n, dtype=np.float64)
        
        # Energy propagation
        for iteration in range(self.max_iterations):
            # Accumulate trust from current energy
            trust += (1 - self.energy_decay) * energy
            
            # Propagate energy
            new_energy = self.energy_decay * (P.T @ energy)
            
            # Check convergence
            if self.check_convergence(new_energy, energy):
                self.iterations_used = iteration + 1
                logger.debug(f"Appleseed converged at iteration {iteration + 1}")
                trust += new_energy  # Add remaining energy
                break
            
            energy = new_energy
        else:
            self.iterations_used = self.max_iterations
            logger.warning(f"Appleseed did not converge in {self.max_iterations} iterations")
            trust += energy  # Add remaining energy
        
        # Normalize
        total = trust.sum()
        if total > 0:
            trust = trust / total
        
        return trust
    
    def _build_row_stochastic_matrix(self, trust_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Build row-stochastic matrix from trust matrix."""
        # Calculate row sums
        row_sums = np.asarray(trust_matrix.sum(axis=1)).ravel()
        
        # Handle zero rows (nodes with no outgoing trust)
        row_sums[row_sums == 0] = 1.0
        
        # Create diagonal matrix for normalization
        D_inv = sp.diags(1.0 / row_sums, format='csr')
        
        # P = D^(-1) * W
        P = D_inv @ trust_matrix
        
        return P
    
    def _initialize_energy(self, n: int, converters: List[str] = None,
                          node_to_idx: Dict[str, int] = None) -> np.ndarray:
        """Initialize energy distribution for Appleseed."""
        energy = np.zeros(n, dtype=np.float64)
        
        if converters and node_to_idx:
            # Start energy at converter nodes
            converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
            if converter_indices:
                energy[converter_indices] = 1.0 / len(converter_indices)
            else:
                # No converters found, use uniform distribution
                energy[:] = 1.0 / n
        else:
            # No converters specified, use uniform distribution
            energy[:] = 1.0 / n
        
        return energy