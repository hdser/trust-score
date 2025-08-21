"""PageRank algorithm implementation."""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional
import logging

from ..base import SocialTrustAlgorithm, IterativeAlgorithm

logger = logging.getLogger(__name__)


class PageRankAlgorithm(SocialTrustAlgorithm, IterativeAlgorithm):
   """PageRank algorithm for computing social trust scores."""
   
   def __init__(self, config):
       super().__init__(config)
       self.damping = config.damping  # Damping factor
       self.max_iterations = config.iterations
   
   def compute(self, trust_matrix: sp.csr_matrix, 
               converters: List[str] = None,
               node_to_idx: Dict[str, int] = None,
               **kwargs) -> np.ndarray:
       """Compute PageRank scores."""
       n = trust_matrix.shape[0]
       
       if n == 0:
           return np.array([])
       
       logger.debug(f"Computing PageRank for {n} nodes")
       
       # Build row-stochastic matrix
       P, dangling = self._build_row_stochastic_matrix(trust_matrix)
       
       # Initialize personalization vector
       personalization = self._build_personalization_vector(n, converters, node_to_idx)
       
       # Initialize PageRank vector
       pagerank = personalization.copy()
       
       # Power iteration
       for iteration in range(self.max_iterations):
           prev = pagerank.copy()
           
           # Handle dangling nodes
           dangling_sum = prev[dangling].sum()
           
           # PageRank update: PR = d * P^T * PR + (1-d) * e/n + d * dangling_sum * e/n
           pagerank = self.damping * (P.T @ prev) + (1 - self.damping) * personalization
           pagerank += self.damping * dangling_sum * personalization
           
           # Check convergence
           if self.check_convergence(pagerank, prev):
               self.iterations_used = iteration + 1
               logger.debug(f"PageRank converged at iteration {iteration + 1}")
               break
       else:
           self.iterations_used = self.max_iterations
           logger.warning(f"PageRank did not converge in {self.max_iterations} iterations")
       
       return pagerank
   
   def _build_row_stochastic_matrix(self, trust_matrix: sp.csr_matrix) -> tuple:
       """Build row-stochastic matrix and identify dangling nodes."""
       # Calculate row sums
       row_sums = np.asarray(trust_matrix.sum(axis=1)).ravel()
       
       # Identify dangling nodes (nodes with no outgoing edges)
       dangling = (row_sums == 0)
       
       # Handle dangling nodes
       row_sums[dangling] = 1.0
       
       # Create diagonal matrix for normalization
       D_inv = sp.diags(1.0 / row_sums, format='csr')
       
       # P = D^(-1) * W
       P = D_inv @ trust_matrix
       
       return P, dangling
   
   def _build_personalization_vector(self, n: int, converters: List[str] = None,
                                   node_to_idx: Dict[str, int] = None) -> np.ndarray:
       """Build personalization vector for PageRank."""
       personalization = np.ones(n) / n
       
       # Give preference to converters
       if converters and node_to_idx:
           converter_indices = [node_to_idx[c] for c in converters if c in node_to_idx]
           if converter_indices:
               personalization = np.zeros(n)
               personalization[converter_indices] = 1.0 / len(converter_indices)
       
       return personalization