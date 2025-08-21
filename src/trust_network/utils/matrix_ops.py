"""Matrix operations utilities."""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class MatrixBuilder:
    """Efficient sparse matrix construction utilities."""
    
    def build_trust_matrix(self, edges: List, node_to_idx: Dict[str, int], n: int) -> sp.csr_matrix:
        """Build trust adjacency matrix from edges."""
        if n == 0:
            return sp.csr_matrix((0, 0))
        
        # Pre-allocate arrays for efficient construction
        row_indices = []
        col_indices = []
        data = []
        
        # Add edges
        for edge in edges:
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            row_indices.append(i)
            col_indices.append(j)
            data.append(edge.weight)
        
        # Add self-loops (self-trust = 1.0)
        for i in range(n):
            row_indices.append(i)
            col_indices.append(i)
            data.append(1.0)
        
        # Build sparse matrix directly in CSR format
        W = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float64
        )
        
        logger.debug(f"Built trust matrix: {W.shape}, nnz={W.nnz}")
        return W
    
    def build_balance_matrix(self, balances: Dict[str, Dict[str, float]], 
                           node_to_idx: Dict[str, int], n: int) -> sp.csr_matrix:
        """Build balance matrix from balance data."""
        if n == 0:
            return sp.csr_matrix((0, 0))
        
        row_indices = []
        col_indices = []
        data = []
        
        for holder, tokens in balances.items():
            if holder not in node_to_idx:
                continue
            h_idx = node_to_idx[holder]
            
            for token, amount in tokens.items():
                if token not in node_to_idx:
                    continue
                t_idx = node_to_idx[token]
                
                row_indices.append(h_idx)
                col_indices.append(t_idx)
                data.append(amount)
        
        # Build directly in CSR format
        B = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, n),
            dtype=np.float64
        )
        
        logger.debug(f"Built balance matrix: {B.shape}, nnz={B.nnz}")
        return B
    
    def build_rate_matrix(self, rates: Dict[str, Dict[str, float]], 
                         converters: Set[str], node_to_idx: Dict[str, int], n: int) -> np.ndarray:
        """Build conversion rate matrix."""
        num_converters = len(converters)
        
        if num_converters == 0 or n == 0:
            return np.zeros((0, 0))
        
        R = np.zeros((num_converters, n), dtype=np.float64)
        
        converter_list = sorted(converters)  # Ensure consistent ordering
        
        for i, converter in enumerate(converter_list):
            if converter not in rates:
                continue
            
            for token, rate in rates[converter].items():
                if token not in node_to_idx:
                    continue
                t_idx = node_to_idx[token]
                # Use direct assignment to numpy array (no sparse matrix issue)
                R[i, t_idx] = rate
        
        logger.debug(f"Built rate matrix: {R.shape}")
        return R


class SparseMatrixOps:
    """Optimized sparse matrix operations."""
    
    @staticmethod
    def normalize_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Normalize matrix rows to sum to 1."""
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        
        D_inv = sp.diags(1.0 / row_sums, format='csr')
        return D_inv @ matrix
    
    @staticmethod
    def normalize_columns(matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Normalize matrix columns to sum to 1."""
        col_sums = np.asarray(matrix.sum(axis=0)).ravel()
        col_sums[col_sums == 0] = 1.0
        
        D_inv = sp.diags(1.0 / col_sums, format='csr')
        return matrix @ D_inv
    
    @staticmethod
    def remove_diagonal_efficiently(matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Remove diagonal elements efficiently without modifying original matrix."""
        # Convert to COO for efficient filtering
        coo = matrix.tocoo()
        
        # Keep only off-diagonal elements
        mask = coo.row != coo.col
        
        # Build new matrix
        result = sp.csr_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])),
            shape=matrix.shape,
            dtype=matrix.dtype
        )
        
        return result
    
    @staticmethod
    def matrix_power(matrix: sp.csr_matrix, power: int) -> sp.csr_matrix:
        """Compute matrix power efficiently."""
        if power == 0:
            return sp.eye(matrix.shape[0], format='csr')
        elif power == 1:
            return matrix.copy()
        elif power < 0:
            raise ValueError("Negative powers not supported")
        
        # Binary exponentiation
        result = sp.eye(matrix.shape[0], format='csr')
        base = matrix.copy()
        
        while power > 0:
            if power % 2 == 1:
                result = result @ base
            base = base @ base
            power //= 2
        
        return result
    
    @staticmethod
    def efficient_matrix_vector_multiply(matrix: sp.csr_matrix, vector: np.ndarray) -> np.ndarray:
        """Efficient sparse matrix-vector multiplication."""
        # Use optimized BLAS routines when available
        return matrix @ vector