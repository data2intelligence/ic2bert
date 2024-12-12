"""
Tokenizer for gene expression data.

This module provides a tokenizer that converts continuous gene expression values
into discrete tokens using binning strategy. The tokenizer can be initialized
with either predefined bins or learn the binning from provided data.
"""

from typing import Optional, Tuple
import numpy as np


class BinnedExpressionTokenizer:
    """
    Tokenizer for converting gene expression values into discrete tokens.
    
    This tokenizer bins continuous gene expression values into discrete tokens,
    which can be used as input for the IC2Bert model. The binning can be either
    based on provided data or use default ranges.
    
    Attributes:
        _n_expressions_bins: Number of bins for discretization
        _gene_expression_bins: Array of bin edges
        min_value: Minimum value in the binning range
        max_value: Maximum value in the binning range
    """
    
    def __init__(self, n_expressions_bins: int, data: Optional[np.ndarray] = None):
        """
        Initialize the tokenizer.
        
        Args:
            n_expressions_bins: Number of bins to use for discretization
            data: Optional array of gene expression values to determine bin ranges.
                 If not provided, uses standard normal distribution range (-3, 3).
        """
        self._n_expressions_bins = n_expressions_bins

        if data is not None:
            self.min_value = np.min(data)
            self.max_value = np.max(data)
        else:
            # Default to standard normal distribution range
            self.min_value = -3
            self.max_value = 3

        self._gene_expression_bins = np.linspace(
            self.min_value,
            self.max_value,
            self._n_expressions_bins + 1
        )

    def tokenize(self, gene_expressions: np.ndarray) -> np.ndarray:
        """
        Convert gene expression values to tokens.
        
        Args:
            gene_expressions: Array of gene expression values
            
        Returns:
            Array of token indices starting from 0
        """
        tokens = np.digitize(gene_expressions, self._gene_expression_bins)
        return tokens.astype(np.int32) - 1  # Subtract 1 to start from 0
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert tokens back to approximate gene expression values.
        
        Args:
            tokens: Array of token indices
            
        Returns:
            Array of approximate gene expression values (bin centers)
        """
        # Add 1 to tokens since we subtracted 1 during tokenization
        bin_indices = tokens + 1
        # Get bin centers
        bin_centers = (self._gene_expression_bins[:-1] + self._gene_expression_bins[1:]) / 2
        return bin_centers[bin_indices]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (number of bins)."""
        return self._n_expressions_bins

    @property
    def mask_token_id(self) -> int:
        """Return the mask token ID (used for MLM)."""
        return self._n_expressions_bins

    def get_bin_edges(self) -> np.ndarray:
        """Return the bin edges used for discretization."""
        return self._gene_expression_bins.copy()

    def get_bin_centers(self) -> np.ndarray:
        """Return the centers of bins."""
        return (self._gene_expression_bins[:-1] + self._gene_expression_bins[1:]) / 2

    def get_token_range(self) -> Tuple[int, int]:
        """Return the range of possible token values."""
        return (0, self._n_expressions_bins - 1)

    def __repr__(self) -> str:
        """Return string representation of the tokenizer."""
        return (
            f"BinnedExpressionTokenizer(n_bins={self._n_expressions_bins}, "
            f"range=[{self.min_value:.2f}, {self.max_value:.2f}])"
        )