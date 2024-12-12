"""
Model head implementations for IC2Bert.

This module contains the head modules used in IC2Bert:
- MLMHead: Masked Language Modeling head
- SimpleLMHead: Simple Language Modeling head for binary classification
"""

from typing import Dict, Optional
import jax
import jax.numpy as jnp
import haiku as hk


class MLMHead(hk.Module):
    """
    Masked Language Modeling head for IC2Bert.
    
    This head predicts masked tokens in the input sequence.
    
    Attributes:
        vocab_size: Size of the token vocabulary
        embed_dim: Dimension of the input embeddings
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, name: Optional[str] = None):
        """
        Initialize MLM head.
        
        Args:
            vocab_size: Size of the token vocabulary
            embed_dim: Dimension of the input embeddings
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLM head.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        return hk.Linear(self.vocab_size)(x)


class SimpleLMHead(hk.Module):
    """
    Simple Language Modeling head for binary classification.
    
    This head performs global average pooling followed by binary classification.
    
    Attributes:
        embed_dim: Dimension of the input embeddings
        w_init: Weight initialization strategy
    """
    
    def __init__(self, embed_dim: int, name: Optional[str] = None):
        """
        Initialize Simple LM head.
        
        Args:
            embed_dim: Dimension of the input embeddings
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of the Simple LM head.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Dictionary containing 'logits' of shape [batch_size]
        """
        # Global average pooling
        x = jnp.mean(x, axis=1)
        logits = hk.Linear(1, w_init=self.w_init)(x)
        return {"logits": logits.squeeze(-1)}  # Remove the last dimension