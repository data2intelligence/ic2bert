"""
Attention mechanisms for IC2Bert.

This module implements various attention mechanisms used in IC2Bert:
- MultiHeadAttention: Multi-head attention with explicit parameter initialization
- SelfAttentionBlock: Self-attention block with feed-forward network
"""

from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import haiku as hk
from .ia3 import IA3Layer


class MultiHeadAttention(hk.Module):
    """Multi-head attention with explicit parameter initialization.
    
    Implementation of multi-head attention with explicit parameter initialization
    for better control over the initialization process.
    
    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        w_init: Weight initialization function
        qkv_params: Parameters for query, key, value, and output projections
    """
    
    def __init__(self, embed_dim: int, num_heads: int, name: Optional[str] = None):
        """Initialize multi-head attention.
        
        Args:
            embed_dim: Size of embeddings
            num_heads: Number of attention heads
            name: Optional name for the module
            
        Raises:
            AssertionError: If embed_dim is not divisible by num_heads
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')

        def create_params(shape: Tuple[int, ...]) -> Dict[str, jnp.ndarray]:
            """Create parameters with explicit initialization."""
            return {
                "w": hk.get_parameter("w", shape, init=self.w_init),
                "b": hk.get_parameter("b", shape[-1:], init=jnp.zeros),
            }

        self.qkv_params = {
            "query": create_params([embed_dim, embed_dim]),
            "key": create_params([embed_dim, embed_dim]),
            "value": create_params([embed_dim, embed_dim]),
            "output": create_params([embed_dim, embed_dim]),
        }

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
                - embeddings: Output tensor of shape [batch_size, seq_len, embed_dim]
                - attention_weights: Attention weights tensor
        """
        batch_size = x.shape[0]

        def linear(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            """Apply linear transformation."""
            return jnp.dot(x, params["w"]) + params["b"]

        # Linear projections
        q = linear(x, self.qkv_params["query"])
        k = linear(x, self.qkv_params["key"])
        v = linear(x, self.qkv_params["value"])

        def reshape_head(x: jnp.ndarray) -> jnp.ndarray:
            """Reshape input for multi-head attention."""
            return x.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Reshape and transpose for attention
        q = reshape_head(q).transpose(0, 2, 1, 3)  # [B, H, T, D]
        k = reshape_head(k).transpose(0, 2, 1, 3)  # [B, H, T, D]
        v = reshape_head(v).transpose(0, 2, 1, 3)  # [B, H, T, D]

        # Compute attention scores
        scale = jnp.sqrt(self.head_dim).astype(x.dtype)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            scores = jnp.where(mask, scores, float('-inf'))

        # Compute attention weights and context
        weights = jax.nn.softmax(scores)
        context = jnp.matmul(weights, v)

        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)

        # Output projection
        output = linear(context, self.qkv_params["output"])

        return {"embeddings": output, "attention_weights": weights}


class SelfAttentionBlock(hk.Module):
    """Self-attention block using Haiku's built-in modules.
    
    Implements a self-attention block including multi-head attention,
    layer normalization, and feed-forward network.
    
    Attributes:
        attention: Multi-head attention module
        layer_norm1: First layer normalization
        layer_norm2: Second layer normalization
        ffn_dense1: First feed-forward layer
        ffn_dense2: Second feed-forward layer
        dropout_rate: Dropout rate
        use_ia3: Whether to use IA3 adaptation
    """
    
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        key_size: int,
        ffn_embed_dim: int,
        use_ia3: bool = False,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        """Initialize self-attention block.
        
        Args:
            num_heads: Number of attention heads
            embed_dim: Size of embeddings
            key_size: Size of key vectors
            ffn_embed_dim: Size of feed-forward network
            use_ia3: Whether to use IA3 adaptation
            dropout_rate: Dropout rate
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.attention = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform'),
            value_size=key_size,
            model_size=embed_dim,
            name="multi_head_attention"
        )

        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # Initialize FFN
        self.ffn_dense1 = hk.Linear(
            ffn_embed_dim,
            w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
        )
        self.ffn_dense2 = hk.Linear(
            embed_dim,
            w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
        )

        self.dropout_rate = dropout_rate
        self.use_ia3 = use_ia3

        if use_ia3:
            self.ia3 = IA3Layer(embed_dim=embed_dim, num_heads=num_heads)

    def __call__(
        self,
        x: jnp.ndarray,
        is_training: bool = False,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of self-attention block.
        
        Args:
            x: Input tensor
            is_training: Whether in training mode
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing:
                - embeddings: Output tensor
                - attention_weights: Attention weights
        """
        # Attention with residual connection
        normalized_x = self.layer_norm1(x)
        attention_output = self.attention(normalized_x, normalized_x, normalized_x, mask=attention_mask)

        if is_training and self.dropout_rate > 0:
            attention_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, attention_output)

        x = x + attention_output

        # FFN with residual connection
        normalized_x = self.layer_norm2(x)
        h = self.ffn_dense1(normalized_x)
        h = jax.nn.gelu(h)

        if is_training and self.dropout_rate > 0:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)

        h = self.ffn_dense2(h)

        if self.use_ia3:
            _, _, h = self.ia3(None, None, h)

        if is_training and self.dropout_rate > 0:
            h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)

        x = x + h

        return {
            "embeddings": x,
            "attention_weights": attention_output
        }