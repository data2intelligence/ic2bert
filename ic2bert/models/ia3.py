"""
Implementation of IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) for IC2Bert.

This module provides an implementation of the IA3 adaptation layer, which modifies
the inner activations of a transformer model through learned scaling factors.
The implementation includes stability improvements through careful initialization
and normalization.
"""

from typing import Optional, Tuple, Dict
import jax
import jax.numpy as jnp
import haiku as hk


class IA3Layer(hk.Module):
    """Enhanced IA3 adaptation layer with improved initialization and scaling.
    
    This layer implements the IA3 adaptation technique, which modifies transformer
    activations through learned scaling factors. It includes stability improvements
    through careful initialization and normalization.
    
    Attributes:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        l_k: Learned key scaling parameters
        l_v: Learned value scaling parameters
        l_ff: Learned feed-forward scaling parameters
        layer_norm: Layer normalization for stability
    """
    
    def __init__(self, embed_dim: int, num_heads: int, name: Optional[str] = None):
        """Initialize IA3 layer.
        
        Args:
            embed_dim: Size of embeddings
            num_heads: Number of attention heads
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        def init_near_one(shape, dtype):
            """Initialize parameters with small deviation from ones."""
            return jnp.ones(shape, dtype) + jax.random.normal(
                hk.next_rng_key(), shape) * 0.01

        # Initialize parameters with small values around 1 for better stability
        self.l_k = hk.get_parameter(
            "l_k",
            [1, 1, self.head_dim],
            init=init_near_one
        )

        self.l_v = hk.get_parameter(
            "l_v",
            [1, 1, self.head_dim],
            init=init_near_one
        )

        self.l_ff = hk.get_parameter(
            "l_ff",
            [1, 1, embed_dim],
            init=init_near_one
        )

        # Add layer normalization for better training stability
        self.layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(
        self,
        k: Optional[jnp.ndarray],
        v: Optional[jnp.ndarray],
        ff_output: Optional[jnp.ndarray]
    ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Apply IA3 adaptation with improved stability.
        
        This method applies learned scaling factors to key (k), value (v), and
        feed-forward (ff_output) activations. The scaling is bounded using sigmoid
        activation for stability.
        
        Args:
            k: Optional key tensor to scale
            v: Optional value tensor to scale
            ff_output: Optional feed-forward output tensor to scale
            
        Returns:
            Tuple of (scaled_k, scaled_v, scaled_ff_output), where each element
            is None if the corresponding input was None
        """
        if k is not None:
            l_k_broadcast = jnp.reshape(self.l_k, (1, 1, 1, self.head_dim))
            k = k * jax.nn.sigmoid(l_k_broadcast)  # Use sigmoid for bounded scaling

        if v is not None:
            l_v_broadcast = jnp.reshape(self.l_v, (1, 1, 1, self.head_dim))
            v = v * jax.nn.sigmoid(l_v_broadcast)

        if ff_output is not None:
            l_ff_broadcast = jnp.reshape(self.l_ff, (1, 1, self.embed_dim))
            ff_output = ff_output * jax.nn.sigmoid(l_ff_broadcast)
            ff_output = self.layer_norm(ff_output)  # Apply normalization

        return k, v, ff_output

    def get_scaling_factors(self) -> Dict[str, jnp.ndarray]:
        """Get current scaling factors.
        
        Returns:
            Dictionary containing current sigmoid-activated scaling factors
            for key, value, and feed-forward components.
        """
        return {
            'key_scale': jax.nn.sigmoid(self.l_k),
            'value_scale': jax.nn.sigmoid(self.l_v),
            'ffn_scale': jax.nn.sigmoid(self.l_ff)
        }