"""
Core IC2Bert model implementation.

This module implements the main IC2Bert model, which combines gene embeddings,
expression embeddings, self-attention blocks, and prediction heads for
immunotherapy response prediction.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Optional, List

from ..config.model_config import IC2BertConfig
from .heads import MLMHead, SimpleLMHead
from .attention import SelfAttentionBlock


class IC2Bert(hk.Module):
    """IC2Bert model using Haiku's built-in modules.
    
    This model implements a BERT-like architecture specialized for gene expression data
    and immunotherapy response prediction. It combines gene and expression embeddings
    with transformer layers for processing.
    
    Attributes:
        _config: Model configuration
        _use_ia3: Whether to use IA3 adaptation
        w_init: Weight initialization strategy
        gene_embedding: Gene embedding layer
        expression_embedding: Expression embedding layer
        attention_blocks: List of self-attention blocks
        lm_head: Language modeling head for binary classification
        mlm_head: Masked language modeling head
    """
    
    def __init__(self, config: IC2BertConfig, use_ia3: bool = False, name: Optional[str] = None):
        """Initialize IC2Bert model.
        
        Args:
            config: Model configuration object
            use_ia3: Whether to use IA3 adaptation layers
            name: Optional name for the module
        """
        super().__init__(name=name)
        self._config = config
        self._use_ia3 = use_ia3
        self.w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Initialize attention blocks
        self.attention_blocks = self._initialize_attention_blocks()
        
        # Initialize heads
        self.lm_head = SimpleLMHead(embed_dim=self._config.embed_dim)
        self.mlm_head = MLMHead(
            vocab_size=self._config.n_expressions_bins + 1,
            embed_dim=self._config.embed_dim
        )

    def _initialize_embeddings(self) -> None:
        """Initialize gene and expression embedding layers."""
        # Gene embedding
        self.gene_embedding = hk.Embed(
            vocab_size=self._config.n_genes,
            embed_dim=self._config.init_gene_embed_dim,
            w_init=self.w_init,
            name="gene_embedding"
        )
        
        # Gene projection if needed
        if self._config.project_gene_embedding:
            self.gene_projection = hk.Linear(
                self._config.embed_dim,
                w_init=self.w_init,
                name="gene_projection"
            )
        
        # Expression embedding
        self.expression_embedding = hk.Embed(
            vocab_size=self._config.n_expressions_bins,
            embed_dim=self._config.embed_dim,
            w_init=self.w_init,
            name="expression_embedding"
        )

    def _initialize_attention_blocks(self) -> List[SelfAttentionBlock]:
        """Initialize self-attention blocks."""
        return [
            SelfAttentionBlock(
                num_heads=self._config.num_attention_heads,
                embed_dim=self._config.embed_dim,
                key_size=self._config.key_size,
                ffn_embed_dim=self._config.ffn_embed_dim,
                use_ia3=self._use_ia3,
                dropout_rate=self._config.dropout_rate,
                name=f"self_attention_block_{i}"
            )
            for i in range(self._config.num_layers)
        ]

    def __call__(
        self,
        tokens: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        is_training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of the IC2Bert model.
        
        Args:
            tokens: Input token indices of shape [batch_size, seq_len]
            attention_mask: Optional attention mask
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing:
                - mlm_logits: Masked language modeling logits
                - logits: Binary classification logits
                - embeddings_{i}: Optional intermediate embeddings
                - attention_map_layer_{i}: Optional attention maps
        """
        if len(tokens.shape) == 1:
            tokens = tokens.reshape(1, -1)
        
        batch_size, seq_len = tokens.shape
        outs = {}
        
        # Apply embeddings
        x = self._apply_embeddings(tokens, batch_size, seq_len)
        
        # Apply dropout during training
        if is_training and self._config.dropout_rate > 0:
            x = hk.dropout(hk.next_rng_key(), self._config.dropout_rate, x)
        
        # Process through attention blocks
        x = self._process_attention_blocks(
            x, is_training, attention_mask, batch_size, seq_len, outs
        )
        
        # Apply heads
        outs["mlm_logits"] = self.mlm_head(x)
        lm_outputs = self.lm_head(x)
        outs["logits"] = lm_outputs["logits"]
        
        return outs

    def _apply_embeddings(
        self,
        tokens: jnp.ndarray,
        batch_size: int,
        seq_len: int
    ) -> jnp.ndarray:
        """Apply gene and expression embeddings.
        
        Args:
            tokens: Input token indices
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Combined embeddings tensor
        """
        x = self.expression_embedding(tokens)
        
        if self._config.use_gene_embedding:
            gene_indices = jnp.arange(self._config.n_genes)
            gene_embeds = self.gene_embedding(gene_indices)
            
            if self._config.project_gene_embedding:
                gene_embeds = self.gene_projection(gene_embeds)
            
            gene_embeds = jnp.expand_dims(gene_embeds, axis=0)
            gene_embeds = jnp.broadcast_to(gene_embeds, (batch_size, seq_len, gene_embeds.shape[-1]))
            x = x + gene_embeds
            
        return x

    def _process_attention_blocks(
        self,
        x: jnp.ndarray,
        is_training: bool,
        attention_mask: Optional[jnp.ndarray],
        batch_size: int,
        seq_len: int,
        outs: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Process input through attention blocks.
        
        Args:
            x: Input tensor
            is_training: Whether in training mode
            attention_mask: Optional attention mask
            batch_size: Batch size
            seq_len: Sequence length
            outs: Output dictionary to store intermediate values
            
        Returns:
            Processed tensor
        """
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, 1, seq_len, seq_len))
            
        for i, block in enumerate(self.attention_blocks):
            output = block(x, is_training=is_training, attention_mask=attention_mask)
            x = output["embeddings"]
            
            if (i + 1) in self._config.embeddings_layers_to_save:
                outs[f"embeddings_{i + 1}"] = output["embeddings"]
            if (i + 1) in self._config.attention_layers_to_save:
                outs[f"attention_map_layer_{i + 1}"] = output["attention_weights"]
                
        return x