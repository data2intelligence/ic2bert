"""
Configuration classes for IC2Bert model, training, and data processing.

This module contains configuration classes for:
- Model architecture (IC2BertConfig)
- Training parameters (TrainingConfig)
- Data processing (DataConfig)
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


class IC2BertConfig:
    """Configuration class for IC2Bert model architecture.
    
    Attributes:
        n_genes: Number of genes in the model
        n_expressions_bins: Number of expression value bins
        embed_dim: Dimension of embeddings
        num_attention_heads: Number of attention heads
        ffn_embed_dim: Feed-forward network embedding dimension
        num_layers: Number of transformer layers
        init_gene_embed_dim: Initial gene embedding dimension
        key_size: Key size for attention (defaults to embed_dim // num_attention_heads)
        use_gene_embedding: Whether to use gene embeddings
        project_gene_embedding: Whether to project gene embeddings
        use_memory_efficient_attention: Whether to use memory-efficient attention
        use_gradient_checkpointing: Whether to use gradient checkpointing
        embeddings_layers_to_save: Tuple of embedding layer indices to save
        attention_layers_to_save: Tuple of attention layer indices to save
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self, 
        n_genes: int,
        n_expressions_bins: int,
        embed_dim: int,
        num_attention_heads: int,
        ffn_embed_dim: int,
        num_layers: int,
        init_gene_embed_dim: Optional[int] = None,
        key_size: Optional[int] = None,
        use_gene_embedding: bool = True,
        project_gene_embedding: bool = False,
        use_memory_efficient_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        embeddings_layers_to_save: Tuple[int, ...] = (),
        attention_layers_to_save: Tuple[int, ...] = (),
        dropout_rate: float = 0.1
    ):
        self.n_genes = n_genes
        self.n_expressions_bins = n_expressions_bins
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.num_layers = num_layers
        self.key_size = key_size if key_size is not None else embed_dim // num_attention_heads
        self.use_gene_embedding = use_gene_embedding
        self.project_gene_embedding = project_gene_embedding
        self.dropout_rate = dropout_rate
        
        if init_gene_embed_dim is None:
            self.init_gene_embed_dim = embed_dim if not project_gene_embedding else embed_dim // 2
        else:
            self.init_gene_embed_dim = init_gene_embed_dim
        
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embeddings_layers_to_save = embeddings_layers_to_save
        self.attention_layers_to_save = attention_layers_to_save

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'n_genes': self.n_genes,
            'n_expressions_bins': self.n_expressions_bins,
            'embed_dim': self.embed_dim,
            'num_attention_heads': self.num_attention_heads,
            'ffn_embed_dim': self.ffn_embed_dim,
            'num_layers': self.num_layers,
            'init_gene_embed_dim': self.init_gene_embed_dim,
            'key_size': self.key_size,
            'use_gene_embedding': self.use_gene_embedding,
            'project_gene_embedding': self.project_gene_embedding,
            'use_memory_efficient_attention': self.use_memory_efficient_attention,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'dropout_rate': self.dropout_rate
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'IC2BertConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    grad_clip_norm: float
    min_epochs: int
    patience: int
    dropout_rate: float
    use_swa: bool = False
    use_augmentation: bool = False
    grad_accumulation_steps: int = 1
    save_checkpoint_steps: int = 1000
    eval_steps: int = 100
    logging_steps: int = 10


@dataclass
class DataConfig:
    """Configuration for data processing."""
    datasets_dir: str
    gene_list_path: str
    splits_dir: str
    pretrain_ratio: float = 0.8
    random_seed: int = 42
    val_ratio: float = 0.1
    use_cache: bool = True
    max_seq_length: Optional[int] = None
    num_workers: int = 4


class FocalLossConfig(NamedTuple):
    """Configuration for focal loss parameters."""
    use_focal_loss: bool
    gamma: float
    alpha: Optional[float]
    class_weights: bool = False
    pos_weight: Optional[float] = None


def get_default_config(dataset_size: str = "medium") -> Tuple[IC2BertConfig, TrainingConfig, DataConfig]:
    """Get default configurations based on dataset size.
    
    Args:
        dataset_size: Size category of the dataset ("small", "medium", or "large")
    
    Returns:
        Tuple of (model_config, training_config, data_config)
    """
    # Base model configuration
    model_config = IC2BertConfig(
        n_genes=20000,  # placeholder, should be set based on actual data
        n_expressions_bins=256,
        embed_dim=128,
        num_attention_heads=4,
        ffn_embed_dim=256,
        num_layers=4,
        dropout_rate=0.1
    )
    
    # Size-specific training configurations
    training_configs = {
        "small": TrainingConfig(
            learning_rate=5e-5,
            batch_size=8,
            num_epochs=100,
            warmup_steps=100,
            weight_decay=0.015,
            grad_clip_norm=0.5,
            min_epochs=40,
            patience=20,
            dropout_rate=0.2
        ),
        "medium": TrainingConfig(
            learning_rate=1e-4,
            batch_size=16,
            num_epochs=50,
            warmup_steps=200,
            weight_decay=0.01,
            grad_clip_norm=1.0,
            min_epochs=30,
            patience=15,
            dropout_rate=0.15
        ),
        "large": TrainingConfig(
            learning_rate=2e-4,
            batch_size=32,
            num_epochs=30,
            warmup_steps=300,
            weight_decay=0.008,
            grad_clip_norm=1.0,
            min_epochs=25,
            patience=10,
            dropout_rate=0.1
        )
    }
    
    # Data configuration
    data_config = DataConfig(
        datasets_dir="./data/",
        gene_list_path="./intersect_prior.csv",
        splits_dir="./splits/",
        pretrain_ratio=0.8,
        val_ratio=0.1,
        use_cache=True,
        num_workers=4
    )
    
    return model_config, training_configs[dataset_size], data_config