"""
Command-line argument parsing for IC2Bert.

This module handles the configuration and parsing of command-line arguments,
including dataset-size dependent defaults and validation.
"""

import argparse
from typing import Dict, Any
from ..data.dataset import get_dataset_size, DATASET_NAMES 
from ..utils.utils import validate_args 

# Default configurations based on dataset size
SIZE_BASED_DEFAULTS = {
    'batch_size': {
        'small': 8,
        'medium': 16,
        'large': 32,
        'default': 16
    },
    'learning_rate': {
        'small': 5e-5,
        'medium': 1e-4,
        'large': 2e-4,
        'default': 1e-4
    },
    'min_epochs': {
        'small': 40,
        'medium': 30,
        'large': 25,
        'default': 30
    },
    'patience': {
        'small': 20,
        'medium': 15,
        'large': 10,
        'default': 15
    }
}

# Model configuration defaults
MODEL_DEFAULTS = {
    'embed_dim': 128,
    'num_attention_heads': 4,
    'num_layers': 4,
    'ffn_embed_dim': 256
}

# Training configuration defaults
TRAINING_DEFAULTS = {
    'warmup_steps': 200,
    'weight_decay': 0.01,
    'dropout_rate': 0.1,
    'grad_clip_norm': 1.0
}

def get_size_based_default(param_name: str, dataset_size: str) -> Any:
    """Get default value based on dataset size.
    
    Args:
        param_name: Name of the parameter
        dataset_size: Size category of dataset
        
    Returns:
        Default value for the parameter
    """
    defaults = SIZE_BASED_DEFAULTS.get(param_name, {})
    return defaults.get(dataset_size, defaults.get('default'))

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="IC2Bert Parameter Ablation and LODOCV Study"
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['pretrain', 'evaluate'],
        help="Operation mode (pretrain or evaluate)"
    )
    required.add_argument(
        "--trial_num",
        type=int,
        required=True,
        help="Trial number"
    )
    required.add_argument(
        "--n_expressions_bins",
        type=int,
        required=True,
        help="Number of expression bins"
    )
    required.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files"
    )
    required.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory for dataset splits"
    )
    required.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for reproducibility"
    )
    required.add_argument(
        "--holdout_dataset",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to holdout during pretraining"
    )

    # Optional training arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training"
    )
    optional.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    optional.add_argument(
        "--min_epochs",
        type=int,
        help="Minimum number of epochs"
    )
    optional.add_argument(
        "--patience",
        type=int,
        help="Patience for early stopping"
    )
    optional.add_argument(
        "--warmup_steps",
        type=int,
        default=TRAINING_DEFAULTS['warmup_steps'],
        help="Number of warmup steps"
    )
    optional.add_argument(
        "--weight_decay",
        type=float,
        default=TRAINING_DEFAULTS['weight_decay'],
        help="Weight decay for regularization"
    )
    optional.add_argument(
        "--dropout_rate",
        type=float,
        default=TRAINING_DEFAULTS['dropout_rate'],
        help="Dropout rate"
    )
    optional.add_argument(
        "--grad_clip_norm",
        type=float,
        default=TRAINING_DEFAULTS['grad_clip_norm'],
        help="Gradient clipping norm"
    )

    # Model configuration
    model_config = parser.add_argument_group('model configuration')
    model_config.add_argument(
        "--embed_dim",
        type=int,
        default=MODEL_DEFAULTS['embed_dim'],
        help="Embedding dimension"
    )
    model_config.add_argument(
        "--num_attention_heads",
        type=int,
        default=MODEL_DEFAULTS['num_attention_heads'],
        help="Number of attention heads"
    )
    model_config.add_argument(
        "--num_layers",
        type=int,
        default=MODEL_DEFAULTS['num_layers'],
        help="Number of transformer layers"
    )
    model_config.add_argument(
        "--ffn_embed_dim",
        type=int,
        default=MODEL_DEFAULTS['ffn_embed_dim'],
        help="FFN embedding dimension"
    )

    # Mode-specific arguments
    mode_specific = parser.add_argument_group('mode-specific arguments')
    mode_specific.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory for saving checkpoints (required for pretrain mode)"
    )
    mode_specific.add_argument(
        "--pretrained_checkpoint",
        type=str,
        help="Path to pretrained checkpoint (required for evaluate mode)"
    )

    # Other optional flags
    optional.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with reduced parameters"
    )
    optional.add_argument(
        "--use_swa",
        action="store_true",
        help="Use Stochastic Weight Averaging"
    )
    optional.add_argument(
        "--use_augmentation",
        action="store_true",
        help="Use data augmentation"
    )

    return parser

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments.
    
    Returns:
        Parsed and validated arguments
        
    Raises:
        argparse.ArgumentError: If validation fails
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.mode == 'pretrain' and not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required for pretrain mode")
    if args.mode == 'evaluate' and not args.pretrained_checkpoint:
        parser.error("--pretrained_checkpoint is required for evaluate mode")

    # Set dataset-size dependent defaults
    dataset_size = get_dataset_size(args.holdout_dataset)
    
    # Set defaults if not provided
    for param_name in ['batch_size', 'learning_rate', 'min_epochs', 'patience']:
        if getattr(args, param_name) is None:
            setattr(args, param_name, get_size_based_default(param_name, dataset_size))

    # Validate all arguments
    try:
        validate_args(args)
    except ValueError as e:
        parser.error(str(e))

    return args