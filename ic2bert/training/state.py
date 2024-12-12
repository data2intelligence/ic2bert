"""
Training state management for IC2Bert.

This module handles the creation and management of training states for both
training and evaluation phases of IC2Bert. It provides utilities for state
initialization and management.
"""

from typing import Dict, NamedTuple, Tuple, Callable
import optax
import jax
import jax.numpy as jnp
import haiku as hk

from ..config.model_config import IC2BertConfig
from ..models.bert import IC2Bert
from ..models.attention import MultiHeadAttention, SelfAttentionBlock


class TrainState(NamedTuple):
    """Training state container.
    
    Attributes:
        step: Current training step
        params: Model parameters
        opt_state: Optimizer state
        apply_fn: Model application function
        tx: Gradient transformation
        rng: Random number generator
    """
    step: int
    params: hk.Params
    opt_state: optax.OptState
    apply_fn: Callable
    tx: optax.GradientTransformation
    rng: jnp.ndarray


def build_bulk_rna_bert_forward_fn(model_config: IC2BertConfig) -> Callable:
    """Build forward function for IC2Bert.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Forward function for the model
    """
    def forward_fn(tokens: jnp.ndarray, is_training: bool = False) -> Dict:
        model = IC2Bert(config=model_config)
        return model(tokens=tokens, is_training=is_training)
    return forward_fn


def create_train_state(
    rng: jnp.ndarray,
    model_config: IC2BertConfig,
    input_shape: Tuple[int, ...],
    learning_rate: float = 1e-4,
    grad_clip_norm: float = 1.0
) -> TrainState:
    """Create initialized train state.
    
    Args:
        rng: Random number generator key
        model_config: Model configuration
        input_shape: Shape of input data (excluding batch dimension)
        learning_rate: Learning rate for optimizer
        grad_clip_norm: Gradient clipping norm
        
    Returns:
        Initialized TrainState instance
    """
    def forward_fn(tokens, is_training=False):
        model = IC2Bert(config=model_config)
        return model(tokens=tokens, is_training=is_training)

    model = hk.transform(forward_fn)

    # Create dummy batch for initialization
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1,) + input_shape, dtype=jnp.int32)

    # Initialize parameters
    params = model.init(init_rng, dummy_input, is_training=True)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate)
    )

    return TrainState(
        step=0,
        params=params,
        opt_state=tx.init(params),
        apply_fn=model.apply,
        tx=tx,
        rng=rng,
    )


def create_evaluation_state(
    pretrained_params: Dict,
    model_config: IC2BertConfig,
    learning_rate: float = 1e-5,
    grad_clip_norm: float = 1.0
) -> TrainState:
    """Create evaluation state from pretrained parameters.
    
    Args:
        pretrained_params: Pretrained model parameters
        model_config: Model configuration
        learning_rate: Learning rate for fine-tuning
        grad_clip_norm: Gradient clipping norm
        
    Returns:
        TrainState instance configured for evaluation/fine-tuning
    """
    rng = jax.random.PRNGKey(0)
    model = hk.transform(build_bulk_rna_bert_forward_fn(model_config))

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate)
    )
    opt_state = tx.init(pretrained_params)

    return TrainState(
        step=0,
        params=pretrained_params,
        opt_state=opt_state,
        apply_fn=model.apply,
        tx=tx,
        rng=rng
    )


def update_train_state(
    state: TrainState,
    grads: Dict,
    new_rng: jnp.ndarray
) -> TrainState:
    """Update training state with new gradients.
    
    Args:
        state: Current training state
        grads: Computed gradients
        new_rng: New random number generator key
        
    Returns:
        Updated TrainState instance
    """
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    return TrainState(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        apply_fn=state.apply_fn,
        tx=state.tx,
        rng=new_rng
    )


def get_current_lr(state: TrainState) -> float:
    """Get current learning rate from optimizer state.
    
    Args:
        state: Current training state
        
    Returns:
        Current learning rate
    """
    if hasattr(state.tx, 'learning_rate'):
        return state.tx.learning_rate
    # For chain of transformations, attempt to find learning rate
    if hasattr(state.tx, 'transforms'):
        for transform in state.tx.transforms:
            if hasattr(transform, 'learning_rate'):
                return transform.learning_rate
    return None