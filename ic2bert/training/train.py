"""
Training functionality for IC2Bert.

This module implements training pipelines for:
1. Pretraining
2. Transfer learning
3. Downstream evaluation

Each pipeline includes training loops, optimization, and evaluation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
from jax.tree_util import tree_leaves
from sklearn.metrics import roc_auc_score

from .state import TrainState, create_train_state, create_evaluation_state
from .metrics import (
    create_mlm_masks, mlm_loss_fn, calculate_reconstruction_accuracy,
    zero_shot_evaluate
)
from ..config.model_config import IC2BertConfig
from ..data.dataset import DATASET_NAMES
from ..data.tokenizer import BinnedExpressionTokenizer
from ..data.splits import split_pretrain_data
from ..utils.utils import get_dataset_characteristics, save_checkpoint, save_best_checkpoint

logger = logging.getLogger(__name__)


# Model initialization and forward pass
def build_bulk_rna_bert_forward_fn(model_config: IC2BertConfig):
    """Build forward function with proper initialization."""
    def forward_fn(tokens: jnp.ndarray, is_training: bool = False,
                  attention_mask: Optional[jnp.ndarray] = None):
        model = IC2Bert(config=model_config)
        return model(tokens=tokens, attention_mask=attention_mask, is_training=is_training)
    return forward_fn


def pretrain_step(state: TrainState, batch: Dict):
    """Single pretraining step."""
    rng, new_rng = jax.random.split(state.rng)

    def loss_fn(params):
        masked_tokens, mlm_labels = create_mlm_masks(rng, batch['tokens'])
        outputs = state.apply_fn(params, rng, masked_tokens, is_training=True)
        loss = mlm_loss_fn(outputs['mlm_logits'], mlm_labels)
        accuracy = calculate_reconstruction_accuracy(outputs['mlm_logits'], mlm_labels)
        return loss, (outputs['mlm_logits'], mlm_labels, accuracy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, labels, accuracy)), grads = grad_fn(state.params)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state._replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng
    )

    return new_state, loss, optax.global_norm(grads), optax.global_norm(updates), accuracy

def val_step_fn(state: TrainState, batch: Dict):
    """Validation step function."""
    rng = jax.random.PRNGKey(0)  # Fixed RNG for consistency
    masked_tokens, mlm_labels = create_mlm_masks(rng, batch['tokens'])
    outputs = state.apply_fn(state.params, rng, masked_tokens, is_training=False)
    loss = mlm_loss_fn(outputs['mlm_logits'], mlm_labels)
    accuracy = calculate_reconstruction_accuracy(outputs['mlm_logits'], mlm_labels)
    return loss, accuracy

val_step = jax.jit(val_step_fn)


def pretrain(
    pretrain_data: Dict,
    model_config: IC2BertConfig,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: str,
    args,
    save_every: int = 100,
    state: Optional[TrainState] = None,
    start_epoch: int = 0
):
    """Pretrain the model with reconstruction accuracy tracking."""
    if state is None:
        rng = jax.random.PRNGKey(args.random_seed)  # Use provided random seed
        model = hk.transform(build_bulk_rna_bert_forward_fn(model_config))
        state = create_train_state(rng, model, learning_rate, pretrain_data['tokens'].shape[1:])

    split_data = split_pretrain_data(pretrain_data)
    train_data = split_data['train']
    val_data = split_data['val']

    total_tokens = 0
    target_tokens = int(12_000_000_000 * (938 / 19042))
    best_val_accuracy = float('-inf')
    best_reconstruction_accuracy = float('-inf')
    best_epoch = 0
    start_time = datetime.now()
    current_epoch = start_epoch

    # Create metrics dictionary for tracking
    metrics = {
        'trial': args.trial_num,
        'n_bins': args.n_expressions_bins,
        'split_seed': args.random_seed,
        'reconstruction_accuracies': [],
        'val_accuracies': [],
        'epochs': [],
        'best_epoch': None,
        'best_reconstruction_accuracy': None,
        'best_val_accuracy': None
    }

    try:
        while total_tokens < target_tokens:
            current_epoch += 1
            epoch_start_time = datetime.now()
            tokens_this_epoch = 0

            # Training
            state = state._replace(rng=jax.random.split(state.rng)[0])
            permutation = jax.random.permutation(state.rng, train_data['tokens'].shape[0])
            tokens_shuffled = train_data['tokens'][permutation]

            total_loss = 0.0
            total_accuracy = 0.0
            total_reconstruction_accuracy = 0.0
            num_batches = 0

            for i in range(0, tokens_shuffled.shape[0], batch_size):
                batch = {'tokens': tokens_shuffled[i:i+batch_size]}
                try:
                    state, loss, grad_norm, clipped_grad_norm, accuracy = pretrain_step(state, batch)

                    if jnp.isnan(loss):
                        raise ValueError(f"NaN loss encountered at step {state.step}")

                    total_loss += loss
                    total_accuracy += accuracy
                    total_reconstruction_accuracy += accuracy
                    num_batches += 1
                    batch_tokens = batch['tokens'].size
                    total_tokens += batch_tokens
                    tokens_this_epoch += batch_tokens

                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue

                if num_batches % 100 == 0:
                    logger.info(f"Epoch {current_epoch} - Batch {num_batches}")
                    logger.info(f"  Tokens processed: {total_tokens:,}/{target_tokens:,}")
                    logger.info(f"  Current loss: {loss:.4f}")
                    logger.info(f"  Current accuracy: {accuracy:.4f}")

            avg_train_loss = total_loss / max(num_batches, 1)
            avg_train_accuracy = total_accuracy / max(num_batches, 1)
            avg_reconstruction_accuracy = total_reconstruction_accuracy / max(num_batches, 1)

            # Validation
            val_losses = []
            val_accuracies = []
            for i in range(0, val_data['tokens'].shape[0], batch_size):
                val_batch = {'tokens': val_data['tokens'][i:i+batch_size]}
                try:
                    val_loss, val_accuracy = val_step_fn(state, val_batch)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue

            avg_val_loss = jnp.mean(jnp.array(val_losses)) if val_losses else float('inf')
            avg_val_accuracy = jnp.mean(jnp.array(val_accuracies)) if val_accuracies else float('-inf')

            # Update metrics
            metrics['reconstruction_accuracies'].append(float(avg_reconstruction_accuracy))
            metrics['val_accuracies'].append(float(avg_val_accuracy))
            metrics['epochs'].append(current_epoch)

            # Update best metrics
            if avg_reconstruction_accuracy > best_reconstruction_accuracy:
                best_reconstruction_accuracy = avg_reconstruction_accuracy
                metrics['best_reconstruction_accuracy'] = float(best_reconstruction_accuracy)

            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                best_epoch = current_epoch
                metrics['best_epoch'] = best_epoch
                metrics['best_val_accuracy'] = float(best_val_accuracy)
                save_best_checkpoint(state, current_epoch, checkpoint_dir, best_val_accuracy, args)
                logger.info(f"  New best validation accuracy: {best_val_accuracy:.4f}")

            # Log epoch completion
            epoch_duration = datetime.now() - epoch_start_time
            logger.info(f"Epoch {current_epoch} completed in {epoch_duration}:")
            logger.info(f"  Tokens this epoch: {tokens_this_epoch:,}")
            logger.info(f"  Total tokens: {total_tokens:,}/{target_tokens:,}")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Reconstruction Accuracy: {avg_reconstruction_accuracy:.4f}")
            logger.info(f"  Validation Accuracy: {avg_val_accuracy:.4f}")

            # Save periodic checkpoint and metrics
            if current_epoch % save_every == 0:
                save_checkpoint(state, current_epoch, checkpoint_dir, args)
                metrics_path = os.path.join(args.output_dir, 'pretrain_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise
    finally:
        # Save final metrics
        metrics_path = os.path.join(args.output_dir, 'pretrain_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save final checkpoint
        try:
            final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pkl")
            save_checkpoint(state, current_epoch, checkpoint_dir, args)
            logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {str(e)}")

        total_duration = datetime.now() - start_time
        logger.info(f"Training completed in {total_duration}")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        logger.info(f"Best reconstruction accuracy: {best_reconstruction_accuracy:.4f}")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Total tokens processed: {total_tokens:,}/{target_tokens:,}")

    return state, metrics

def run_pretraining(
    pretrain_datasets: List[Dict],
    all_genes: List[str],
    args,
    experiment_dir: str
) -> Tuple[TrainState, 'BinnedExpressionTokenizer']:
    """Run pretraining pipeline."""
    """Run pretraining pipeline.

    Args:
        pretrain_datasets: List of datasets for pretraining
        all_genes: List of all genes
        args: Command line arguments
        experiment_dir: Directory for experiment outputs
    """
    logger.info("Initializing pretraining...")

    # Create model config
    model_config = IC2BertConfig(
        n_genes=len(all_genes),
        n_expressions_bins=args.n_expressions_bins,
        embed_dim=128,
        num_attention_heads=4,
        ffn_embed_dim=256,
        num_layers=4,
        use_gene_embedding=True,
        project_gene_embedding=True,
        init_gene_embed_dim=64,
        dropout_rate=0.1
    )

    # Save model configuration
    with open(os.path.join(experiment_dir, 'model_config.json'), 'w') as f:
        json.dump(vars(model_config), f, indent=2)

    # Prepare pretraining data
    all_train_data = []
    for dataset in pretrain_datasets:
        all_train_data.append(dataset['train']['X'])

    pretrain_data = np.vstack(all_train_data)

    # Initialize tokenizer
    tokenizer = BinnedExpressionTokenizer(
        n_expressions_bins=model_config.n_expressions_bins,
        data=pretrain_data
    )

    # Tokenize pretraining data
    pretrain_tokens = tokenizer.tokenize(pretrain_data)

    # Initialize model state
    rng = jax.random.PRNGKey(args.random_seed)
    state = create_train_state(
        rng=rng,
        model_config=model_config,
        input_shape=pretrain_tokens.shape[1:]
    )

    # Run pretraining
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    state, metrics = pretrain(
        pretrain_data={'tokens': pretrain_tokens},
        model_config=model_config,
        num_epochs=100 if not args.test_mode else 2,
        batch_size=32 if not args.test_mode else 4,
        learning_rate=1e-4,
        checkpoint_dir=checkpoint_dir,
        args=args,
        save_every=5,
        state=state,
        start_epoch=0
    )

    # Save metrics
    metrics_path = os.path.join(experiment_dir, 'pretrain_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return state, tokenizer

def transfer_learn(
    pretrained_state: TrainState,
    dataset: Dict,
    dataset_name: str,
    tokenizer: 'BinnedExpressionTokenizer',
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    experiment_dir: str,
    use_ia3: bool = False
):
    """Transfer learning with dataset-specific optimization."""
    from jax.tree_util import tree_leaves
    start_time = datetime.now()
    
    # Calculate dataset sizes
    train_size = len(dataset['train']['X'])
    test_size = len(dataset['test']['X'])
    dataset_size = train_size + test_size


    # Calculate class ratio
    train_pos_ratio = float(np.mean(dataset['train']['y']))
    
    # Get dataset characteristics with proper focal loss parameters
    dataset_chars, closest_match = get_dataset_characteristics(
        dataset_name=dataset_name,
        train_pos_ratio=train_pos_ratio,
        dataset_size=dataset_size
    )
    
    # Use matched dataset name
    actual_dataset_name = closest_match if closest_match != dataset_name else dataset_name
    
    # Log dataset information
    if closest_match != dataset_name:
        logger.warning(
            f"Dataset identity mismatch detected:\n"
            f"  Provided name: {dataset_name}\n"
            f"  Size-based match: {closest_match}\n"
            f"  Actual size: {dataset_size} (train={train_size}, test={test_size})\n"
            f"  Class distribution: {train_pos_ratio:.3f} positive ratio"
        )

    # Create checkpoint directory ONCE using actual_dataset_name
    checkpoint_dir = os.path.join(experiment_dir, 'transfer_checkpoints', actual_dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Store checkpoint paths for consistency
    checkpoint_paths = {
        'dir': checkpoint_dir,
        'best': os.path.join(checkpoint_dir, 'best_checkpoint.pkl'),
        'final': os.path.join(checkpoint_dir, 'final_checkpoint.pkl'),
        'periodic': lambda epoch: os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
    }    
   
    # Enhanced size-based hyperparameter selection
    is_small = dataset_size < 50
    is_medium = 50 <= dataset_size < 100
    is_large = dataset_size >= 100

    size_category = "small" if is_small else "medium" if is_medium else "large"
    logger.info(f"Dataset size category: {size_category} (based on total size: {dataset_size})")
    
    # Base hyperparameters by size
    base_hyperparams = {
        'small': {
            'batch_size': 8,
            'learning_rate': 1e-4,  # Increased from learning_rate * 0.5
            'min_epochs': 40,
            'patience': 20,
            'warmup_steps': 100,
            'weight_decay': 0.015,
            'dropout_rate': 0.2,
            'clip_norm': 0.5,
            'decay_factor': lambda step: 0.95 ** (step / 100)
        },
        'medium': {
            'batch_size': 16,
            'learning_rate': 2e-4,  # Increased from learning_rate
            'min_epochs': 30,
            'patience': 15,
            'warmup_steps': 200,
            'weight_decay': 0.01,
            'dropout_rate': 0.15,
            'clip_norm': 0.75,
            'decay_factor': lambda step: 0.97 ** (step / 150)
        },
        'large': {
            'batch_size': 32,
            'learning_rate': 5e-4,  # Increased from learning_rate * 2.0
            'min_epochs': 25,
            'patience': 10,
            'warmup_steps': 300,
            'weight_decay': 0.008,
            'dropout_rate': 0.1,
            'clip_norm': 1.0,
            'decay_factor': lambda step: 0.98 ** (step / 200)
        }
    }[size_category]

    # Combine all parameters with proper precedence
    hyperparams = {**base_hyperparams, **dataset_chars['special_params']}

    # Adjust learning rate using learning_rate_factor if specified
    if 'learning_rate_factor' in hyperparams:
        hyperparams['learning_rate'] *= hyperparams['learning_rate_factor']

    # Log hyperparameters
    logger.info(f"Training parameters for {size_category} dataset:")
    logger.info(f"  Learning rate: {hyperparams['learning_rate']:.2e}")
    logger.info(f"  Batch size: {hyperparams['batch_size']}")
    logger.info(f"  Min epochs: {hyperparams['min_epochs']}")
    logger.info(f"  Patience: {hyperparams['patience']}")
    logger.info(f"  Weight decay: {hyperparams['weight_decay']}")
    logger.info(f"  Dropout rate: {hyperparams['dropout_rate']}")
    logger.info(f"  Gradient clip norm: {hyperparams['clip_norm']}")
    if hyperparams.get('use_focal_loss', False):
        logger.info("\nFocal Loss Configuration:")
        logger.info(f"  Gamma: {hyperparams['focal_gamma']:.2f}")
        logger.info(f"  Alpha: {hyperparams['focal_alpha']:.3f}")
        logger.info(f"  Class weights enabled: {hyperparams.get('class_weights', False)}")
    
    # Learning rate schedule
    def lr_schedule(step):
        warmup_factor = jnp.minimum(step / hyperparams['warmup_steps'], 1.0)
        decay_factor = hyperparams['decay_factor'](step)
        initial_scale = 10.0
        lr = hyperparams['learning_rate'] * initial_scale * warmup_factor * decay_factor
        return jnp.maximum(lr, 1e-6)  # Set minimum learning rate

    def cyclic_lr_schedule(step):
        cycle_length = 1000
        cycle_position = step % cycle_length
        cycle_factor = 0.5 * (1 + jnp.cos(jnp.pi * cycle_position / cycle_length))
        base_lr = lr_schedule(step)
        return base_lr * (0.1 + 0.9 * cycle_factor)  # Vary between 10% and 100% of base rate

    def riaz_lr_schedule(step, hyperparams):
        """Conservative learning rate schedule for Riaz dataset."""
        warmup_steps = hyperparams['warmup_steps']
        max_lr = hyperparams['max_learning_rate']
        min_lr = hyperparams['min_learning_rate']
        warmup_ratio = hyperparams['warmup_ratio']
        
        # Initial warmup phase
        if step < warmup_steps:
            warmup_progress = step / warmup_steps
            return min_lr + (max_lr * warmup_ratio - min_lr) * warmup_progress
        
        # Exponential decay after warmup
        decay_rate = 0.95
        decay_steps = 100
        current_step = step - warmup_steps
        decay_factor = decay_rate ** (current_step / decay_steps)
        
        lr = max_lr * warmup_ratio * decay_factor
        return jnp.clip(lr, min_lr, max_lr)
    
    # Data setup
    X_train, y_train = dataset['train']['X'], dataset['train']['y']
    X_test, y_test = dataset['test']['X'], dataset['test']['y']
    tokens_train = tokenizer.tokenize(X_train)
    tokens_test = tokenizer.tokenize(X_test)
    
    # Dataset info
    dataset_info = {
        'dataset_name': actual_dataset_name,
        'dataset_size': dataset_size,
        'train_size': train_size,
        'test_size': test_size,
        'train_positive_ratio': float(y_train.mean()),
        'test_positive_ratio': float(y_test.mean()),
        'size_category': size_category,
        'cancer_type': dataset_chars['cancer_type'],
        'signature': dataset_chars['signature'],
        'focal_loss_config': {
            'enabled': hyperparams.get('use_focal_loss', False),
            'gamma': hyperparams.get('focal_gamma', None),
            'alpha': hyperparams.get('focal_alpha', None),
        },
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }     

    with open(os.path.join(checkpoint_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Model configuration
    model_config = IC2BertConfig(
        n_genes=X_train.shape[1],
        n_expressions_bins=tokenizer._n_expressions_bins,
        embed_dim=128,
        num_attention_heads=4,
        ffn_embed_dim=256,
        num_layers=4,
        dropout_rate=hyperparams['dropout_rate'],
        use_gene_embedding=True,
        project_gene_embedding=True,
        init_gene_embed_dim=64
    )
    
    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(hyperparams['clip_norm']),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=hyperparams['weight_decay']
        )
    )
    
    # Initialize state
    state = TrainState(
        step=0,
        params=pretrained_state.params,
        opt_state=tx.init(pretrained_state.params),
        apply_fn=pretrained_state.apply_fn,
        tx=tx,
        rng=jax.random.PRNGKey(0)
    )

    # Loss function
    def loss_fn(params, batch_tokens, batch_labels, is_training=True):
        """Loss function with enhanced focal loss handling."""
        outputs = state.apply_fn(params, state.rng, batch_tokens, is_training=is_training)
        logits = outputs['logits']
        
        try:
            if hyperparams.get('use_focal_loss', False):
                # Validate focal loss parameters
                if 'focal_gamma' not in hyperparams or 'focal_alpha' not in hyperparams:
                    logger.warning("Focal loss enabled but parameters missing, falling back to BCE")
                    return optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels).mean()
                
                probs = jax.nn.sigmoid(logits)
                pt = batch_labels * probs + (1 - batch_labels) * (1 - probs)
                focal_weight = (1 - pt) ** hyperparams['focal_gamma']
                alpha = hyperparams['focal_alpha']
                alpha_weight = alpha * batch_labels + (1 - alpha) * (1 - batch_labels)
                loss = -alpha_weight * focal_weight * jnp.log(pt + 1e-8)
            elif hyperparams.get('class_weights', False):
                # Validate class weights
                if 'pos_weight' not in hyperparams:
                    logger.warning("Class weights enabled but pos_weight missing, falling back to BCE")
                    return optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels).mean()
                
                weights = batch_labels * hyperparams['pos_weight'] + (1 - batch_labels)
                loss = weights * optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels)
            else:
                loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels)
            
            base_loss = jnp.mean(loss)
            
            # Add L2 regularization for small datasets
            if is_small:
                l2_loss = hyperparams['weight_decay'] * sum(
                    jnp.sum(jnp.square(p)) for p in tree_leaves(params)
                ) / dataset_size
                return base_loss + l2_loss
            
            return base_loss
            
        except Exception as e:
            logger.error(f"Error in loss computation: {str(e)}")
            logger.error(f"Falling back to standard BCE loss")
            return optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_labels).mean()
    
    # Training step
    def train_step(state, batch):
        def compute_loss(params):
            return loss_fn(params, batch['tokens'], batch['labels'], is_training=True)
            
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        
        new_state = state._replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            rng=jax.random.split(state.rng)[0]
        )
        
        return new_state, loss
    
    # Define save_model_checkpoint with the correct path
    def save_model_checkpoint(state, metrics, checkpoint_name):
        """Save model checkpoint with metrics using correct dataset name."""
        try:
            if checkpoint_name.startswith('checkpoint_epoch_'):
                checkpoint_path = checkpoint_paths['periodic'](int(checkpoint_name.split('_')[-1]))
            else:
                checkpoint_path = checkpoint_paths[checkpoint_name.replace('_checkpoint', '')]
                
            metrics_path = checkpoint_path.replace('.pkl', '_metrics.json')
            
            checkpoint_data = {
                'state': {
                    'step': state.step,
                    'params': state.params,
                    'opt_state': state.opt_state,
                    'rng': state.rng,
                },
                'model_config': model_config.__dict__,
                'training_config': {
                    'learning_rate': hyperparams['learning_rate'],
                    'batch_size': hyperparams['batch_size'],
                    'weight_decay': hyperparams['weight_decay'],
                    'clip_norm': hyperparams['clip_norm'],
                    'original_dataset_name': dataset_name,
                    'actual_dataset_name': actual_dataset_name,
                    'size_category': size_category,
                    'total_parameters': sum(x.size for x in tree_leaves(state.params)),
                    'loss_config': {
                        'use_focal_loss': hyperparams.get('use_focal_loss', False),
                        'focal_gamma': hyperparams.get('focal_gamma', None),
                        'focal_alpha': hyperparams.get('focal_alpha', None),
                        'class_weights': hyperparams.get('class_weights', False),
                        'pos_weight': hyperparams.get('pos_weight', None)
                    }
                },
                'metrics': metrics,
                'dataset_info': dataset_info,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Saved {checkpoint_name} checkpoint to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    # Training loop
    best_auc = 0.0
    best_params = None
    best_state = None
    best_checkpoint_path = None
    patience_counter = 0
    training_history = []
    
    try:
        for epoch in range(num_epochs):
            # Shuffle training data
            rng, shuffle_rng = jax.random.split(state.rng)
            permutation = jax.random.permutation(shuffle_rng, tokens_train.shape[0])
            tokens_shuffled = tokens_train[permutation]
            y_shuffled = y_train[permutation]
            
            # Training
            epoch_losses = []
            for i in range(0, len(tokens_shuffled), hyperparams['batch_size']):
                batch_idx = slice(i, min(i + hyperparams['batch_size'], len(tokens_shuffled)))
                batch = {
                    'tokens': tokens_shuffled[batch_idx],
                    'labels': y_shuffled[batch_idx]
                }
                try:
                    state, loss = train_step(state, batch)
                    epoch_losses.append(float(loss))
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            if not epoch_losses:
                logger.warning(f"No valid losses in epoch {epoch + 1}, skipping...")
                continue
                
            mean_epoch_loss = float(np.mean(epoch_losses))
            
            # Evaluation
            try:
                test_logits = state.apply_fn(
                    state.params,
                    state.rng,
                    tokens_test,
                    is_training=False
                )['logits']
                test_probs = jax.nn.sigmoid(test_logits)
                current_auc = roc_auc_score(y_test, test_probs)
            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
                continue
            
            # Create metrics dictionary
            epoch_metrics = {
                'epoch': epoch + 1,
                'loss': mean_epoch_loss,
                'auc': float(current_auc),
                'learning_rate': float(lr_schedule(state.step)),
                'best_auc': float(best_auc),
                'n_batches': len(epoch_losses),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            training_history.append(epoch_metrics)
            
            # Periodic checkpoint saving
            if (epoch + 1) % 10 == 0:

                current_lr = float(lr_schedule(state.step))
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {mean_epoch_loss:.4f}, "
                    f"AUC: {current_auc:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Best AUC: {best_auc:.4f}"
                )

                try:
                    save_model_checkpoint(state, epoch_metrics, f"checkpoint_epoch_{epoch+1}")
                except Exception as e:
                    logger.error(f"Error saving periodic checkpoint: {str(e)}")
            
            # Update best model
            if current_auc > best_auc:
                best_auc = current_auc
                best_params = state.params
                best_state = state
                patience_counter = 0
                
                try:
                    best_checkpoint_path = save_model_checkpoint(
                        state,
                        {**epoch_metrics, 'checkpoint_type': 'best'},
                        "best_checkpoint"
                    )
                except Exception as e:
                    logger.error(f"Error saving best checkpoint: {str(e)}")
            else:
                patience_counter += 1
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {mean_epoch_loss:.4f}, "
                    f"AUC: {current_auc:.4f}, "
                    f"LR: {lr_schedule(state.step):.2e}, "
                    f"Best AUC: {best_auc:.4f}"
                )
            
            # Early stopping
            if patience_counter >= hyperparams['patience'] and epoch >= hyperparams['min_epochs']:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final evaluation using best parameters
        try:
            final_logits = state.apply_fn(
                best_params,
                state.rng,
                tokens_test,
                is_training=False
            )['logits']
            final_probs = jax.nn.sigmoid(final_logits)
            final_auc = roc_auc_score(y_test, final_probs)
        except Exception as e:
            logger.error(f"Error in final evaluation: {str(e)}")
            final_auc = best_auc
        
        # Calculate final metrics
        final_metrics = {
            'final_auc': float(final_auc),
            'best_auc': float(best_auc),
            'total_epochs': epoch + 1,
            'early_stopped': patience_counter >= hyperparams['patience'],
            'training_history': training_history,
            'dataset_info': dataset_info,
            'hyperparameters': {k: v for k, v in hyperparams.items() 
                              if not callable(v)},  # Exclude lambda functions
            'total_steps': state.step,
            'final_learning_rate': float(lr_schedule(state.step)),
            'convergence_epoch': epoch + 1 - patience_counter,
            'training_time': str(datetime.now() - start_time),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        try:
            final_checkpoint_path = save_model_checkpoint(
                best_state,  # Save the best state as final
                final_metrics,
                "final_checkpoint"
            )
        except Exception as e:
            logger.error(f"Error saving final checkpoint: {str(e)}")
            final_checkpoint_path = best_checkpoint_path
        
        # Save complete training history
        try:
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'dataset': actual_dataset_name,
                    'history': training_history,
                    'final_metrics': final_metrics,
                    'hyperparameters': {k: v for k, v in hyperparams.items() 
                                      if not callable(v)}
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}")
        
        # Log final results
        logger.info(f"Training completed for {actual_dataset_name}")
        logger.info(f"Training time: {str(datetime.now() - start_time)}")
        logger.info(f"Best AUC: {best_auc:.4f}")
        logger.info(f"Final AUC: {final_auc:.4f}")
        logger.info(f"Total epochs: {epoch + 1}")
        logger.info(f"Early stopped: {patience_counter >= hyperparams['patience']}")
        logger.info(f"Checkpoints saved in {checkpoint_dir}")
        
        return final_auc, checkpoint_paths['final']
        
    except Exception as e:
        logger.error(f"Unexpected error in transfer learning: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise


def run_downstream_evaluation(
    pretrain_datasets: List[Dict],
    holdout_data: Dict,
    all_genes: List[str],
    args
) -> Dict:
    """Run downstream evaluation pipeline."""
    if not args.pretrained_checkpoint:
        raise ValueError("Pretrained checkpoint path is required for evaluate mode")

    logger.info("Running downstream evaluation...")

    # Load pretrained checkpoint
    with open(args.pretrained_checkpoint, "rb") as f:
        checkpoint = pickle.load(f)

    pretrained_params = checkpoint["state"]["params"]

    # Load or create model configuration
    pretrain_dir = os.path.dirname(os.path.dirname(args.pretrained_checkpoint))
    config_path = os.path.join(pretrain_dir, 'model_config.json')

    if os.path.exists(config_path):
        logger.info(f"Loading model configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = IC2BertConfig(**config_dict)
    else:
        logger.info("Creating new model configuration")
        model_config = IC2BertConfig(
            n_genes=len(all_genes),
            n_expressions_bins=args.n_expressions_bins,
            embed_dim=128,
            num_attention_heads=4,
            ffn_embed_dim=256,
            num_layers=4,
            use_gene_embedding=True,
            project_gene_embedding=True,
            init_gene_embed_dim=64,
            dropout_rate=0.1
        )

    # Create directory for transfer learning results
    transfer_results_dir = os.path.join(args.output_dir, 'transfer_learning_results')
    os.makedirs(transfer_results_dir, exist_ok=True)

    # Initialize results dictionary
    results = {
        'trial_info': {
            'trial_num': args.trial_num,
            'n_expressions_bins': args.n_expressions_bins,
            'random_seed': args.random_seed,
            'holdout_dataset': args.holdout_dataset
        },
        'pretraining_datasets': {},
        'holdout_dataset': {},
        'processed_datasets': [],
        'unprocessed_datasets': []
    }

    # Track processed datasets using a set for efficient lookup
    processed_datasets = set()
    target_datasets = set(DATASET_NAMES)

    # Initialize tokenizer with all data
    all_data = []
    for dataset in pretrain_datasets:
        all_data.append(dataset['train']['X'])
        all_data.append(dataset['test']['X'])
    all_data.extend([holdout_data['train']['X'], holdout_data['test']['X']])

    tokenizer = BinnedExpressionTokenizer(
        n_expressions_bins=args.n_expressions_bins,
        data=np.vstack(all_data)
    )

    # Create dataset mapping (excluding holdout)
    dataset_mapping = {}
    dataset_names = [name for name in DATASET_NAMES if name != args.holdout_dataset]
    for idx, dataset in enumerate(pretrain_datasets):
        if idx < len(dataset_names):
            dataset_mapping[dataset_names[idx]] = dataset

    # Create dataset mapping (excluding holdout)
    dataset_mapping = {}
    dataset_names = [name for name in DATASET_NAMES if name != args.holdout_dataset]
    for idx, dataset in enumerate(pretrain_datasets):
        if idx < len(dataset_names):
            dataset_mapping[dataset_names[idx]] = dataset

    # Add holdout dataset to mapping
    dataset_mapping[args.holdout_dataset] = holdout_data

    # Process all datasets
    for dataset_name in DATASET_NAMES:
        logger.info(f"Evaluating dataset: {dataset_name}")

        try:
            dataset = dataset_mapping[dataset_name]

            # Zero-shot evaluation
            zero_shot_metrics = zero_shot_evaluate(
                state=create_evaluation_state(pretrained_params, model_config),
                dataset=dataset,
                tokenizer=tokenizer
            )

            # Transfer learning evaluation
            transfer_auc, checkpoint_path = transfer_learn(
                pretrained_state=create_evaluation_state(pretrained_params, model_config),
                dataset=dataset,
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                num_epochs=100 if not args.test_mode else 2,
                batch_size=16 if not args.test_mode else 4,
                learning_rate=1e-5,
                experiment_dir=transfer_results_dir,
                use_ia3=False
            )

            # Store results in appropriate category
            result_entry = {
                'zero_shot_metrics': zero_shot_metrics,
                'transfer_learning_auc': float(transfer_auc),
                'checkpoint_path': checkpoint_path
            }

            if dataset_name == args.holdout_dataset:
                results['holdout_dataset'] = result_entry
            else:
                results['pretraining_datasets'][dataset_name] = result_entry

            processed_datasets.add(dataset_name)

            # Log results
            logger.info(f"Results for {dataset_name}:")
            logger.info(f"  Zero-shot AUROC: {zero_shot_metrics['auc']:.4f}")
            logger.info(f"  Zero-shot accuracy: {zero_shot_metrics['accuracy']:.4f}")
            logger.info(f"  Transfer learning AUROC: {transfer_auc:.4f}")

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            logger.error(traceback.format_exc())

    # Prepare final results
    final_results = {
        'trial_info': results['trial_info'],
        'pretraining_datasets': results['pretraining_datasets'],
        'holdout_dataset': results['holdout_dataset'],
        'processed_datasets': list(processed_datasets),
        'unprocessed_datasets': list(target_datasets - processed_datasets)
    }

    # Save results
    results_file = os.path.join(
        transfer_results_dir,
        f'downstream_results_trial{args.trial_num}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Log summary
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Successfully processed {len(processed_datasets)} datasets")
    if final_results['unprocessed_datasets']:
        logger.warning(f"Failed to process {len(final_results['unprocessed_datasets'])} datasets: {final_results['unprocessed_datasets']}")

    return final_results


# Learning rate schedules
def get_lr_schedule(schedule_type: str, hyperparams: Dict):
    """Get learning rate schedule based on configuration."""
    def base_schedule(step):
        warmup_factor = jnp.minimum(step / hyperparams['warmup_steps'], 1.0)
        decay_factor = hyperparams['decay_factor'](step)
        initial_scale = 10.0
        lr = hyperparams['learning_rate'] * initial_scale * warmup_factor * decay_factor
        return jnp.maximum(lr, 1e-6)

    def cyclic_schedule(step):
        cycle_length = 1000
        cycle_position = step % cycle_length
        cycle_factor = 0.5 * (1 + jnp.cos(jnp.pi * cycle_position / cycle_length))
        base_lr = base_schedule(step)
        return base_lr * (0.1 + 0.9 * cycle_factor)

    def riaz_schedule(step):
        """Conservative learning rate schedule for Riaz dataset."""
        warmup_steps = hyperparams['warmup_steps']
        max_lr = hyperparams['max_learning_rate']
        min_lr = hyperparams['min_learning_rate']
        warmup_ratio = hyperparams['warmup_ratio']
        
        if step < warmup_steps:
            warmup_progress = step / warmup_steps
            return min_lr + (max_lr * warmup_ratio - min_lr) * warmup_progress
        
        decay_rate = 0.95
        decay_steps = 100
        current_step = step - warmup_steps
        decay_factor = decay_rate ** (current_step / decay_steps)
        
        lr = max_lr * warmup_ratio * decay_factor
        return jnp.clip(lr, min_lr, max_lr)

    schedules = {
        'base': base_schedule,
        'cyclic': cyclic_schedule,
        'riaz': riaz_schedule
    }
    
    return schedules.get(schedule_type, base_schedule)


class TrainingMetrics:
    """Helper class for tracking and logging training metrics."""
    def __init__(self, args):
        self.metrics = {
            'trial': args.trial_num,
            'n_bins': args.n_expressions_bins,
            'split_seed': args.random_seed,
            'reconstruction_accuracies': [],
            'val_accuracies': [],
            'epochs': [],
            'best_epoch': None,
            'best_reconstruction_accuracy': None,
            'best_val_accuracy': None
        }

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                if isinstance(self.metrics[key], list):
                    self.metrics[key].append(float(value))
                else:
                    self.metrics[key] = float(value)

    def save(self, output_dir: str):
        """Save metrics to file."""
        metrics_path = os.path.join(output_dir, 'pretrain_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def __getitem__(self, key):
        return self.metrics[key]

