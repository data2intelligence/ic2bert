"""
Loss functions and evaluation metrics for IC2Bert.

This module provides implementations of:
1. Masked Language Model (MLM) masking and loss
2. Classification loss functions
3. Evaluation metrics for zero-shot and supervised settings
4. Confidence metrics calculation
"""

import logging
from typing import Dict, Tuple, List
import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# MLM Functions
def create_mlm_masks(
    rng: jnp.ndarray,
    tokens: jnp.ndarray,
    mask_prob: float = 0.15
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create masks for masked language modeling.
    
    Args:
        rng: Random number generator key
        tokens: Input tokens
        mask_prob: Probability of masking
        
    Returns:
        Tuple of (masked_tokens, mlm_labels)
    """
    batch_size, seq_len = tokens.shape

    # Create random mask
    rng, mask_rng = jax.random.split(rng)
    mask = jax.random.bernoulli(mask_rng, p=mask_prob, shape=tokens.shape)

    # Create random numbers for mask/random/keep decision
    rng, decision_rng = jax.random.split(rng)
    rand_nums = jax.random.uniform(decision_rng, shape=tokens.shape)

    # Mask token (assuming 0 is reserved for padding)
    mask_token_id = tokens.max() + 1

    # Create masked tokens
    masked_tokens = jnp.where(
        mask,
        jnp.where(
            rand_nums < 0.8,
            mask_token_id,  # 80% mask
            jnp.where(
                rand_nums < 0.9,
                tokens,  # 10% keep
                jax.random.randint(decision_rng, tokens.shape, 0, mask_token_id)  # 10% random
            )
        ),
        tokens
    )

    # Create MLM labels (only for masked positions)
    mlm_labels = jnp.where(mask, tokens, -100)  # -100 is ignored in loss computation

    return masked_tokens, mlm_labels

def mlm_loss_fn(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute MLM loss with numerical stability.
    
    Args:
        logits: Predicted logits
        labels: True labels
        
    Returns:
        Mean loss value
    """
    epsilon = 1e-8

    # Compute stable softmax
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    sum_exp_logits = jnp.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / (sum_exp_logits + epsilon)

    # Compute cross-entropy loss
    labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(labels_onehot * jnp.log(probs + epsilon), axis=-1)

    # Mask out padding tokens
    mask = (labels != -100).astype(jnp.float32)
    masked_loss = loss * mask

    # Compute mean loss
    total_tokens = jnp.sum(mask)
    mean_loss = jnp.sum(masked_loss) / (total_tokens + epsilon)

    return mean_loss

def loss_fn(params: Dict, state: 'TrainState', batch: Dict, model: 'IC2Bert') -> jnp.ndarray:
    """Compute classification loss for a batch.
    
    Args:
        params: Model parameters
        state: Training state
        batch: Batch of data
        model: Model instance
        
    Returns:
        Mean loss value
    """
    logits = model.apply(params, state.rng, batch['tokens'].astype(jnp.int32))['logits']
    labels = batch['labels']
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

def calculate_reconstruction_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> jnp.ndarray:
    """Calculate reconstruction accuracy for MLM.
    
    Args:
        logits: Predicted logits
        labels: True labels
        
    Returns:
        Reconstruction accuracy
    """
    predictions = jnp.argmax(logits, axis=-1)
    mask = (labels != -100)
    correct = jnp.sum(jnp.equal(predictions, labels) * mask)
    total = jnp.sum(mask)
    return jnp.where(total > 0, correct / total, 0.0)

def zero_shot_evaluate(
    state: 'TrainState',
    dataset: Dict,
    tokenizer: 'BinnedExpressionTokenizer'
) -> Dict:
    """Perform zero-shot evaluation.
    
    Args:
        state: Training state
        dataset: Dataset to evaluate
        tokenizer: Tokenizer instance
        
    Returns:
        Dictionary of evaluation metrics
    """
    X_test = dataset['test']['X']
    y_test = dataset['test']['y']
    tokens = tokenizer.tokenize(X_test)

    # Get predictions
    outputs = state.apply_fn(state.params, state.rng, tokens, is_training=False)
    logits = outputs['logits']
    probabilities = jax.nn.sigmoid(logits)
    predictions = probabilities > 0.5

    # Calculate metrics
    metrics = _calculate_basic_metrics(predictions, probabilities, y_test)
    
    # Add prediction details
    metrics['prediction_details'] = {
        'probabilities': probabilities.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': y_test.tolist()
    }

    # Add confidence metrics
    confidence_metrics = calculate_confidence_metrics(probabilities, predictions, y_test)
    metrics.update(confidence_metrics)

    _log_evaluation_metrics(metrics)
    return metrics

def _calculate_basic_metrics(
    predictions: jnp.ndarray,
    probabilities: jnp.ndarray,
    true_labels: jnp.ndarray
) -> Dict:
    """Calculate basic classification metrics.
    
    Args:
        predictions: Binary predictions
        probabilities: Prediction probabilities
        true_labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    tp = jnp.sum((predictions == 1) & (true_labels == 1))
    fp = jnp.sum((predictions == 1) & (true_labels == 0))
    fn = jnp.sum((predictions == 0) & (true_labels == 1))
    tn = jnp.sum((predictions == 0) & (true_labels == 0))

    return {
        'auc': float(roc_auc_score(true_labels, probabilities)),
        'accuracy': float(jnp.mean(predictions == true_labels)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        'positive_ratio': float(jnp.mean(true_labels)),
        'predicted_positive_ratio': float(jnp.mean(predictions)),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_samples': len(true_labels)
    }

def calculate_confidence_metrics(
    probabilities: jnp.ndarray,
    predictions: jnp.ndarray,
    true_labels: jnp.ndarray
) -> Dict:
    """Calculate confidence-based metrics.
    
    Args:
        probabilities: Prediction probabilities
        predictions: Binary predictions
        true_labels: True labels
        
    Returns:
        Dictionary of confidence metrics
    """
    metrics = {}
    
    # Average confidence for correct/incorrect predictions
    correct_mask = predictions == true_labels
    metrics['avg_confidence_correct'] = float(
        jnp.mean(jnp.abs(probabilities[correct_mask] - 0.5)) + 0.5
    ) if jnp.any(correct_mask) else 0.0
    
    incorrect_mask = ~correct_mask
    metrics['avg_confidence_incorrect'] = float(
        jnp.mean(jnp.abs(probabilities[incorrect_mask] - 0.5)) + 0.5
    ) if jnp.any(incorrect_mask) else 0.0

    # Confidence distribution
    metrics['confidence_distribution'] = _calculate_confidence_distribution(
        probabilities, predictions, true_labels
    )

    return metrics

def _calculate_confidence_distribution(
    probabilities: jnp.ndarray,
    predictions: jnp.ndarray,
    true_labels: jnp.ndarray
) -> Dict:
    """Calculate confidence distribution metrics.
    
    Args:
        probabilities: Prediction probabilities
        predictions: Binary predictions
        true_labels: True labels
        
    Returns:
        Dictionary of confidence distribution metrics
    """
    distribution = {}
    confidence_levels = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    for low, high in confidence_levels:
        mask = (probabilities >= low) & (probabilities < high)
        if jnp.any(mask):
            correct_ratio = float(jnp.mean(predictions[mask] == true_labels[mask]))
            count = int(jnp.sum(mask))
        else:
            correct_ratio = 0.0
            count = 0

        distribution[f'{low:.1f}-{high:.1f}'] = {
            'count': count,
            'accuracy': correct_ratio
        }

    return distribution

def _log_evaluation_metrics(metrics: Dict) -> None:
    """Log evaluation metrics."""
    logger.info("Zero-shot ICB Classification Results:")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  True Positives: {metrics['true_positives']}")
    logger.info(f"  False Positives: {metrics['false_positives']}")
    logger.info(f"  True Negatives: {metrics['true_negatives']}")
    logger.info(f"  False Negatives: {metrics['false_negatives']}")
    logger.info(f"  Total Samples: {metrics['total_samples']}")
    logger.info(f"  Actual Positive Ratio: {metrics['positive_ratio']:.4f}")
    logger.info(f"  Predicted Positive Ratio: {metrics['predicted_positive_ratio']:.4f}")

    logger.info("\nConfidence Metrics:")
    logger.info(f"  Average Confidence (Correct): {metrics['avg_confidence_correct']:.4f}")
    logger.info(f"  Average Confidence (Incorrect): {metrics['avg_confidence_incorrect']:.4f}")
    
    logger.info("\nConfidence Distribution:")
    for conf_range, stats in metrics['confidence_distribution'].items():
        logger.info(f"  {conf_range}: {stats['count']} samples, {stats['accuracy']:.4f} accuracy")