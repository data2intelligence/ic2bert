"""
Utility functions for IC2Bert.

This module provides various utility functions including:
1. Dataset and focal loss parameter configuration
2. Logging setup
3. Checkpoint management
4. Argument validation
"""

import os
import json
import pickle
import logging
import shutil
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import argparse
from dataclasses import dataclass

# List of dataset names
DATASET_NAMES = [
    "CCRCC_ICB_Miao2018", "mRCC_Atezo+Bev_McDermott2018",
    "Melanoma_Ipilimumab_VanAllen2015", "mRCC_Atezolizumab_McDermott2018",
    "Melanoma_Nivolumab_Riaz2017", "NSCLC_ICB_Ravi2023",
    "Melanoma_PD1_Hugo2016", "PanCancer_Pembrolizumab_Yang2021",
    "Hepatocellular_Atezo+Bev_Finn2020", "Melanoma_PD1_Liu2019",
    "Hepatocellular_Atezolizumab_Finn2020", "mGC_Pembrolizumab_Kim2018",
    "Urothelial_Atezolizumab_Mariathasan2018"
]

# Logger setup
logger = logging.getLogger(__name__)

def setup_logging(log_file: str) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to log file
    
    Raises:
        Exception: If logging setup fails
    """
    try:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Reset existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Configure handlers
        logger.setLevel(logging.INFO)
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
        
        for handler in handlers:
            handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(handler)

        logger.info(f"Logging to: {log_file}")

    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        print(f"Attempted log file path: {log_file}")
        raise

def get_focal_loss_params(dataset_name: str, train_pos_ratio: float) -> Dict:
    """
    Get dataset-specific focal loss parameters.

    Args:
        dataset_name: Name of the dataset
        train_pos_ratio: Positive class ratio in training data
    """
    FOCAL_PARAMETERS = {
        # Small datasets (N < 50) - More aggressive focal loss due to limited data
        "CCRCC_ICB_Miao2018": {  # N=33, kidney small
            "use_focal_loss": True,
            "focal_gamma": 2.5,    # Higher gamma for more focus on hard examples
            "focal_alpha": None,   # Will use train_pos_ratio
            "class_balance_threshold": 0.3  # Threshold for considering class imbalance
        },

        "Melanoma_PD1_Hugo2016": {  # N=26, melanoma small
            "use_focal_loss": True,
            "focal_gamma": 2.5,    # Higher gamma for smallest dataset
            "focal_alpha": None,   # Will use train_pos_ratio
            "class_balance_threshold": 0.3
        },

        "Melanoma_Ipilimumab_VanAllen2015": {
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "focal_alpha": None,
            "class_balance_threshold": 0.25
        },

        "Hepatocellular_Atezolizumab_Finn2020": {  # N=43, liver small
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "focal_alpha": None,
            "class_balance_threshold": 0.25
        },

        "mGC_Pembrolizumab_Kim2018": {  # N=45, gastric small
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "focal_alpha": None,
            "class_balance_threshold": 0.25
        },

        # Medium datasets (50 <= N < 100) - Moderate focal loss settings
        "Melanoma_Nivolumab_Riaz2017": {  # N=51, melanoma medium
            "use_focal_loss": True,
            "focal_gamma": 1.5,    # Lower gamma for medium dataset
            "focal_alpha": None,
            "class_balance_threshold": 0.2
        },

        "PanCancer_Pembrolizumab_Yang2021": {  # N=64, pancancer medium
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "focal_alpha": None,
            "class_balance_threshold": 0.2
        },

        "mRCC_Atezolizumab_McDermott2018": {  # N=74, kidney medium
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "focal_alpha": None,
            "class_balance_threshold": 0.2
        },

        "mRCC_Atezo+Bev_McDermott2018": {  # N=82, kidney medium
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "focal_alpha": None,
            "class_balance_threshold": 0.2
        },

        "NSCLC_ICB_Ravi2023": {  # N=90, lung medium
            "use_focal_loss": True,
            "focal_gamma": 1.5,
            "focal_alpha": None,
            "class_balance_threshold": 0.2
        },

        # Large datasets (N >= 100) - Use focal loss only if significant imbalance
        "Melanoma_PD1_Liu2019": {  # N=121, melanoma large
            "use_focal_loss": "auto",  # Only if imbalanced
            "focal_gamma": 1.0,    # Lower gamma for large dataset
            "focal_alpha": None,
            "class_balance_threshold": 0.15
        },

        "Hepatocellular_Atezo+Bev_Finn2020": {  # N=245, liver large
            "use_focal_loss": "auto",
            "focal_gamma": 1.0,
            "focal_alpha": None,
            "class_balance_threshold": 0.15
        },

        "Urothelial_Atezolizumab_Mariathasan2018": {  # N=298, bladder large
            "use_focal_loss": "auto",
            "focal_gamma": 1.0,
            "focal_alpha": None,
            "class_balance_threshold": 0.15
        }
    }

    def configure_focal_loss(dataset_params: Dict, pos_ratio: float) -> Dict:
        """Configure focal loss parameters based on dataset characteristics and class ratio."""
        imbalance = abs(pos_ratio - 0.5)
        threshold = dataset_params["class_balance_threshold"]

        params = {
            "use_focal_loss": (
                dataset_params["use_focal_loss"] if dataset_params["use_focal_loss"] != "auto"
                else imbalance > threshold
            ),
            "focal_gamma": dataset_params["focal_gamma"],
            "focal_alpha": pos_ratio  # Use actual class ratio
        }

        if imbalance > threshold:
            # Adjust gamma based on imbalance severity
            gamma_adjustment = min(imbalance * 2, 1.0)  # Cap adjustment at 1.0
            params["focal_gamma"] = min(params["focal_gamma"] + gamma_adjustment, 3.0)

        return params

    dataset_params = FOCAL_PARAMETERS.get(dataset_name)
    if dataset_params is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return configure_focal_loss(dataset_params, train_pos_ratio)

def get_dataset_characteristics(dataset_name: str, train_pos_ratio: float, dataset_size: int) -> Dict:
    """
    Get dataset characteristics including properly configured focal loss parameters.

    Args:
        dataset_name: Name of the dataset
        train_pos_ratio: Positive class ratio in training data
        dataset_size: Total size of the dataset

    Returns:
        Tuple containing (dataset characteristics dict, closest matching dataset name)
    """
    # Base dataset characteristics
    DATASET_CHARACTERISTICS = {
        "CCRCC_ICB_Miao2018": {
            "expected_size": 33,
            "signature": "kidney_small",
            "cancer_type": "kidney",
            "special_params": {
                "batch_size": 4,
                "learning_rate_factor": 0.3,
                "dropout_rate": 0.25,
                "weight_decay": 0.02,
                "use_focal_loss": True,
                "focal_gamma": 2.0,
                "focal_alpha": train_pos_ratio,  # Use actual class ratio
                "grad_accumulation_steps": 8,
                "min_epochs": 50,
                "patience": 25,
                "use_mixup": True,
                "mixup_alpha": 0.1,
                "use_swa": True,
                "swa_start": 0.75,
                "warmup_steps": 100
            }
        },
        "mRCC_Atezo+Bev_McDermott2018": {
            "expected_size": 82,
            "signature": "kidney_medium",
            "cancer_type": "kidney",
            "special_params": {
                "batch_size": 16,
                "learning_rate": 1e-5,
                "learning_rate_factor": 1.0,
                "dropout_rate": 0.15,
                "weight_decay": 0.01,
                "grad_accumulation_steps": 4,
                "use_mixup": True,
                "mixup_alpha": 0.2,
                "use_swa": True,
                "swa_start": 0.8
            }
        },
        "Melanoma_Ipilimumab_VanAllen2015": {
            "expected_size": 42,
            "signature": "melanoma_small",
            "cancer_type": "melanoma",
            "special_params": {
                "batch_size": 8,
                "learning_rate_factor": 0.5,
                "dropout_rate": 0.2,
                "weight_decay": 0.015,
                "use_augmentation": True,
                "aug_strength": 0.1,
                "grad_accumulation_steps": 6,
                "use_focal_loss": True,
                "focal_gamma": 1.5,
                "min_epochs": 45
            }
        },
        "mRCC_Atezolizumab_McDermott2018": {
            "expected_size": 74,
            "signature": "kidney_medium",
            "cancer_type": "kidney",
            "special_params": {
                "batch_size": 16,
                "learning_rate_factor": 1.0,
                "dropout_rate": 0.15,
                "weight_decay": 0.01,
                "use_swa": True,
                "use_mixup": True,
                "mixup_alpha": 0.15,
                "grad_accumulation_steps": 4
            }
        },
        "Melanoma_Nivolumab_Riaz2017": {
            "expected_size": 51,
            "signature": "melanoma_medium",
            "cancer_type": "melanoma",
            "special_params": {
                "batch_size": 4,
                "learning_rate_factor": 0.25,
                "dropout_rate": 0.3,
                "weight_decay": 0.025,
                "use_focal_loss": True,
                "focal_gamma": 2.0,
                "grad_accumulation_steps": 8,
                "use_mixup": True,
                "mixup_alpha": 0.1,
                "min_epochs": 50,
                "patience": 25
            }
        },
        "NSCLC_ICB_Ravi2023": {
            "expected_size": 90,
            "signature": "lung_medium",
            "cancer_type": "lung",
            "special_params": {
                "batch_size": 16,
                "learning_rate_factor": 1.2,
                "dropout_rate": 0.15,
                "weight_decay": 0.01,
                "use_swa": True,
                "use_gradient_centralization": True,
                "grad_accumulation_steps": 4
            }
        },
        "Melanoma_PD1_Hugo2016": {
            "expected_size": 26,
            "signature": "melanoma_small",
            "cancer_type": "melanoma",
            "special_params": {
                "batch_size": 4,
                "learning_rate_factor": 0.25,
                "dropout_rate": 0.3,
                "weight_decay": 0.025,
                "use_focal_loss": True,
                "focal_gamma": 2.0,
                "grad_accumulation_steps": 8,
                "use_mixup": True,
                "mixup_alpha": 0.1,
                "min_epochs": 50,
                "patience": 25
            }
        },
        "PanCancer_Pembrolizumab_Yang2021": {
            "expected_size": 64,
            "signature": "pancancer_medium",
            "cancer_type": "pancancer",
            "special_params": {
                "batch_size": 16,
                "learning_rate_factor": 0.8,
                "dropout_rate": 0.2,
                "weight_decay": 0.015,
                "use_domain_adaptation": True,
                "use_gradient_centralization": True,
                "grad_accumulation_steps": 4,
                "use_swa": True
            }
        },
        "Hepatocellular_Atezo+Bev_Finn2020": {
            "expected_size": 245,
            "signature": "liver_large",
            "cancer_type": "liver",
            "special_params": {
                "batch_size": 32,
                "learning_rate_factor": 2.0,
                "dropout_rate": 0.1,
                "weight_decay": 0.008,
                "use_gradient_centralization": True,
                "use_lookahead": True,
                "grad_accumulation_steps": 2,
                "warmup_steps": 300
            }
        },
        "Melanoma_PD1_Liu2019": {
            "expected_size": 121,
            "signature": "melanoma_large",
            "cancer_type": "melanoma",
            "special_params": {
                "batch_size": 32,
                "learning_rate_factor": 1.5,
                "dropout_rate": 0.1,
                "weight_decay": 0.008,
                "use_gradient_centralization": True,
                "grad_accumulation_steps": 2,
                "use_lookahead": True
            }
        },
        "Hepatocellular_Atezolizumab_Finn2020": {
            "expected_size": 43,
            "signature": "liver_small",
            "cancer_type": "liver",
            "special_params": {
                "batch_size": 8,
                "learning_rate_factor": 0.5,
                "dropout_rate": 0.2,
                "weight_decay": 0.015,
                "use_focal_loss": True,
                "focal_gamma": 1.5,
                "grad_accumulation_steps": 6,
                "use_mixup": True,
                "mixup_alpha": 0.15,
                "min_epochs": 45
            }
        },
        "mGC_Pembrolizumab_Kim2018": {
            "expected_size": 45,
            "signature": "gastric_small",
            "cancer_type": "gastric",
            "special_params": {
                "batch_size": 8,
                "learning_rate_factor": 0.5,
                "dropout_rate": 0.2,
                "weight_decay": 0.015,
                "use_focal_loss": True,
                "focal_gamma": 1.5,
                "grad_accumulation_steps": 6,
                "use_mixup": True,
                "mixup_alpha": 0.15,
                "min_epochs": 45,
                "patience": 20
            }
        },
        "Urothelial_Atezolizumab_Mariathasan2018": {
            "expected_size": 298,
            "signature": "bladder_large",
            "cancer_type": "bladder",
            "special_params": {
                "batch_size": 32,
                "learning_rate_factor": 2.0,
                "dropout_rate": 0.1,
                "weight_decay": 0.008,
                "use_gradient_centralization": True,
                "use_lookahead": True,
                "grad_accumulation_steps": 2,
                "warmup_steps": 300,
                "use_swa": False  # Large dataset, standard optimization is sufficient
            }
        }
    }

    # Handle dataset size mismatch
    closest_match = None
    min_diff = float('inf')

    for name, chars in DATASET_CHARACTERISTICS.items():
        diff = abs(chars["expected_size"] - dataset_size)
        if diff < min_diff:
            min_diff = diff
            closest_match = name

    # Get characteristics for matched dataset
    characteristics = DATASET_CHARACTERISTICS.get(closest_match, {
        "expected_size": dataset_size,
        "signature": "unknown",
        "cancer_type": "unknown",
        "special_params": {}
    }).copy()

    special_params = characteristics["special_params"]

    # Calculate class imbalance
    imbalance_ratio = max(train_pos_ratio, 1 - train_pos_ratio) / min(train_pos_ratio, 1 - train_pos_ratio)
    severe_imbalance = imbalance_ratio > 3

    # Configure focal loss parameters based on dataset characteristics and class imbalance
    if "small" in characteristics["signature"] or severe_imbalance:
        focal_params = {
            "use_focal_loss": True,
            "focal_gamma": special_params.get("focal_gamma", 2.0),  # Use existing gamma if specified
            "focal_alpha": train_pos_ratio,  # Use actual class ratio
            "class_weights": True,
            "pos_weight": (1 - train_pos_ratio) / (train_pos_ratio + 1e-8)  # Avoid division by zero
        }
        special_params.update(focal_params)

    # Size-specific optimizations
    size_specific_params = {
        "small": {
            "use_augmentation": True,
            "aug_strength": 0.1,
            "use_swa": True,
            "swa_start": 0.75,
            "grad_accumulation_steps": special_params.get("grad_accumulation_steps", 8)
        },
        "medium": {
            "use_swa": True,
            "grad_accumulation_steps": special_params.get("grad_accumulation_steps", 4)
        },
        "large": {
            "use_gradient_centralization": True,
            "use_lookahead": True,
            "grad_accumulation_steps": special_params.get("grad_accumulation_steps", 2)
        }
    }

    # Add size-specific parameters
    if "small" in characteristics["signature"]:
        special_params.update(size_specific_params["small"])
    elif "large" in characteristics["signature"]:
        special_params.update(size_specific_params["large"])
    else:  # medium
        special_params.update(size_specific_params["medium"])

    # Cancer-type specific optimizations
    cancer_specific_params = {
        "pancancer": {
            "use_domain_adaptation": True,
            "domain_lambda": 0.1,
            "use_gradient_centralization": True
        },
        "melanoma": {
            "use_augmentation": special_params.get("use_augmentation", True),
            "aug_strength": special_params.get("aug_strength", 0.1)
        },
        "kidney": {
            "use_mixup": True,
            "mixup_alpha": special_params.get("mixup_alpha", 0.2)
        }
    }

    # Add cancer-type specific parameters
    cancer_type = characteristics["cancer_type"].lower()
    if cancer_type in cancer_specific_params:
        special_params.update(cancer_specific_params[cancer_type])

    # Add validation configuration
    validation_config = {
        "validation_frequency": 1 if "small" in characteristics["signature"] else 5,
        "full_validation_frequency": 10,
        "save_predictions": True,
        "compute_metrics": ["auc", "accuracy", "precision", "recall", "f1",
                          "confusion_matrix", "pr_curve"]
    }
    characteristics["validation_config"] = validation_config

    # Add metrics tracking configuration
    characteristics["metrics_config"] = {
        "track_class_distribution": True,
        "track_gradient_stats": True,
        "track_loss_components": True,
        "track_batch_statistics": "small" in characteristics["signature"]
    }

    # Log configuration details
    logger.info(f"\nDataset characteristics for {dataset_name}:")
    logger.info(f"Size category: {characteristics['signature']}")
    logger.info(f"Cancer type: {characteristics['cancer_type']}")
    logger.info(f"Class distribution - Positive ratio: {train_pos_ratio:.3f}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    logger.info("\nSpecial parameters:")
    for param, value in special_params.items():
        logger.info(f"  {param}: {value}")


    # Get focal loss parameters
    focal_params = get_focal_loss_params(dataset_name, train_pos_ratio)

    # Update special parameters with focal loss configuration
    if focal_params["use_focal_loss"]:
        special_params.update(focal_params)

        # Log focal loss configuration
        logger.info(f"\nFocal Loss Configuration for {dataset_name}:")
        logger.info(f"  Gamma: {focal_params['focal_gamma']:.2f}")
        logger.info(f"  Alpha (class balance): {focal_params['focal_alpha']:.3f}")
        logger.info(f"  Class distribution - Positive ratio: {train_pos_ratio:.3f}")

    return characteristics, closest_match


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state, epoch: int, args: argparse.Namespace) -> str:
        """Save checkpoint for given epoch."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pkl"
        )
        
        serializable_state = {
            "step": state.step,
            "params": state.params,
            "opt_state": state.opt_state,
            "rng": state.rng,
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "state": serializable_state,
                "epoch": epoch,
                "args": vars(args)
            }, f)
        
        logger.info(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
        return checkpoint_path
    
    def save_best_checkpoint(
        self,
        state,
        epoch: int,
        best_val_accuracy: float,
        args: argparse.Namespace
    ) -> None:
        """Save best checkpoint with metrics."""
        checkpoint_path = self.save_checkpoint(state, epoch, args)
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pkl")
        
        try:
            shutil.copy2(checkpoint_path, best_checkpoint_path)
            logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
            
            # Save metrics
            metrics_path = os.path.join(self.checkpoint_dir, "best_performance.json")
            metrics = {
                "epoch": epoch,
                "val_accuracy": float(best_val_accuracy),
                "checkpoint_path": best_checkpoint_path,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "training_args": vars(args)
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved best performance metrics (accuracy: {best_val_accuracy:.4f})")
            
        except Exception as e:
            logger.error(f"Error saving best checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str, model, learning_rate: float, input_shape: Tuple):
        """Load checkpoint from path."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        serializable_state = checkpoint["state"]
        epoch = checkpoint["epoch"]
        
        # Recreate TrainState
        state = TrainState(
            step=serializable_state["step"],
            params=serializable_state["params"],
            opt_state=serializable_state["opt_state"],
            apply_fn=model.apply,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate)
            ),
            rng=serializable_state["rng"],
        )
        
        return state, epoch

def save_checkpoint(state, epoch, checkpoint_dir, args):
    """Save checkpoint for the given epoch."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
    
    serializable_state = {
        "step": state.step,
        "params": state.params,
        "opt_state": state.opt_state,
        "rng": state.rng,
    }
    
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the checkpoint
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "state": serializable_state, 
            "epoch": epoch,
            "args": vars(args)  # Save training arguments for reproducibility
        }, f)
    
    logger.info(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
    return checkpoint_path

def save_best_checkpoint(state, epoch, checkpoint_dir, best_val_accuracy, args):
    """Save best checkpoint with performance metrics."""
    # First save the checkpoint itself
    checkpoint_path = save_checkpoint(state, epoch, checkpoint_dir, args)
    
    # Create a specific "best" checkpoint by copying the epoch checkpoint
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pkl")
    try:
        import shutil
        shutil.copy2(checkpoint_path, best_checkpoint_path)
        logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving best checkpoint: {str(e)}")
    
    # Save performance metrics
    metrics_path = os.path.join(checkpoint_dir, "best_performance.json")
    metrics = {
        "epoch": epoch,
        "val_accuracy": float(best_val_accuracy),
        "checkpoint_path": best_checkpoint_path,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_args": vars(args)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved best performance metrics (accuracy: {best_val_accuracy:.4f})")

def load_checkpoint(checkpoint_path, model, learning_rate, input_shape):
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    serializable_state = checkpoint["state"]
    epoch = checkpoint["epoch"]

    # Recreate the TrainState with the loaded data
    state = TrainState(
        step=serializable_state["step"],
        params=serializable_state["params"],
        opt_state=serializable_state["opt_state"],
        apply_fn=model.apply,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        ),
        rng=serializable_state["rng"],
    )

    return state, epoch

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments.
    
    Args:
        args: Command line arguments
        
    Raises:
        ValueError: If any validation fails
    """
    # Check required numeric values
    if args.trial_num <= 0:
        raise ValueError(f"Trial number must be positive, got {args.trial_num}")
    if args.n_expressions_bins <= 0:
        raise ValueError(f"Number of expression bins must be positive, got {args.n_expressions_bins}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)
    
    # Mode-specific validation
    if args.mode == 'pretrain':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    elif args.mode == 'evaluate':
        if not os.path.exists(args.pretrained_checkpoint):
            raise ValueError(f"Checkpoint not found at {args.pretrained_checkpoint}")
    
    # Validate training parameters
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.learning_rate}")
    if args.min_epochs <= 0:
        raise ValueError(f"Minimum epochs must be positive, got {args.min_epochs}")
    if args.patience < 0:
        raise ValueError(f"Patience must be non-negative, got {args.patience}")

