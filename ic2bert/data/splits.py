"""
Dataset splitting functionality for IC2Bert.

This module handles the creation, saving, loading, and validation of dataset splits
for the Leave-One-Dataset-Out Cross-Validation (LODOCV) procedure.
"""

import os
import json
import logging
from typing import Dict, Optional, NamedTuple
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import jax

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class DatasetSplit(NamedTuple):
    """Store split information for a single dataset.
    
    Attributes:
        name: Name of the dataset
        pretrain_indices: Indices for pretraining set
        test_indices: Indices for test set
        labels: Full label array for the dataset
    """
    name: str
    pretrain_indices: np.ndarray
    test_indices: np.ndarray
    labels: np.ndarray

def create_dataset_splits(config: Dict, splits_dir: str, random_seed: int, pretrain_ratio: float = 0.8) -> Dict[str, DatasetSplit]:
    """Create dataset splits with trial-specific random seed.
    
    Args:
        config: Configuration dictionary
        splits_dir: Directory to save splits
        random_seed: Random seed for reproducibility
        pretrain_ratio: Ratio of data to use for pretraining
        
    Returns:
        Dictionary mapping dataset names to DatasetSplit objects
    """
    os.makedirs(splits_dir, exist_ok=True)
    splits_file = os.path.join(splits_dir, 'dataset_splits.json')

    logger.info(f"Creating new dataset splits with random seed {random_seed}")
    splits = {}

    for dataset_name in DATASET_NAMES:
        logger.info(f"Processing dataset: {dataset_name}")
        dataset_path = os.path.join(config['data']['datasets_dir'], f"{dataset_name}")

        data = pd.read_csv(dataset_path)
        if 'ICB_Response' not in data.columns:
            raise ValueError(f"Dataset {dataset_name} missing ICB_Response column")

        labels = data['ICB_Response'].values
        indices = np.arange(len(data))

        # Create stratified split
        pretrain_idx, test_idx = train_test_split(
            indices,
            test_size=1-pretrain_ratio,
            stratify=labels,
            random_state=random_seed
        )

        # Verify split properties
        train_labels = labels[pretrain_idx]
        test_labels = labels[test_idx]

        if len(set(pretrain_idx).intersection(set(test_idx))) > 0:
            raise ValueError(f"Split overlap detected in dataset {dataset_name}")

        # Check stratification
        train_ratio = train_labels.mean()
        test_ratio = test_labels.mean()
        if abs(train_ratio - test_ratio) > 0.1:
            logger.warning(f"Large class imbalance detected in {dataset_name}")
            logger.warning(f"Train positive ratio: {train_ratio:.3f}")
            logger.warning(f"Test positive ratio: {test_ratio:.3f}")

        splits[dataset_name] = DatasetSplit(
            name=dataset_name,
            pretrain_indices=pretrain_idx,
            test_indices=test_idx,
            labels=labels
        )

        logger.info(f"  Total samples: {len(indices)}")
        logger.info(f"  Pretrain samples: {len(pretrain_idx)}")
        logger.info(f"  Test samples: {len(test_idx)}")
        logger.info(f"  Pretrain positive ratio: {train_ratio:.3f}")
        logger.info(f"  Test positive ratio: {test_ratio:.3f}")

    save_splits(splits, splits_file, random_seed)
    return splits

def save_splits(splits: Dict[str, DatasetSplit], filepath: str, random_seed: int) -> None:
    """Save splits information with trial seed information.
    
    Args:
        splits: Dictionary of dataset splits
        filepath: Path to save JSON file
        random_seed: Random seed used for splitting
    """
    splits_dir = os.path.dirname(filepath)
    os.makedirs(splits_dir, exist_ok=True)

    splits_info = {
        'random_seed': random_seed,
        'creation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'datasets': {
            name: {
                'n_total_samples': len(split.pretrain_indices) + len(split.test_indices),
                'n_pretrain_samples': len(split.pretrain_indices),
                'n_test_samples': len(split.test_indices),
                'pretrain_indices': split.pretrain_indices.tolist(),
                'test_indices': split.test_indices.tolist(),
                'pretrain_positive_ratio': float(split.labels[split.pretrain_indices].mean()),
                'test_positive_ratio': float(split.labels[split.test_indices].mean())
            }
            for name, split in splits.items()
        }
    }

    with open(filepath, 'w') as f:
        json.dump(splits_info, f, indent=2)

    # Save human-readable summary
    _save_splits_summary(splits_info, splits_dir, random_seed)
    
    logger.info(f"Saved splits information to {splits_dir} (random seed: {random_seed})")

def _save_splits_summary(splits_info: Dict, splits_dir: str, random_seed: int) -> None:
    """Save human-readable summary of splits.
    
    Args:
        splits_info: Dictionary containing splits information
        splits_dir: Directory to save summary
        random_seed: Random seed used for splitting
    """
    summary_path = os.path.join(splits_dir, 'splits_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Dataset Split Summary (Random Seed: {random_seed})\n")
        f.write("==========================================\n\n")
        for name, info in splits_info['datasets'].items():
            f.write(f"Dataset: {name}\n")
            f.write(f"  Total samples: {info['n_total_samples']}\n")
            f.write(f"  Pretrain samples: {info['n_pretrain_samples']}\n")
            f.write(f"  Test samples: {info['n_test_samples']}\n")
            f.write(f"  Pretrain positive ratio: {info['pretrain_positive_ratio']:.3f}\n")
            f.write(f"  Test positive ratio: {info['test_positive_ratio']:.3f}\n")
            f.write("\n")

def load_splits(filepath: str) -> Dict[str, DatasetSplit]:
    """Load and validate splits from JSON file.
    
    Args:
        filepath: Path to the JSON splits file
    
    Returns:
        Dictionary mapping dataset names to DatasetSplit objects
    """
    logger.info(f"Loading splits from {filepath}")

    with open(filepath, 'r') as f:
        splits_info = json.load(f)

    splits = {}
    datasets_info = splits_info.get('datasets', {})

    for name, info in datasets_info.items():
        pretrain_indices = np.array(info['pretrain_indices'])
        test_indices = np.array(info['test_indices'])

        if len(set(pretrain_indices).intersection(set(test_indices))) > 0:
            raise ValueError(f"Split overlap detected in loaded splits for {name}")

        splits[name] = DatasetSplit(
            name=name,
            pretrain_indices=pretrain_indices,
            test_indices=test_indices,
            labels=np.array([])  # Empty array since we don't need full labels when loading
        )

        logger.info(f"Loaded splits for {name}:")
        logger.info(f"  Pretrain samples: {len(pretrain_indices)}")
        logger.info(f"  Test samples: {len(test_indices)}")

    return splits

def verify_splits_usage(config: Dict, splits: Dict[str, DatasetSplit]) -> None:
    """Verify that splits are being used correctly.
    
    Args:
        config: Configuration dictionary
        splits: Dictionary of dataset splits to verify
        
    Raises:
        ValueError: If any verification fails
    """
    logger.info("Verifying splits usage...")

    for dataset_name in DATASET_NAMES:
        if dataset_name not in splits:
            raise ValueError(f"Missing splits for dataset {dataset_name}")

        dataset_path = os.path.join(config['data']['datasets_dir'], f"{dataset_name}")
        data = pd.read_csv(dataset_path)
        split = splits[dataset_name]

        if np.max(split.pretrain_indices) >= len(data) or np.max(split.test_indices) >= len(data):
            raise ValueError(f"Split indices out of bounds for dataset {dataset_name}")

        if len(set(split.pretrain_indices).intersection(set(split.test_indices))) > 0:
            raise ValueError(f"Split overlap detected in dataset {dataset_name}")

        all_indices = set(split.pretrain_indices).union(set(split.test_indices))
        if len(all_indices) != len(data):
            raise ValueError(f"Not all samples are used in splits for dataset {dataset_name}")

        logger.info(f"Splits verified for {dataset_name}")

def split_pretrain_data(pretrain_data, val_ratio: float = 0.1) -> Dict:
    """Split pretraining data into train and validation sets.
    
    Args:
        pretrain_data: Dictionary containing pretraining data
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Dictionary containing train and validation data
    """
    total_samples = pretrain_data['tokens'].shape[0]
    val_size = int(total_samples * val_ratio)

    rng = jax.random.PRNGKey(0)
    permutation = jax.random.permutation(rng, total_samples)
    shuffled_tokens = pretrain_data['tokens'][permutation]

    return {
        'train': {'tokens': shuffled_tokens[val_size:]},
        'val': {'tokens': shuffled_tokens[:val_size]}
    }