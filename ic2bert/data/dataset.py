"""
Dataset loading and processing functionality for IC2Bert.

This module handles loading, processing, and management of gene expression datasets
for immunotherapy response prediction.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple, Set
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .splits import DatasetSplit

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

class DatasetInfo(NamedTuple):
    """Information about a dataset."""
    name: str
    size: int
    n_genes: int
    train_size: int
    test_size: int
    pos_ratio: float
    cancer_type: str

class ProcessedDataset(NamedTuple):
    """Processed dataset with train/test splits."""
    train: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    info: DatasetInfo

def get_dataset_path(config: Dict, dataset_name: str) -> str:
    """Get full path to dataset file."""
    return os.path.join(config['data']['datasets_dir'], f"{dataset_name}")

def get_dataset_size(dataset_name: str) -> str:
    """Determine dataset size category."""
    DATASET_SIZES = {
        "CCRCC_ICB_Miao2018": "small",          # N=33
        "mRCC_Atezo+Bev_McDermott2018": "medium", # N=82
        "Melanoma_Ipilimumab_VanAllen2015": "small", # N=42
        "mRCC_Atezolizumab_McDermott2018": "medium", # N=74
        "Melanoma_Nivolumab_Riaz2017": "medium",  # N=51
        "NSCLC_ICB_Ravi2023": "medium",          # N=90
        "Melanoma_PD1_Hugo2016": "small",        # N=26
        "PanCancer_Pembrolizumab_Yang2021": "medium", # N=64
        "Hepatocellular_Atezo+Bev_Finn2020": "large", # N=245
        "Melanoma_PD1_Liu2019": "large",         # N=121
        "Hepatocellular_Atezolizumab_Finn2020": "small", # N=43
        "mGC_Pembrolizumab_Kim2018": "small",    # N=45
        "Urothelial_Atezolizumab_Mariathasan2018": "large" # N=298
    }
    return DATASET_SIZES.get(dataset_name, "medium")

def load_and_process_dataset(config: Dict, dataset_name: str, split: DatasetSplit) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """
    Load and process a single dataset.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset to load
        split: DatasetSplit containing train/test indices
        
    Returns:
        Tuple of (processed_dataset, available_genes)
    
    Raises:
        RuntimeError: If dataset cannot be loaded
        ValueError: If dataset format is invalid
    """
    dataset_path = get_dataset_path(config, dataset_name)

    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {dataset_name}: {str(e)}")

    if 'ICB_Response' not in data.columns:
        raise ValueError(f"Dataset {dataset_name} missing ICB_Response column")

    with open(config['data']['gene_list_path'], 'r') as f:
        genes = [line.strip() for line in f if line.strip()]

    available_genes = [gene for gene in genes if gene in data.columns]
    if not available_genes:
        raise ValueError(f"No genes from the gene list found in dataset {dataset_name}")

    X = data[available_genes].values.astype(np.float32)
    y = data['ICB_Response'].values.astype(np.float32)

    if np.max(split.pretrain_indices) >= len(X) or np.max(split.test_indices) >= len(X):
        raise ValueError(f"Split indices out of bounds for dataset {dataset_name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    processed_dataset = {
        'train': {
            'X': X[split.pretrain_indices],
            'y': y[split.pretrain_indices]
        },
        'test': {
            'X': X[split.test_indices],
            'y': y[split.test_indices]
        }
    }

    logger.info(f"Processed dataset {dataset_name}:")
    logger.info(f"  Train: {len(split.pretrain_indices)} samples")
    logger.info(f"  Test: {len(split.test_indices)} samples")
    logger.info(f"  Number of genes: {len(available_genes)}")

    return processed_dataset, available_genes

def load_all_datasets(config: Dict, splits: Optional[Dict[str, DatasetSplit]] = None,
                     save_splits: bool = True) -> Tuple[List[Dict[str, Dict[str, np.ndarray]]], List[str], Dict[str, DatasetSplit]]:
    """
    Load all datasets with optional splitting.
    
    Args:
        config: Configuration dictionary
        splits: Optional pre-defined splits to use
        save_splits: Whether to save the splits if newly created
        
    Returns:
        Tuple of (datasets, gene_list, splits)
    """
    all_datasets = []
    all_genes = set()

    with open(config['data']['gene_list_path'], 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]

    if splits is None:
        splits = create_dataset_splits(config, config['data']['splits_dir'],
                                     random_seed=42, pretrain_ratio=0.8)

    for dataset_name in DATASET_NAMES:
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name not in splits:
            raise ValueError(f"Missing split information for dataset {dataset_name}")
        split = splits[dataset_name]

        processed_dataset, dataset_genes = load_and_process_dataset(
            config=config,
            dataset_name=dataset_name,
            split=split
        )

        all_datasets.append(processed_dataset)
        all_genes.update(dataset_genes)

        logger.info(f"  Train samples: {len(processed_dataset['train']['X'])}")
        logger.info(f"  Test samples: {len(processed_dataset['test']['X'])}")
        logger.info(f"  Number of genes: {len(dataset_genes)}")

    all_genes = sorted(list(all_genes))
    n_genes = len(all_genes)
    
    for dataset in all_datasets:
        if dataset['train']['X'].shape[1] != n_genes:
            raise ValueError(f"Inconsistent number of genes in dataset")

    logger.info(f"Loaded {len(all_datasets)} datasets with {n_genes} genes")
    return all_datasets, all_genes, splits

def load_datasets_with_holdout(config: Dict, splits: Dict[str, DatasetSplit],
                             holdout_dataset: str) -> Tuple[Dict, Dict, List[str]]:
    """
    Load datasets with holdout handling.
    
    Args:
        config: Configuration dictionary
        splits: Dataset splits dictionary
        holdout_dataset: Name of dataset to holdout
        
    Returns:
        Tuple of (pretrain_datasets, holdout_dataset_data, all_genes)
    """
    logger.info(f"Loading datasets with holdout dataset: {holdout_dataset}")

    pretrain_datasets = []
    all_genes = set()

    holdout_split = splits[holdout_dataset]
    holdout_data, holdout_genes = load_and_process_dataset(
        config=config,
        dataset_name=holdout_dataset,
        split=holdout_split
    )
    all_genes.update(holdout_genes)

    for dataset_name in DATASET_NAMES:
        if dataset_name == holdout_dataset:
            continue

        split = splits[dataset_name]
        processed_dataset, dataset_genes = load_and_process_dataset(
            config=config,
            dataset_name=dataset_name,
            split=split
        )

        pretrain_datasets.append(processed_dataset)
        all_genes.update(dataset_genes)

    logger.info(f"Loaded {len(pretrain_datasets)} datasets for pretraining")
    logger.info(f"Holdout dataset: {holdout_dataset}")
    logger.info(f"Total genes: {len(all_genes)}")

    return pretrain_datasets, holdout_data, sorted(list(all_genes))

