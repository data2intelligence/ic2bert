#!/usr/bin/env python3
"""
LODOCV Transfer Learning for IC2Bert.
Implements Leave-One-Dataset-Out Cross-Validation with dataset-specific parameters.
"""

import os
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, NamedTuple

import jax
import numpy as np

from .config.model_config import IC2BertConfig
from .data.dataset import load_datasets_with_holdout
from .data.tokenizer import BinnedExpressionTokenizer
from .data.splits import load_splits, create_dataset_splits, verify_splits_usage
from .utils.utils import setup_logging
from .training.train import run_pretraining, run_downstream_evaluation
from .cli.args import parse_arguments

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Application configuration."""
    datasets_dir: str = './data/'
    gene_list_path: str = './intersect_prior.csv'

class ExperimentData(NamedTuple):
    """Container for experiment data."""
    pretrain_datasets: Dict
    holdout_data: Optional[Dict]
    all_genes: list
    splits: Dict

class ExperimentRunner:
    """Handles experiment execution and resource management."""
    
    def __init__(self, args):
        """Initialize experiment runner.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.config = Config()
        self.start_time = datetime.now()
        
    def setup(self) -> None:
        """Setup experiment environment."""
        self._initialize_runtime()
        self._setup_logging()
        self._create_directories()
        self._set_random_seeds()
    
    def _initialize_runtime(self) -> None:
        """Initialize runtime environment."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
        try:
            jax.devices('gpu')
            print("Using GPU")
        except RuntimeError:
            print("GPU not found, using CPU")
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = os.path.join(
            self.args.output_dir,
            'logs',
            f'{self.args.mode}_trial_{self.args.trial_num}.log'
        )
        setup_logging(log_file)
        logger.info(f"Starting {self.args.mode} mode for trial {self.args.trial_num}")
        logger.info(f"Holdout dataset: {self.args.holdout_dataset}")
    
    def _create_directories(self) -> None:
        """Create required directories."""
        if self.args.mode == 'pretrain':
            dirs = [self.args.output_dir, self.args.checkpoint_dir, self.args.splits_dir]
        else:
            dirs = [self.args.output_dir, self.args.splits_dir]
            
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.args.random_seed)
    
    def _load_data(self) -> ExperimentData:
        """Load and prepare experiment data."""
        # Load or create splits
        splits_file = os.path.join(self.args.splits_dir, 'dataset_splits.json')
        if os.path.exists(splits_file):
            splits = load_splits(splits_file)
            logger.info(f"Loaded existing splits from {splits_file}")
        else:
            splits = create_dataset_splits(
                {'data': vars(self.config)},
                self.args.splits_dir,
                self.args.random_seed
            )
            logger.info("Created new dataset splits")
        
        # Verify splits
        verify_splits_usage({'data': vars(self.config)}, splits)
        
        # Load datasets
        pretrain_datasets, holdout_data, all_genes = load_datasets_with_holdout(
            config={'data': vars(self.config)},
            splits=splits,
            holdout_dataset=self.args.holdout_dataset
        )
        
        return ExperimentData(
            pretrain_datasets=pretrain_datasets,
            holdout_data=holdout_data,
            all_genes=all_genes,
            splits=splits
        )
    
    def _run_mode(self, data: ExperimentData) -> None:
        """Run the specified mode."""
        if self.args.mode == 'pretrain':
            run_pretraining(
                pretrain_datasets=data.pretrain_datasets,
                all_genes=data.all_genes,
                args=self.args,
                experiment_dir=self.args.output_dir
            )
        elif self.args.mode == 'evaluate':
            run_downstream_evaluation(
                pretrain_datasets=data.pretrain_datasets,
                holdout_data=data.holdout_data,
                all_genes=data.all_genes,
                args=self.args
            )
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
    
    def _save_summary(self) -> None:
        """Save execution summary."""
        execution_summary = {
            'trial_num': self.args.trial_num,
            'mode': self.args.mode,
            'holdout_dataset': self.args.holdout_dataset,
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': str(datetime.now() - self.start_time),
            'args': vars(self.args)
        }
        
        summary_path = os.path.join(self.args.output_dir, 'execution_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(execution_summary, f, indent=2)
    
    def run(self) -> None:
        """Run the experiment."""
        try:
            self.setup()
            data = self._load_data()
            self._run_mode(data)
            self._save_summary()
        except Exception as e:
            logger.error(f"Error in experiment execution: {str(e)}")
            raise
        finally:
            duration = datetime.now() - self.start_time
            logger.info(f"Execution completed in {duration}")

def main() -> None:
    """Main execution pipeline with holdout dataset handling."""
    args = parse_arguments()
    runner = ExperimentRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
