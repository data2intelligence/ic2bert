# IC2Bert

IC2Bert is a novel predictive model that uses masked gene expression pretraining combined with domain-specific supervised fine-tuning to enhance predictive robustness across heterogeneous immune checkpoint blockade (ICB) response cohorts.

## Abstract

Bulk RNA-seq-based prediction of immune checkpoint blockade (ICB) responses has been extensively studied to distinguish responders from non-responders. However, cohort heterogeneity remains a major challenge, hindering the robustness and generalizability of predictive models across diverse RNA-seq datasets. In this study, we present IC2Bert, a novel model that employs masked gene expression pretraining combined with domain-specific supervised fine-tuning to enhance predictive robustness across heterogeneous ICB response cohorts. To ensure an objective evaluation, we assessed the model's performance using a Leave-One-Dataset-Out Cross-Validation (LODOCV) approach. IC2Bert demonstrated significantly improved predictive accuracy and robustness compared to existing methods, effectively addressing the challenges posed by cohort heterogeneity.

## Features

- Masked gene expression pretraining
- Domain-specific supervised fine-tuning
- Leave-One-Dataset-Out Cross-Validation (LODOCV)
- Dataset-specific parameter optimization
- Comprehensive evaluation metrics
- GPU acceleration support
- Distributed training capabilities

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.1+ (for GPU support)
- cuDNN 8.9.2+

### Setup
1. Clone the repository:
```bash
git clone https://github.com/data2intelligence/ic2bert.git
cd ic2bert
```

2. Create and activate conda environment:
```bash
conda create -n ic2bert python=3.8
conda activate ic2bert
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Pretraining phase:
```bash
python -m ic2bert.main \
    --mode pretrain \
    --trial_num 1 \
    --n_expressions_bins 256 \
    --holdout_dataset "dataset_name" \
    --output_dir "./output" \
    --checkpoint_dir "./checkpoints" \
    --splits_dir "./splits" \
    --random_seed 42
```

2. Evaluation phase:
```bash
python -m ic2bert.main \
    --mode evaluate \
    --trial_num 1 \
    --n_expressions_bins 256 \
    --holdout_dataset "dataset_name" \
    --pretrained_checkpoint "./checkpoints/best_checkpoint.pkl" \
    --output_dir "./results" \
    --splits_dir "./splits"
```

### SLURM Submission

For distributed training on a SLURM cluster:
```bash
sbatch run_icb_bert.sh
```

## Datasets

The model has been evaluated on 13 different ICB response cohorts:
- CCRCC_ICB_Miao2018
- mRCC_Atezo+Bev_McDermott2018
- Melanoma_Ipilimumab_VanAllen2015
- And more...

## Model Architecture

IC2Bert consists of:
1. Gene Expression Encoder
2. Transformer-based Feature Extractor
3. Domain-specific Fine-tuning Heads
4. Dataset-specific Parameter Optimization

## Contact

For questions and support, please contact:
- Email: seongyong.park@nih.gov
- Issues: https://github.com/data2intelligence/ic2bert/issues

