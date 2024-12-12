#!/bin/bash
#SBATCH --job-name=ic2bert_lodocv
#SBATCH --output=logs/trial_%A_%a.out
#SBATCH --error=logs/trial_%A_%a.err
#SBATCH --array=0-909%20  # (13 datasets * 10 trials * 7 n_bins = 910 jobs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Exit on any error
set -e

# Function for error handling
handle_error() {
    local exit_code=$?
    local line_num=$1
    echo "Error on line $line_num: Exit code $exit_code"
    # Update experiment status
    if [ -f "${HOLDOUT_DIR}/experiment_metadata.json" ]; then
        update_metadata "failed" "Error on line $line_num with exit code $exit_code"
    fi
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# Function to update metadata
update_metadata() {
    local status=$1
    local message=${2:-""}
    local temp_file="${HOLDOUT_DIR}/experiment_metadata_temp.json"
    jq --arg status "$status" \
       --arg message "$message" \
       --arg end_time "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
       '. + {status: $status, end_time: $end_time, error_message: $message}' \
       "${HOLDOUT_DIR}/experiment_metadata.json" > "$temp_file" && \
    mv "$temp_file" "${HOLDOUT_DIR}/experiment_metadata.json"
}

# Load environment
echo "Loading environment..."
source ~/bin/myconda
conda activate ic2bert

module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify SLURM array task ID
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

# Print debug information
echo "Debug Information:"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Dataset definitions
DATASETS=(
    "CCRCC_ICB_Miao2018"
    "mRCC_Atezo+Bev_McDermott2018"
    "Melanoma_Ipilimumab_VanAllen2015"
    "mRCC_Atezolizumab_McDermott2018"
    "Melanoma_Nivolumab_Riaz2017"
    "NSCLC_ICB_Ravi2023"
    "Melanoma_PD1_Hugo2016"
    "PanCancer_Pembrolizumab_Yang2021"
    "Hepatocellular_Atezo+Bev_Finn2020"
    "Melanoma_PD1_Liu2019"
    "Hepatocellular_Atezolizumab_Finn2020"
    "mGC_Pembrolizumab_Kim2018"
    "Urothelial_Atezolizumab_Mariathasan2018"
)

N_BINS_VALUES=(2 4 8 16 32 64 128)

declare -A DATASET_SIZES=(
    ["CCRCC_ICB_Miao2018"]="small"
    ["mRCC_Atezo+Bev_McDermott2018"]="medium"
    ["Melanoma_Ipilimumab_VanAllen2015"]="small"
    ["mRCC_Atezolizumab_McDermott2018"]="medium"
    ["Melanoma_Nivolumab_Riaz2017"]="medium"
    ["NSCLC_ICB_Ravi2023"]="medium"
    ["Melanoma_PD1_Hugo2016"]="small"
    ["PanCancer_Pembrolizumab_Yang2021"]="medium"
    ["Hepatocellular_Atezo+Bev_Finn2020"]="large"
    ["Melanoma_PD1_Liu2019"]="large"
    ["Hepatocellular_Atezolizumab_Finn2020"]="small"
    ["mGC_Pembrolizumab_Kim2018"]="small"
    ["Urothelial_Atezolizumab_Mariathasan2018"]="large"
)

# Calculate indices
N_DATASETS=${#DATASETS[@]}
N_TRIALS=10
N_BINS=${#N_BINS_VALUES[@]}
JOBS_PER_TRIAL=$((N_DATASETS * N_BINS))

echo "Calculation variables:"
echo "N_DATASETS: ${N_DATASETS}"
echo "N_BINS: ${N_BINS}"
echo "JOBS_PER_TRIAL: ${JOBS_PER_TRIAL}"

TRIAL_NUM=$((SLURM_ARRAY_TASK_ID / JOBS_PER_TRIAL + 1))
REMAINDER=$((SLURM_ARRAY_TASK_ID % JOBS_PER_TRIAL))
DATASET_IDX=$((REMAINDER / N_BINS))
N_BINS_IDX=$((REMAINDER % N_BINS))

# Get experiment parameters
HOLDOUT_DATASET=${DATASETS[$DATASET_IDX]}
DATASET_SIZE=${DATASET_SIZES[$HOLDOUT_DATASET]}
N_BINS_VALUE=${N_BINS_VALUES[$N_BINS_IDX]}
RANDOM_SEED=$((1000 + TRIAL_NUM))

# Set dataset-specific parameters
case $DATASET_SIZE in
    "small")
        BATCH_SIZE=8
        LEARNING_RATE=0.00005
        MIN_EPOCHS=40
        PATIENCE=20
        ;;
    "medium")
        BATCH_SIZE=16
        LEARNING_RATE=0.0001
        MIN_EPOCHS=30
        PATIENCE=15
        ;;
    "large")
        BATCH_SIZE=32
        LEARNING_RATE=0.0002
        MIN_EPOCHS=25
        PATIENCE=10
        ;;
esac

# Setup directories
BASE_DIR="/data/parks34/projects/ic2bert/ablation_lodocv"
TRIAL_DIR="${BASE_DIR}/trial_${TRIAL_NUM}"
HOLDOUT_DIR="${TRIAL_DIR}/holdout_${HOLDOUT_DATASET}/bins_${N_BINS_VALUE}"

# Create directories
echo "Creating directories..."
mkdir -p "${HOLDOUT_DIR}"/{checkpoints,logs,results,splits}

# Initialize experiment metadata
cat > ${HOLDOUT_DIR}/experiment_metadata.json << EOF
{
    "trial_num": ${TRIAL_NUM},
    "dataset_name": "${HOLDOUT_DATASET}",
    "dataset_size": "${DATASET_SIZE}",
    "n_bins": ${N_BINS_VALUE},
    "batch_size": ${BATCH_SIZE},
    "learning_rate": ${LEARNING_RATE},
    "min_epochs": ${MIN_EPOCHS},
    "patience": ${PATIENCE},
    "random_seed": ${RANDOM_SEED},
    "slurm_job_id": "${SLURM_JOB_ID}",
    "slurm_array_task_id": ${SLURM_ARRAY_TASK_ID},
    "start_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "running"
}
EOF

# Log configuration
echo "Starting IC2Bert LODOCV experiment:"
echo "=================================="
echo "Time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Trial: ${TRIAL_NUM}"
echo "Holdout Dataset: ${HOLDOUT_DATASET} (${DATASET_SIZE})"
echo "N_bins: ${N_BINS_VALUE}"
echo "Random seed: ${RANDOM_SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "----------------------------------"

# Run pretraining
echo "Starting pretraining phase..."
python -m ic2bert.main \
    --mode pretrain \
    --trial_num ${TRIAL_NUM} \
    --n_expressions_bins ${N_BINS_VALUE} \
    --holdout_dataset ${HOLDOUT_DATASET} \
    --output_dir ${HOLDOUT_DIR} \
    --checkpoint_dir ${HOLDOUT_DIR}/checkpoints \
    --splits_dir ${HOLDOUT_DIR}/splits \
    --random_seed ${RANDOM_SEED} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --min_epochs ${MIN_EPOCHS} \
    --patience ${PATIENCE}

# Run evaluation
echo "Starting evaluation phase..."
python -m ic2bert.main \
    --mode evaluate \
    --trial_num ${TRIAL_NUM} \
    --n_expressions_bins ${N_BINS_VALUE} \
    --holdout_dataset ${HOLDOUT_DATASET} \
    --pretrained_checkpoint ${HOLDOUT_DIR}/checkpoints/best_checkpoint.pkl \
    --output_dir ${HOLDOUT_DIR}/results \
    --splits_dir ${HOLDOUT_DIR}/splits \
    --random_seed ${RANDOM_SEED} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --min_epochs ${MIN_EPOCHS} \
    --patience ${PATIENCE}

# Update metadata with successful completion
update_metadata "completed"

echo "Experiment completed successfully"
