#!/bin/bash 

# run_pipeline_new.sh
# # Usage:
# #   ./run_pipeline_new.sh [--all] [--preprocess] [--train_test] [<dataset>] [<model>]
# #   Available datasets: crowdflower, isear, wassa
# #   Available models: bigru, lstm, mamba

### TC1 Job Script ###

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=preprocess
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

### Load required modules ###
module load cuda/11.8
module load anaconda

### Activate your environment ###
echo "Activating environment"
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate working_env

### Set CUDA variables ###
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME is set to: $CUDA_HOME"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

# Configuration
LOG_DIR="logs"
declare -A PREPROCESS_SCRIPTS=(
    ["crowdflower"]="src/preprocess/preprocess_crowdflower.py"
    ["isear"]="src/preprocess/preprocess_isear.py"
    ["wassa"]="src/preprocess/preprocess_wassa2021.py"
)

declare -A TRAIN_SCRIPTS=(
    # bigru model
    ["crowdflower:bigru"]="src/train_and_test_bigru/train_crowdflower.py"
    ["isear:bigru"]="src/train_and_test_bigru/train_isear.py"
    ["wassa:bigru"]="src/train_and_test_bigru/train_wassa2021.py"

    # bilstm model
    ["crowdflower:bilstm"]="src/train_and_test_bilstm_bert/train_crowdflower.py"
    ["isear:bilstm"]="src/train_and_test_bilstm_bert/train_isear.py"
    ["wassa:bilstm"]="src/train_and_test_bilstm_bert/train_wassa2021.py"

    # mamba model
    ["crowdflower:mamba"]="src/train_and_test_mamba/train_crowdflower.py"
    ["isear:mamba"]="src/train_and_test_mamba/train_isear.py"
    ["wassa:mamba"]="src/train_and_test_mamba/train_wassa2021.py"
)


TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ALL_DATASETS=("crowdflower" "isear" "wassa")
ALL_MODELS=("bigru" "bilstm" "mamba")

# Create log directory
mkdir -p $LOG_DIR

preprocess(){
    local dataset=$1
    echo "Starting preprocessing for $dataset..."
    script=${PREPROCESS_SCRIPTS[$dataset]}

    log_file="$LOG_DIR/${dataset}_preprocess_${TIMESTAMP}.log"

    if [ -z "$script" ]; then
        echo "No preprocessing script found for $dataset"
        return 1
    fi
    python $script 2>&1 | tee $log_file
    return ${PIPESTATUS[0]}
}

train_and_test(){
    local dataset=$1
    local model=$2
    echo "Starting training/testing for $dataset with model $model..."
    script=${TRAIN_SCRIPTS["$dataset:$model"]}

    log_file="$LOG_DIR/${dataset}_${model}_train_${TIMESTAMP}.log"

    if [ -z "$script" ]; then
        echo "No training script found for $dataset with model $model"
        return 1
    fi

    python $script 2>&1 | tee $log_file
    return ${PIPESTATUS[0]}
}

usage(){
    echo "Usage: $0 [--all] [--preprocess] [--train_test] [<dataset>] [<model>]"
    echo "  Options:"
    echo "    --all           Process all datasets"
    echo "    --preprocess    Run only preprocessing"
    echo "    --train_test    Run only training/testing"
    echo "  Datasets: ${!PREPROCESS_SCRIPTS[@]}"
    echo "  Models: ${ALL_MODELS[@]}"
    echo "  Default: Run both preprocessing and training/testing"
    exit 1
}

process_dataset() {
    local dataset=$1
    local model=$2
    local do_preprocess=$3
    local do_train_test=$4

    if $do_preprocess; then
        preprocess $dataset || return 1
    fi

    if $do_train_test; then
        train_and_test $dataset $model || return 1
    fi
}

main() {
    local do_preprocess=false
    local do_train_test=false
    local process_all=false
    local dataset=""
    local model=""

    # Parse flags
    while [ $# -gt 0 ]; do
        case $1 in
            --all)
                process_all=true
                shift
                ;;
            --preprocess)
                do_preprocess=true
                shift
                ;;
            --train_test)
                do_train_test=true
                shift
                ;;
            -*)
                echo "Invalid option: $1"
                usage
                ;;
            *)
                if [[ -z "$dataset" ]]; then
                    dataset=$1
                elif [[ -z "$model" ]]; then
                    model=$1
                else
                    echo "Unexpected argument: $1"
                    usage
                fi
                shift
                ;;
        esac
    done

    # Validate argument combinations
    if $process_all && [[ -n "$dataset" ]]; then
        echo "Error: Cannot specify both --all and dataset"
        usage
    fi

    if ! $do_preprocess && ! $do_train_test; then
        do_preprocess=true
        do_train_test=true
    fi

    if $do_train_test && [[ -z "$model" ]]; then
        echo "Error: --train_test requires specifying a model (bigru, bilstm, mamba)"
        usage
    fi

    # Determine datasets to process
    local datasets=()
    if $process_all; then
        datasets=("${ALL_DATASETS[@]}")
    elif [[ -n "$dataset" ]]; then
        if [[ ! -v PREPROCESS_SCRIPTS[$dataset] ]]; then
            echo "Invalid dataset: $dataset"
            echo "Available datasets: ${!PREPROCESS_SCRIPTS[@]}"
            exit 1
        fi
        datasets=("$dataset")
    else
        echo "Error: Must specify either --all or a dataset"
        usage
    fi

    # Process datasets
    for ds in "${datasets[@]}"; do
        echo "========================================"
        echo "Processing dataset: $ds"
        process_dataset $ds $model $do_preprocess $do_train_test || exit 1
    done

    echo "========================================"
    echo "All operations completed successfully!"
}

# Run main function
main "$@"