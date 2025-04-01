#!/bin/bash

# run_pipeline.sh
# Usage:
#   ./run_pipeline.sh [--all] [--preprocess] [--train_test] [<dataset>]
#   Available datasets: crowdflower, isear, wassa

# Configuration
LOG_DIR="logs"
declare -A PREPROCESS_SCRIPTS=(
    ["crowdflower"]="src/preprocess/preprocess_crowdflower.py"
    ["isear"]="src/preprocess/preprocess_ISEAR.py"
    ["wassa"]="src/preprocess/preprocess_WASSA2021.py"
)
declare -A TRAIN_SCRIPTS=(
    ["crowdflower"]="src/train_and_test/train_crowdflower.py"
    ["isear"]="src/train_and_test/train_isear.py"
    ["wassa"]="src/train_and_test/train_wassa2021.py"
)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ALL_DATASETS=("crowdflower" "isear" "wassa")

# Create log directory
mkdir -p $LOG_DIR

preprocess() {
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

train_and_test() {
    local dataset=$1
    echo "Starting training/testing for $dataset..."
    script=${TRAIN_SCRIPTS[$dataset]}
    log_file="$LOG_DIR/${dataset}_train_${TIMESTAMP}.log"
    
    if [ -z "$script" ]; then
        echo "No training script found for $dataset"
        return 1
    fi
    
    python $script 2>&1 | tee $log_file
    return ${PIPESTATUS[0]}
}

usage() {
    echo "Usage: $0 [--all] [--preprocess] [--train_test] [<dataset>]"
    echo "  Options:"
    echo "    --all           Process all datasets"
    echo "    --preprocess    Run only preprocessing"
    echo "    --train_test    Run only training/testing"
    echo "  Datasets: ${!PREPROCESS_SCRIPTS[@]}"
    echo "  Default: Run both preprocessing and training/testing"
    exit 1
}

process_dataset() {
    local dataset=$1
    local do_preprocess=$2
    local do_train_test=$3

    if $do_preprocess; then
        preprocess $dataset || return 1
    fi
    
    if $do_train_test; then
        train_and_test $dataset || return 1
    fi
}

main() {
    local do_preprocess=false
    local do_train_test=false
    local process_all=false
    local dataset=""

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
                    shift
                else
                    echo "Unexpected argument: $1"
                    usage
                fi
                ;;
        esac
    done

    # Validate arguments
    if $process_all && [[ -n "$dataset" ]]; then
        echo "Error: Cannot specify both --all and dataset"
        usage
    fi

    # Default to both operations if none specified
    if ! $do_preprocess && ! $do_train_test; then
        do_preprocess=true
        do_train_test=true
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
        process_dataset $ds $do_preprocess $do_train_test || exit 1
    done

    echo "========================================"
    echo "All operations completed successfully!"
}

# Run main function
main "$@"
