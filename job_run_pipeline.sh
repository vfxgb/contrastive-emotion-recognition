#!/bin/bash

# Usage:
#     ./run_pipeline.sh [--force_preprocess] <dataset> <model>
# Example:
#     ./run_pipeline.sh --force_preprocess crowdflower bilstm_glove
#   Available datasets: crowdflower, isear, wassa
#   Available models: bilstm_glove, bilstm_bert, mamba
### TC1 Job Script ###

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --job-name=no_attention_isear
#SBATCH --output=./outputlogs/output_%x_%j.out
#SBATCH --error=./errorlogs/error_%x_%j.err

### Load required modules ###
module load cuda/11.8
module load anaconda

### Activate your environment ###
echo "Activating environment"
#source activate working_env
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yaml
conda activate SC4001

### Set CUDA variables ###
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME is set to: $CUDA_HOME"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "Checking GPU availability"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU detected\"}')"

export PYTHONPATH="$PYTHONPATH:./src"
echo $PYTHONPATH

# === Configuration ===
LOG_DIR="logs"
GLOVE_DIR="glove"
GLOVE_FILE="$GLOVE_DIR/glove.6B.300d.txt"
GLOVE_ZIP_URL="https://www.kaggle.com/api/v1/datasets/download/thanakomsn/glove6b300dtxt"
GLOVE_ZIP="$GLOVE_DIR/glove.6B.300d.zip"
BILSTM_BERT_RESULTS_DIR="results/bilstm_bert"
BILSTM_GLOVE_RESULTS_DIR="results/bilstm_glove"
MAMBA_RESULTS_DIR="results/mamba"

# === VALIDATION SETUP === 
VALID_DATASETS=("crowdflower" "isear" "wassa")
VALID_MODELS=("bilstm_glove" "bilstm_bert" "mamba")

FORCE_PREPROCESS=false
DATASET=""
MODEL=""

# === Scripts ===
declare -A PREPROCESS_SCRIPTS=(
    ["crowdflower"]="src/preprocess/preprocess_crowdflower.py"
    ["isear"]="src/preprocess/preprocess_isear.py"
    ["wassa"]="src/preprocess/preprocess_wassa2021.py"
)

declare -A TRAIN_SCRIPTS=(
    # bilstm model without glove 
    ["crowdflower:bilstm_bert"]="src/train_and_test_bilstm_bert/train_crowdflower.py"
    ["isear:bilstm_bert"]="src/train_and_test_bilstm_bert/train_isear.py"
    ["wassa:bilstm_bert"]="src/train_and_test_bilstm_bert/train_wassa2021.py"

    # bilstm model with glove 
    ["crowdflower:bilstm_glove"]="src/train_and_test_bilstm_glove/train_crowdflower.py"
    ["isear:bilstm_glove"]="src/train_and_test_bilstm_glove/train_isear.py"
    ["wassa:bilstm_glove"]="src/train_and_test_bilstm_glove/train_wassa2021.py"

    # mamba model
    ["crowdflower:mamba"]="src/train_and_test_mamba/train_crowdflower.py"
    ["isear:mamba"]="src/train_and_test_mamba/train_isear.py"
    ["wassa:mamba"]="src/train_and_test_mamba/train_wassa2021.py"
)

download_glove() {
    if [ -f "$GLOVE_FILE" ]; then
        echo "GloVe embeddings already exist at $GLOVE_FILE"
        return 0
    fi

    echo "GloVe embeddings not found. Downloading..."
    mkdir -p "$GLOVE_DIR"
    wget -O "$GLOVE_ZIP" "$GLOVE_ZIP_URL"

    if [ $? -ne 0 ]; then
        echo "Error downloading GloVe embeddings."
        exit 1
    fi

    echo "Unzipping GloVe embeddings..."
    unzip "$GLOVE_ZIP" -d "$GLOVE_DIR"

    if [ ! -f "$GLOVE_FILE" ]; then
        echo "Failed to unzip or find $GLOVE_FILE"
        exit 1
    fi

    echo "GloVe embeddings downloaded and extracted to $GLOVE_FILE"

    # Delete the zip file after extraction
    echo "Deleting the zip file..."
    rm -f "$GLOVE_ZIP"
    echo "Zip file deleted."
}

# === USAGE ===
usage() {
    echo "Usage: $0 [--force_preprocess] <dataset> <model>"
    echo "  Options:"
    echo "    --force_preprocess    if set, preprocesses data again"
    echo "                          if not set, preprocesses data if not done before"
    echo "  Datasets: ${VALID_DATASETS[*]}"
    echo "  Models: ${VALID_MODELS[*]}"
    echo "  Please first train on crowdflower and then finetune on isear and wassa."
    exit 1
}

# === ARG PARSING ===

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force_preprocess)
            FORCE_PREPROCESS=true
            shift
            ;;
        crowdflower|isear|wassa)
            if [ -z "$DATASET" ]; then
                DATASET=$1
            else
                echo "Error: Dataset already set to $DATASET. Unexpected value: $1"
                exit 1
            fi
            shift
            ;;
        bilstm_glove|bilstm_bert|mamba)
            if [ -z "$MODEL" ]; then
                MODEL=$1
            else
                echo "Error: Model already set to $MODEL. Unexpected value: $1"
                exit 1
            fi
            shift
            ;;
        -*)
            echo "Invalid option: $1"
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# === VALIDATION CHECKS ===

if [[ -z "$DATASET" || -z "$MODEL" ]]; then
    usage
fi

# === GET SCRIPT TO RUN ===

PREPROCESS_SCRIPT="${PREPROCESS_SCRIPTS[$DATASET]}"
TRAIN_SCRIPT="${TRAIN_SCRIPTS["$DATASET:$MODEL"]}"

if [ -z "$PREPROCESS_SCRIPT" ] || [ -z "$TRAIN_SCRIPT" ]; then
    echo "No script found for dataset=$DATASET and model=$MODEL"
    exit 1
fi

# === LOGGING ===

mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "log_dir is "
echo $LOG_DIR

echo "dataset is "
echo $DATASET

echo "model is "
echo $MODEL

echo "timestamp is "
echo $TIMESTAMP

LOG_FILE="$LOG_DIR/${DATASET}_${MODEL}_${TIMESTAMP}.log"

echo "Checking for spaCy model..."

python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Downloading spaCy model en_core_web_sm..."
    python -m spacy download en_core_web_sm
else
    echo "spaCy model en_core_web_sm already installed."
fi

# === CONDITIONAL PREPROCESSING ===
if [ "$MODEL" = "bilstm_glove" ]; then
    download_glove
    echo "Made directory"
    mkdir -p $BILSTM_GLOVE_RESULTS_DIR
elif [ "$MODEL" = "bilstm_bert" ]; then 
    mkdir -p $BILSTM_BERT_RESULTS_DIR
elif [ "$MODEL" = "mamba" ]; then 
    mkdir -p $MAMBA_RESULTS_DIR
fi

echo "Running preprocessing and training for dataset=$DATASET model=$MODEL (force_preprocess=$FORCE_PREPROCESS)"

if [ "$FORCE_PREPROCESS" = true ]; then
    if [ "$MODEL" = "bilstm_glove" ]; then
        echo "hi A"
        python "$PREPROCESS_SCRIPT" --with_glove --force_preprocess 2>&1 | tee -a "$LOG_FILE"
    else
        echo "hi B"
        python "$PREPROCESS_SCRIPT" --force_preprocess 2>&1 | tee -a "$LOG_FILE"
    fi
else
    if [ "$MODEL" = "bilstm_glove" ]; then
        echo "hi A"
        python "$PREPROCESS_SCRIPT" --with_glove 2>&1 | tee -a "$LOG_FILE"
    else
        echo "hi B"
        python "$PREPROCESS_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    fi
fi

# === TRAINING ===
echo "[Train] Running training for dataset=$DATASET model=$MODEL"
python "$TRAIN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
