#!/bin/bash

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
#source activate working_env
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate working_env

### Set CUDA variables ###
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME is set to: $CUDA_HOME"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

### Check GPU availability directly ###
echo "Checking GPU availability"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU detected\"}')"

### Optional spaCy model download ###
echo "Downloading spaCy model"
python -m spacy download en_core_web_sm
# conda install -c conda-forge spacy tqdm pandas numpy transformers regex -y

### Run your script ###
echo "Running script"
# python /home/UG/bhargavi005/contrastive-emotion-recognition/src/preprocess/preprocess_crowdflower.py
# python /home/UG/bhargavi005/contrastive-emotion-recognition/src/train_and_test_bilstm/train_crowdflower.py
# python /home/UG/bhargavi005/contrastive-emotion-recognition/src/preprocess/preprocess_ISEAR.py
# python /home/UG/bhargavi005/contrastive-emotion-recognition/src/preprocess/preprocess_WASSA2021.py
python /home/UG/bhargavi005/contrastive-emotion-recognition/src/train_and_test_bilstm/train_isear.py
# python /home/UG/bhargavi005/contrastive-emotion-recognition/src/train_and_test_bilstm/train_wassa2021.py