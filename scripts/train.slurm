#!/bin/bash
#SBATCH --job-name=act-train
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=50G
#SBATCH --time=20:00:00

# Activate conda environment and load CUDA
source /apps/local/anaconda3/bin/activate code-lab
module load cuda/12.1

PROJECT_ROOT='/home/badrinath.chandana/git/ACharacterInASimulation/adaptive-computation'
cd "$PROJECT_ROOT"

# Load environment variables from .env
set -a
source .env
set +a

# Run training script
#python train_synthetic_task.py --config configs/default.yaml

#python train.py --config configs/shared_kv_naive.yaml

#python train.py --config configs/naive.yaml

python train.py --config configs/babylm_recursive.yaml

python train.py --config configs/recursive.yaml

python train.py --config configs/babylm_act.yaml

python train.py --config configs/babylm_naive.yaml

python train.py --config configs/babylm_shared_kv_naive.yaml


#python train.py --config configs/act.yaml