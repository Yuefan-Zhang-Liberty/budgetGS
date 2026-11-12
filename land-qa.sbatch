#!/bin/bash
# P0-A QA ckpt real landing under main pipeline (real_landing).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=00:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-landqa-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-landqa-%j.err

set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python3 land_qa.py
echo "finished : $(date --iso-8601=seconds)"
