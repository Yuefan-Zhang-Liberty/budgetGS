#!/bin/bash
# Discriminative pure-L1 fine-tune from P02D ckpt (no quant/prune/noise).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=01:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-purel1-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-purel1-%j.err

set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python3 pure_l1_ft.py
echo "finished : $(date --iso-8601=seconds)"
