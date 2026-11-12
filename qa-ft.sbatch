#!/bin/bash
# P0-A quantization-aware fine-tune: QAT noise path + regs, 8k steps, live eval.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=02:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-qaft-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-qaft-%j.err

set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python3 quant_aware_ft.py
echo "finished : $(date --iso-8601=seconds)"
