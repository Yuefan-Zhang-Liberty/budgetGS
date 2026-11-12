#!/bin/bash
# Smoke test (no nodelist pin): 300 steps, quantize after 150, prune hard,
# exercises the full path incl. mu-law + range coder landing.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --time=00:15:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-joint-smoke-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-joint-smoke-%j.err

set -euo pipefail
SCENE=room
echo "job      : ${SLURM_JOB_ID} smoke2 scene=${SCENE}"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

cd "$HOME/budgetgs-m0b"
python train_joint.py \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
    --scene-name "smoke2-room" \
    --out-dir "runs/joint-smoke2" \
    --steps 300 \
    --warmup 150 \
    --target-n 80000 \
    --budget-mb 10 \
    --prune-every 50 \
    --sh-degree 3 \
    --seed 0 \
    --factor 2

echo "SMOKE2 FINISHED"
