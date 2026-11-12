#!/bin/bash
# Train one Mip-NeRF360 scene (factor=2, 30k steps, SH degree 3) to produce a
# 30k checkpoint for the Direction-A/B landing sweeps. Scene via env SCENE.
# Runs in the main workspace so ckpt lands under runs/<scene>/<scene>_s0/.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=03:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-train-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-train-%j.err

set -euo pipefail
: "${SCENE:?SCENE must be set}"
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) scene=${SCENE}"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"

cd "$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
python -m budgetgs.cli train \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
    --steps 30000 \
    --ckpt-every 1000 \
    --out "runs/${SCENE}" \
    --seed 0 \
    --factor 2 \
    --sh-degree 3

echo "finished : $(date --iso-8601=seconds)"
