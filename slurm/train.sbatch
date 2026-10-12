#!/bin/bash
#
# BudgetGS training: bicycle / garden / room on Mip-NeRF 360 (factor=2),
# 30k steps, checkpoint every 1k steps, SH degree 3.
#
# Slurm array job: task 0..2 -> bicycle / garden / room.
# Validated before submission with:
#   python .claude/skills/slurm-tumcluster/scripts/validate_sbatch.py --tier stud slurm/train.sbatch

#--- Identity ---
#SBATCH --job-name=budgetgs-train

#--- Resources ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=03:00:00

#--- Array ---
#SBATCH --array=0-2

#--- Node selection ---
# Node must be Turing+ (CC >= 7.5): gsplat's packed rasterizers use
# cooperative_groups::labeled_partition, which needs the sm_70+ match
# primitives (absent on the Pascal p6000/titan nodes).
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"

#--- Logging ---
#SBATCH --output=/storage/slurm/zyuef/logs/%x-%A-%a.out
#SBATCH --error=/storage/slurm/zyuef/logs/%x-%A-%a.err

# ================================================================================
# Job body. No #SBATCH directives past here.
# ================================================================================

set -euo pipefail

SCENES=(bicycle garden room)
SCENE="${SCENES[${SLURM_ARRAY_TASK_ID}]}"

echo "job      : ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "scene    : ${SCENE}"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment (mamba env "main") ---
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

#--- Run ---
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
