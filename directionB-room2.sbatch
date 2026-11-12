#!/bin/bash
# Direction B (M0b) fixed-levels sweep on ROOM, scene-generic driver.
# Re-run: core sweep + json/curve already succeeded; this rerun only adds the
# render comparison grids after the GT-label bug fix (kind == "gt" -> "GT").
# Env: micromamba activate main (never call bin/python directly).
#
#SBATCH --job-name=bgs-m0b-room2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=1:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-m0b-room2-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-m0b-room2-%j.err

set -euo pipefail
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"

REPO="$HOME/budgetgs-m0b"
MAIN_REPO="$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
cd "$REPO"

python directionB_scene.py \
  --ckpt "$MAIN_REPO/runs/room/room_s0/ckpt-0030000.pt" \
  --scene "$HOME/storage/user/zyuef/datasets/360_v2/room" \
  --scene-name room \
  --out-dir runs/analysis

echo "finished : $(date --iso-8601=seconds)"
