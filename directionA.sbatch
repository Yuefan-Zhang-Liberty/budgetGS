#!/bin/bash
# Direction A offline per-attribute bit-allocation search (no training).
# Scene passed via env: SCENE=room|bonsai|garden.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=3:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-dirA-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-dirA-%j.err

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

REPO="$HOME/budgetgs-m0b"
MAIN_REPO="$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
cd "$REPO"

python directionA_offline.py \
  --ckpt "$MAIN_REPO/runs/${SCENE}/${SCENE}_s0/ckpt-0030000.pt" \
  --scene "$HOME/storage/user/zyuef/datasets/360_v2/${SCENE}" \
  --scene-name "${SCENE}" \
  --out-dir runs/analysis \
  --budget-mb 10.0

echo "finished : $(date --iso-8601=seconds)"
