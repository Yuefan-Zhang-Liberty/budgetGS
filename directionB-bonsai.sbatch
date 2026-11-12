#!/bin/bash
# Direction B (M0b) fixed-levels sweep on BONSAI, scene-generic driver.
# Submit with a dependency so it starts only after the bonsai training job
# (ckpt-0030000.pt in the MAIN workspace) finishes OK.
# Env fix: micromamba activate main (same as room fixed job), never call
# bin/python directly or gsplat fails to load.
#
#SBATCH --job-name=bgs-m0b-bonsai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=1:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-m0b-bonsai-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-m0b-bonsai-%j.err

set -euo pipefail
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment (mamba env "main") — MUST activate
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"
python -c "import torch.utils.cpp_extension as c; print('torch_cuda_home:', c.CUDA_HOME)"

REPO="$HOME/budgetgs-m0b"
MAIN_REPO="$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
CKPT="$MAIN_REPO/runs/bonsai/bonsai_s0/ckpt-0030000.pt"
SCENE="$HOME/storage/user/zyuef/datasets/360_v2/bonsai"
cd "$REPO"

# Fail fast with a clear message if the training checkpoint is absent.
if [ ! -f "$CKPT" ]; then
  echo "MISSING bonsai checkpoint: $CKPT"
  ls -la "$MAIN_REPO/runs/bonsai" 2>/dev/null || echo "(no runs/bonsai dir)"
  exit 2
fi

python directionB_scene.py \
  --ckpt "$CKPT" \
  --scene "$SCENE" \
  --scene-name bonsai \
  --out-dir runs/analysis

echo "finished : $(date --iso-8601=seconds)"
