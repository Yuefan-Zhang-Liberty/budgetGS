#!/bin/bash
# Direction B (M0b): pinned global levels sweep on room to isolate B1 root cause.
# Room is the smallest scene (578k gaussians).
#
# FIX (2026-10-28): the previous failing version called
#   /home/stud/.../envs/main/bin/python directly WITHOUT activating the env.
# That leaves main/bin/nvcc off PATH and the CUDA libs off LD_LIBRARY_PATH, so
# gsplat prints "No CUDA toolkit found" and its cached gsplat_cuda.so fails to
# load -> _C=None -> AttributeError CameraModelType. Fix = activate main exactly
# like the already-working slurm/smoke.sbatch and slurm/train.sbatch.
#
#SBATCH --job-name=bgs-m0b-room
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=1:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-m0b-room-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-m0b-room-%j.err

set -euo pipefail
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment (mamba env "main") — MUST activate; do NOT call bin/python directly
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"
python -c "import torch.utils.cpp_extension as c; print('torch_cuda_home:', c.CUDA_HOME)"

REPO="$HOME/budgetgs-m0b"
MAIN_REPO="/home/stud/zyuef/project-budget-gaussian-splatting/workspaces/budgetgs"
cd "$REPO"

python directionB_room.py \
  --ckpt "$MAIN_REPO/runs/room/room_s0/ckpt-0030000.pt" \
  --scene /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
  --out-dir runs/analysis

echo "finished : $(date --iso-8601=seconds)"
