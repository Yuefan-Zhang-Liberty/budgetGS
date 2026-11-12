#!/bin/bash
# Train bonsai (Mip-NeRF360, factor=2, 30k steps, SH degree 3) to produce the
# checkpoint required by the Direction B landing sweep. Single-scene version of
# the existing train.sbatch; run in the main workspace so the ckpt lands with
# the room/garden/bicycle checkpoints under runs/<scene>/<scene>_s0/.
#
#SBATCH --job-name=bgs-train-bonsai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"
#SBATCH --time=03:00:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-train-bonsai-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-train-bonsai-%j.err

set -euo pipefail
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment (mamba env "main") — activate, do not call bin/python directly
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
echo "nvcc     : $(which nvcc)"

#--- Run in the main workspace for parity with room/garden/bicycle ckpts ---
cd "$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
python -m budgetgs.cli train \
    --scene "$HOME/storage/user/zyuef/datasets/360_v2/bonsai" \
    --steps 30000 \
    --ckpt-every 1000 \
    --out "runs/bonsai" \
    --seed 0 \
    --factor 2 \
    --sh-degree 3

echo "finished : $(date --iso-8601=seconds)"
