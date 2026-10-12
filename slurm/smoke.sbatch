#!/bin/bash
#
# BudgetGS cluster smoke test: verify torch + gsplat import, GPU visibility and
# one forward + backward pass through the gsplat rasterizer.
#
# Validated before submission with:
#   python .claude/skills/slurm-tumcluster/scripts/validate_sbatch.py --tier stud slurm/smoke.sbatch

#--- Identity ---
#SBATCH --job-name=budgetgs-smoke

#--- Resources ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=00:20:00

#--- Node selection ---
# Node must be Turing+ (CC >= 7.5): gsplat's packed rasterizers use
# cooperative_groups::labeled_partition, which needs the sm_70+ match
# primitives (absent on the Pascal p6000/titan nodes).
#SBATCH --constraint="NOTBIG&GPU_CC:7.5"

#--- Logging ---
#SBATCH --output=/storage/slurm/zyuef/logs/%x-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/%x-%j.err

# ================================================================================
# Job body. No #SBATCH directives past here.
# ================================================================================

set -euo pipefail

echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "restarts : ${SLURM_RESTART_COUNT:-0}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

#--- Environment (mamba env "main") ---
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main

#--- Run ---
cd "$HOME/project-budget-gaussian-splatting/workspaces/budgetgs"
python -c "import torch; print('torch_version', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python slurm/smoke_gsplat.py

echo "finished : $(date --iso-8601=seconds)"
