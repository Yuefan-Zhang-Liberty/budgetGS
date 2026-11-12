#!/bin/bash
# Direction B (M0b): pinned global levels sweep on room to isolate B1 root cause.
# Room is the smallest scene (578k gaussians); exclude node22 (Blackwell sm_120).
#SBATCH --job-name=bgs-m0b-room
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node4,node5,node15,node20,node21
#SBATCH --exclude=node3,node9,node13,node22
#SBATCH --time=1:30:00
#SBATCH --output=/storage/slurm/zyuef/logs/bgs-m0b-room-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/bgs-m0b-room-%j.err

set -euo pipefail
echo "job      : ${SLURM_JOB_ID} (${SLURM_JOB_NAME})"
echo "node     : $(hostname)"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started  : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
PY="$HOME/.local/share/mamba/envs/main/bin/python"
REPO="$HOME/budgetgs-m0b"
MAIN_REPO="/home/stud/zyuef/project-budget-gaussian-splatting/workspaces/budgetgs"

cd "$REPO"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

"$PY" directionB_room.py \
  --ckpt "$MAIN_REPO/runs/room/room_s0/ckpt-0030000.pt" \
  --scene /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
  --out-dir runs/analysis

echo "finished : $(date --iso-8601=seconds)"
