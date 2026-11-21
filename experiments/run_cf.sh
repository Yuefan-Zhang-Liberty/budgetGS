#!/bin/bash
# Fit a shared hashgrid color field against frozen geometry; compare vs SH ref.
# Usage: run_cf.sh <out_subdir> <steps> [extra fit_color_field.py flags ...]
set -euo pipefail

OUT_SUBDIR="$1"
STEPS="$2"
shift 2

cd ~/budgetgs-m0b/experiments
export MAMBA_ROOT_PREFIX=/home/stud/zyuef/.local/share/mamba
export PYTHONPATH=/home/stud/zyuef/budgetgs-m0b
export PATH=/home/stud/zyuef/storage/user/zyuef/micromamba/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate main

mkdir -p "runs/cf-smoke/${OUT_SUBDIR}"
python fit_color_field.py \
  --scene_dir /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
  --ckpt /home/stud/zyuef/budgetgs-m0b/experiments/runs/ablate/room-ad/joint-room.pt \
  --out_dir "runs/cf-smoke/${OUT_SUBDIR}" \
  --steps "${STEPS}" \
  --val_every 500 \
  --factor 2 \
  "$@"

echo "finished : $(date --iso-8601=seconds)"
