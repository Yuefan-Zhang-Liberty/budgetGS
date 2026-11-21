#!/bin/bash
# Train the color-field variant on an arbitrary scene (P0/P1).
# Usage: run_train_cf_scene.sh <out_subdir> <scene_path> <scene_name> [extra flags ...]
set -euo pipefail

OUT_SUBDIR="$1"
SCENE_PATH="$2"
SCENE_NAME="$3"
shift 3

cd ~/budgetgs-m0b/experiments
export MAMBA_ROOT_PREFIX=/home/stud/zyuef/.local/share/mamba
export PYTHONPATH=/home/stud/zyuef/budgetgs-m0b
export PATH=/home/stud/zyuef/storage/user/zyuef/micromamba/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate main

mkdir -p "runs/${OUT_SUBDIR}"
python train_colorfield.py \
  --scene "${SCENE_PATH}" \
  --scene-name "${SCENE_NAME}" \
  --out-dir "runs/${OUT_SUBDIR}" \
  --budget-mb 10 \
  --factor 2 \
  "$@"

echo "finished : $(date --iso-8601=seconds)"
