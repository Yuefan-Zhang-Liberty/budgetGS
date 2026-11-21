#!/bin/bash
# Train the color-field variant (P0). Usage: run_train_cf.sh <out_subdir> [extra flags ...]
set -euo pipefail

OUT_SUBDIR="$1"
shift

cd ~/budgetgs-m0b/experiments
export MAMBA_ROOT_PREFIX=/home/stud/zyuef/.local/share/mamba
export PYTHONPATH=/home/stud/zyuef/budgetgs-m0b
export PATH=/home/stud/zyuef/storage/user/zyuef/micromamba/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate main

mkdir -p "runs/${OUT_SUBDIR}"
python train_colorfield.py \
  --scene /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
  --scene-name room \
  --out-dir "runs/${OUT_SUBDIR}" \
  --budget-mb 10 \
  --factor 2 \
  "$@"

echo "finished : $(date --iso-8601=seconds)"
