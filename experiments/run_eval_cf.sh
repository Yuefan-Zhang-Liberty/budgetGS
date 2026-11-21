#!/bin/bash
# Evaluate real 8-bit color-field landing on an existing joint-room.pt ckpt.
set -euo pipefail

cd ~/budgetgs-m0b/experiments
export MAMBA_ROOT_PREFIX=/home/stud/zyuef/.local/share/mamba
export PYTHONPATH=/home/stud/zyuef/budgetgs-m0b
export PATH=/home/stud/zyuef/storage/user/zyuef/micromamba/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate main

python eval_cf_ckpt.py \
  --ckpt runs/cf-room-f2/joint-room.pt \
  --result-json runs/cf-room-f2/joint-room-result.json \
  --scene /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
  --factor 2

echo "finished : $(date --iso-8601=seconds)"
