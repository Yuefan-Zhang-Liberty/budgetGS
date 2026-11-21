#!/bin/bash
# P0 M3: hashgrid size sweep, serial. Runs 3 full color-field trainings.
#   (12,17)=2.45MB  (8,16)=0.77MB  (12,18)=4.54MB   vs baseline (10,17)=1.94MB
set -euo pipefail

cd ~/budgetgs-m0b/experiments
export MAMBA_ROOT_PREFIX=/home/stud/zyuef/.local/share/mamba
export PYTHONPATH=/home/stud/zyuef/budgetgs-m0b
export PATH=/home/stud/zyuef/storage/user/zyuef/micromamba/bin:$PATH
eval "$(micromamba shell hook -s bash)"
micromamba activate main

run_one() {
  local subdir="$1"; local nlv="$2"; local lg2="$3"
  echo "=== sweep ${subdir} n_levels=${nlv} log2hash=${lg2} ==="
  python train_colorfield.py \
    --scene /home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room \
    --scene-name room \
    --out-dir "runs/${subdir}" \
    --budget-mb 10 --factor 2 \
    --steps 30000 --warmup 15000 --target-n 700000 \
    --n-levels "${nlv}" --log2-hashmap-size "${lg2}" \
    || echo "sweep ${subdir} FAILED"
}

run_one cf-room-f2-h12 12 17
run_one cf-room-f2-h08 8 16
run_one cf-room-f2-h18 12 18

echo "sweep finished : $(date --iso-8601=seconds)"
