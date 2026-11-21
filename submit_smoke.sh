#!/bin/bash
set -euo pipefail
cd "$HOME/budgetgs-m0b"
echo "=== smoke 1: full method (default flags) ==="
SCENE=room OUT_DIR=runs/ablate/smoke-full STEPS=600 WARMUP=300 ABLATE_FLAGS="" sbatch --export=ALL ablate.sbatch
echo "=== smoke 2: base (all ablated) ==="
SCENE=room OUT_DIR=runs/ablate/smoke-base STEPS=600 WARMUP=300 ABLATE_FLAGS="--ablate-a --ablate-d --ablate-c" sbatch --export=ALL ablate.sbatch
echo "=== queue ==="
squeue -u zyuef -o '%.8i %.20j %.10M %.20R'