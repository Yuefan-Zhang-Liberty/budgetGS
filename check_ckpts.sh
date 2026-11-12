#!/bin/bash
M=/home/stud/zyuef/project-budget-gaussian-splatting/workspaces/budgetgs/runs
for s in counter kitchen stump; do
  echo "=== $s ckpt ==="
  ls -la --time-style=+%m-%d_%H:%M "$M/$s/${s}_s0/ckpt-0030000.pt" 2>/dev/null || echo "NO CKPT"
done
echo "############ counter train log (tail) ############"
tail -18 /storage/slurm/zyuef/logs/bgs-train-counter-1967681.out 2>/dev/null
echo "############ kitchen (tail) ############"
tail -8 /storage/slurm/zyuef/logs/bgs-train-kitchen-1967682.out 2>/dev/null
echo "############ stump (tail) ############"
tail -8 /storage/slurm/zyuef/logs/bgs-train-stump-1967683.out 2>/dev/null
