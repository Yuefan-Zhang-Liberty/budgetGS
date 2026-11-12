#!/bin/bash
set -e
cd "$HOME/budgetgs-m0b"
for s in counter kitchen stump; do
  sbatch --export=ALL,SCENE="$s" --job-name="bgs-dirA-$s" directionA.sbatch
done
echo "--- queue ---"
squeue -u zyuef -o "%.8i %.22j %.2t %.9M %.14R"
