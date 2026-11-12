#!/bin/bash
set -e
cd "$HOME/budgetgs-m0b"
PY=/home/stud/zyuef/.local/share/mamba/envs/main/bin/python
"$PY" -m py_compile directionA_offline.py
echo "PY_OK"
for s in room bonsai garden; do
  sbatch --export=ALL,SCENE="$s" --job-name="bgs-dirA-$s" directionA.sbatch
done
echo "--- queue ---"
squeue -u zyuef -o "%.8i %.22j %.2t %.9M %.14R"
