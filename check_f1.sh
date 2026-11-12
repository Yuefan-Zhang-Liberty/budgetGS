#!/bin/bash
echo "=== train-joint.sbatch ==="
cat /home/stud/zyuef/budgetgs-m0b/train-joint.sbatch
echo "=== my pending jobs ==="
squeue -u zyuef -o '%.10i %.12j %.8T %.8M %.8N'
echo "=== grep factor in train_joint.py ==="
grep -n "factor" /home/stud/zyuef/budgetgs-m0b/train_joint.py | head -10
