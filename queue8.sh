#!/bin/bash
# What is queued against node8, priority-sorted, and where do we stand?
echo "=== PD jobs requesting node8 (priority desc) ==="
squeue -w node8 --state=PD -o '%.8i %.10u %.9P %.4D %.10b %.12Q' | sort -k6 -rn | head -20
echo "=== total PD jobs cluster-wide ==="
squeue --state=PD -h | wc -l
echo "=== node8 current running jobs ==="
squeue -w node8 --state=R -o '%.8i %.10u %.10b %.10M'
