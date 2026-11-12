#!/bin/bash
echo "=== node19 node detail ==="
scontrol show node node19 | grep -E 'State|Gres=|AllocTRES|RealMemory|AllocMem'
echo
echo "=== ALL jobs (R+PD) that can land on node19 ==="
squeue -w node19 -a -o '%.8i %.10u %.2t %.6b %.9M %.10l %.12R'
echo
echo "=== node19 partition future reservations via scontrol ==="
scontrol show node node19 | grep -iE 'Reason|PLANNED'
