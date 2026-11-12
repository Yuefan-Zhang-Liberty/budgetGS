#!/bin/bash
echo "=== Total GPUs configured (exclude node22 blackwell, node3 maint) ==="
scontrol show nodes | grep -oE 'gres/gpu=[0-9]+' | awk -F= '{s+=$2} END {print "total gpu (incl all):", s}'
echo
echo "=== GPUs currently allocated ==="
scontrol show nodes | grep -oE 'AllocTRES=.*gres/gpu:[0-9]+' | grep -oE 'gres/gpu:[0-9]+' | awk -F: '{s+=$2} END {print "allocated gpu:", s}'
echo
echo "=== Pending jobs: count and total GPUs requested ==="
squeue --state=PD -h | wc -l
squeue --state=PD -h -o '%b' | grep -oE 'gpu:[0-9]+' | awk -F: '{s+=$2} END {print "pending gpu requested:", s}'
echo
echo "=== My jobs ==="
squeue -u zyuef -o '%.8i %.22j %.2t %.9M %.20S %.14R'
