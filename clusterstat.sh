#!/bin/bash
# GPU allocation on healthy nodes (sm_86/8.9/9.0)
echo "=== healthy nodes (alloc gpu / total) ==="
sinfo -N -h -o '%N %f' | grep -E '8\.6|8\.9|9\.0' | awk '{print $1}' | sort -u | while read -r n; do
  alloc=$(scontrol show node "$n" | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' | head -1)
  total=$(scontrol show node "$n" | grep -oP 'gres/gpu=\K[0-9]+' | head -1)
  state=$(scontrol show node "$n" | grep -oP 'State=\K\S+')
  echo "$n  alloc=${alloc:-0}/${total:-?}  state=$state"
done
echo "=== global queue ==="
echo -n "PD: "; squeue -h -t PD | wc -l
echo -n "R:  "; squeue -h -t R | wc -l
