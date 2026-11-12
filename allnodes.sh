#!/bin/bash
# All GPU nodes: alloc/total + free mem + planned
echo "=== all GPU nodes ==="
for n in node8 node9 node10 node11 node12 node13 node14 node15 node16 node17 node18 node19 node20 node21 node4 node5; do
  alloc=$(scontrol show node "$n" | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' | head -1)
  total=$(scontrol show node "$n" | grep -oP 'gres/gpu=\K[0-9]+' | head -1)
  freemem=$(scontrol show node "$n" | grep -oP 'FreeMem=\K[0-9]+')
  state=$(scontrol show node "$n" | grep -oP 'State=\K\S+')
  model=$(scontrol show node "$n" | grep -oP 'gres/gpu:\K[a-z0-9_]+' | head -1)
  echo "$n | $model | gpu ${alloc:-0}/${total:-?} | freeMem ${freemem}MB | $state"
done
