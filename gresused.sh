#!/bin/bash
# Dump full node state for all nodes, focus on Gres/GresUsed/State.
for n in node4 node5 node8 node9 node10 node11 node12 node13 node14 node15 node16 node17 node18 node19 node20 node21; do
  echo "===== $n ====="
  scontrol show node "$n" | grep -E 'State=|Gres=|GresUsed=|RealMemory=|AllocMem='
done
