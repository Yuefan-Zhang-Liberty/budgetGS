#!/bin/bash
# Report total vs allocated GPUs per node for the gsplat-compatible nodes.
for n in node4 node5 node8 node9 node10 node11 node12 node13 node14 node15 node16 node17 node18 node19 node20 node21; do
  info=$(scontrol show node "$n")
  tot=$(echo "$info" | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
  used=$(echo "$info" | grep -oE 'AllocTRES=.*gres/gpu=[0-9]+' | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
  state=$(echo "$info" | grep -oE 'State=[A-Z+]+' | head -1 | cut -d= -f2)
  if [ -z "$used" ]; then used=0; fi
  free=$((tot - used))
  echo "$n  state=$state  gpu total=$tot used=$used free=$free"
done
