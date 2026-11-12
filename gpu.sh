#!/bin/bash
for n in node4 node5 node14 node15 node16 node17 node18 node19 node20; do
  echo "== $n =="
  scontrol show node "$n" | grep -E 'State=|Gres=|GresUsed='
done
