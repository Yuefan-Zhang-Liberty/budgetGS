#!/bin/bash
base=/home/stud/zyuef/storage/user/zyuef/datasets/360_v2
for s in bonsai room bicycle counter kitchen stump; do
  d=$base/$s
  echo "=== $s ==="
  echo "  top: $(ls "$d" 2>&1 | tr '\n' ' ')"
  echo "  sparse raw: [$(ls -la "$d/sparse" 2>&1 | tr '\n' ' ')]"
  echo "  sparse/0: [$(ls "$d/sparse/0" 2>/dev/null | tr '\n' ' ')]"
  echo "  poses_bounds: $(test -f "$d/poses_bounds.npy" && echo yes || echo NO)"
done
