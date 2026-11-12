#!/bin/bash
for s in counter kitchen stump; do
  d=/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/$s
  echo "=== $s ==="
  ls "$d" 2>&1 | tr '\n' ' '; echo
  echo "  images_2 count: $(ls "$d/images_2" 2>/dev/null | wc -l)"
  echo "  sparse entries: $(ls "$d/sparse" 2>/dev/null | tr '\n' ' ')"
  echo "  images_4 count: $(ls "$d/images_4" 2>/dev/null | wc -l)"
done
