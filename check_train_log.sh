#!/bin/bash
echo "================ counter (1967681) full tail ================"
tail -22 /storage/slurm/zyuef/logs/bgs-train-1967681.out
echo
for j in 1967681 1967682 1967683; do
  echo "================ job $j key lines ================"
  grep -aiE "30000|number of points|points|eval|saving|saved|psnr|done" /storage/slurm/zyuef/logs/bgs-train-$j.out | tail -8
done
