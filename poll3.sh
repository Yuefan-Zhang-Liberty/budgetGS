#!/bin/bash
sacct -n -X -j 1981508,1981509,1981493,1981494 --format=JobID,State,Elapsed,NodeList | tr '\n' ' '
echo
echo "=== memprobe 1981508 (f1) ==="
tail -12 /storage/slurm/zyuef/logs/bgs-memprobe-1981508.out 2>/dev/null
echo "=== memprobe 1981509 (f2) ==="
tail -12 /storage/slurm/zyuef/logs/bgs-memprobe-1981509.out 2>/dev/null
echo "=== pngs ==="
ls -la /home/stud/zyuef/budgetgs-m0b/runs/joint/render-compare-room.png /home/stud/zyuef/budgetgs-m0b/runs/joint-dual1e4/render-compare-bonsai.png 2>/dev/null || echo NO_PNG_YET
