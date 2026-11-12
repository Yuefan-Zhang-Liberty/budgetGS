#!/bin/bash
echo "=== 1981493 err tail ==="
tail -20 /storage/slurm/zyuef/logs/bgs-render-1981493.err 2>/dev/null
echo "=== 1981493 out tail ==="
tail -12 /storage/slurm/zyuef/logs/bgs-render-1981493.out 2>/dev/null
echo "=== 1981494 out tail ==="
tail -8 /storage/slurm/zyuef/logs/bgs-render-1981494.out 2>/dev/null
echo "=== png/json ==="
ls -la /home/stud/zyuef/budgetgs-m0b/runs/joint/render-compare-room.png /home/stud/zyuef/budgetgs-m0b/runs/joint/render-compare-room.json /home/stud/zyuef/budgetgs-m0b/runs/joint-dual1e4/render-compare-bonsai.png /home/stud/zyuef/budgetgs-m0b/runs/joint-dual1e4/render-compare-bonsai.json 2>/dev/null
