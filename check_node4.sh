#!/bin/bash
echo "=== nvidia-smi on node4 ==="
ssh -o StrictHostKeyChecking=no node4 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>&1
echo "=== exit: $? ==="
