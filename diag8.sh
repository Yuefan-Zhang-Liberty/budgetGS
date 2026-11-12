#!/bin/bash
echo "=== /dev/nvidia* ==="
ls -la /dev/nvidia* 2>&1
echo "=== nvidia-smi (full) ==="
nvidia-smi 2>&1 | head -20
echo "=== CUDA_VISIBLE_DEVICES ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== lspci nvidia ==="
lspci 2>/dev/null | grep -i nvidia | head -12
