#!/bin/bash
echo "=== my jobs ==="
squeue -u zyuef -o '%.10i %.9P %.12j %.8T %.10M %.8N'
echo "=== queue by partition ==="
squeue -o '%.10i %.8T %P' -p gpu-a40,gpu-l40s,gpu-h100 2>/dev/null | awk '{c[$2"|"$3]++} END {for (k in c) print k, c[k]}' | sort
echo "=== nodes ==="
sinfo -o '%n %P %G %t' 2>/dev/null | head -30
