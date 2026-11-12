#!/bin/bash
echo "=== nvidia kernel modules ==="
lsmod 2>/dev/null | grep -i nvidia || echo "NO nvidia modules loaded"
echo
echo "=== /proc/driver/nvidia/version ==="
cat /proc/driver/nvidia/version 2>&1 || echo "no /proc/driver/nvidia"
echo
echo "=== kernel log nvidia (last 15) ==="
dmesg 2>/dev/null | grep -i nvidia | tail -15 || echo "dmesg unavailable"
