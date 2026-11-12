#!/bin/bash
set -e
D=/home/stud/zyuef/storage/user/zyuef/datasets/360_v2
echo "=== room/ 顶层 ==="
ls -la "$D/room/" 2>&1 | head -15
echo "=== room/images 文件数 ==="
ls "$D/room/images/" 2>&1 | wc -l
echo "=== room/images 样例 ==="
ls "$D/room/images/" 2>&1 | head -3
echo "=== room/images_2 文件数 ==="
ls "$D/room/images_2/" 2>&1 | wc -l
echo "=== bonsai/images 文件数 ==="
ls "$D/bonsai/images/" 2>&1 | wc -l
echo "=== bonsai/images_2 文件数 ==="
ls "$D/bonsai/images_2/" 2>&1 | wc -l
echo "=== room/sparse 结构 ==="
find "$D/room/sparse" -maxdepth 2 2>&1 | head -10
echo "=== 磁盘 ==="
df -h / 2>&1 | tail -1
