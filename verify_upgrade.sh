#!/usr/bin/env bash
set -e
cd /home/stud/zyuef/budgetgs-m0b
echo "=== mtime ==="
ls -l --time-style=+%H:%M:%S train_joint.py directionB_scene.py test_upgrade.py \
  budgetgs/modules/quantize.py budgetgs/modules/serialize.py
echo "=== markers (mulaw / mu_map / mu double) ==="
grep -c mulaw train_joint.py
grep -c mu_map directionB_scene.py
grep -c 'struct.pack("<d"' budgetgs/modules/serialize.py
echo "=== remote ast ==="
python3 -m py_compile train_joint.py directionB_scene.py test_upgrade.py \
  budgetgs/modules/quantize.py budgetgs/modules/serialize.py
echo REMOTE_AST_OK
