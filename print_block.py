#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print the exact text of the real_landing evaluation block in train_joint.py."""
P = "/home/stud/zyuef/budgetgs-m0b/train_joint.py"
s = open(P, encoding="utf-8").read()
lines = s.splitlines()
for i in range(885, min(len(lines), 935)):
    print(f"{i+1}\t{lines[i]!r}")
