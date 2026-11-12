#!/bin/bash
cd /home/stud/zyuef/budgetgs-m0b
rm -f rc-room.* rc-bonsai.*
split -b 10485760 runs/joint/render-compare-room.png rc-room.
split -b 10485760 runs/joint-dual1e4/render-compare-bonsai.png rc-bonsai.
ls -la rc-room.* rc-bonsai.*
