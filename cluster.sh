#!/bin/bash
# For each gsplat-compatible node, show free GPUs, free RAM (GB), free CPUs.
for n in node4 node5 node8 node9 node10 node11 node12 node13 node14 node15 node16 node17 node18 node19 node20 node21; do
  info=$(scontrol show node "$n")
  gtot=$(echo "$info" | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
  gused=$(echo "$info" | grep -oE 'AllocTRES=.*gres/gpu=[0-9]+' | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
  ctot=$(echo "$info" | grep -oE 'CPUTot=[0-9]+' | cut -d= -f2)
  cused=$(echo "$info" | grep -oE 'CPUAlloc=[0-9]+' | cut -d= -f2)
  rtot=$(echo "$info" | grep -oE 'RealMemory=[0-9]+' | cut -d= -f2)
  rused=$(echo "$info" | grep -oE 'AllocMem=[0-9]+' | cut -d= -f2)
  if [ -z "$gused" ]; then gused=0; fi
  if [ -z "$cused" ]; then cused=0; fi
  if [ -z "$rused" ]; then rused=0; fi
  gfree=$((gtot - gused)); cfree=$((ctot - cused)); rfree=$(( (rtot - rused) / 1024 ))
  state=$(echo "$info" | grep -oE 'State=[A-Z+]+' | head -1 | cut -d= -f2)
  echo "$n  gpu_free=$gfree  ram_free=${rfree}GB  cpu_free=$cfree  state=$state"
done
