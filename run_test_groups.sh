#!/usr/bin/env bash
set -uo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
for g in means scales quats opacity sh0 shn; do
  echo "===== GROUP $g ====="
  timeout 90 python test_groups.py "$g"
  echo "exit=$?"
done
echo ALL_DONE
