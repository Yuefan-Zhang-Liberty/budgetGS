#!/usr/bin/env bash
set -uo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
timeout 120 python test_groups.py shn
echo "exit=$?"
