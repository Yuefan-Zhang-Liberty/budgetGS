#!/usr/bin/env bash
set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
python -m py_compile train_joint.py
echo REMOTE_SYNTAX_OK
echo "AdamW markers: $(grep -c 'AdamW' train_joint.py)"
echo "clamp markers: $(grep -c 'raw_c = mu_raw.clamp' train_joint.py)"
