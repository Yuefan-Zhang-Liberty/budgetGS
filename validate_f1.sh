#!/bin/bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
python - <<'PY'
import os
from budgetgs.data.colmap import load_360_scene
base = os.path.expanduser("~/storage/user/zyuef/datasets/360_v2")
for scene in ("room", "bonsai"):
    s = load_360_scene(os.path.join(base, scene), factor=1)
    print(f"{scene} factor=1 OK views={len(s.image_paths)} size={s.width}x{s.height}")
PY
