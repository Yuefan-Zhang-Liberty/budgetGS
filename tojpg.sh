#!/bin/bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd /home/stud/zyuef/budgetgs-m0b
python - <<'PY'
from PIL import Image
import os
pairs = [
    ("runs/joint/render-compare-room.png", "runs/joint/render-compare-room.jpg"),
    ("runs/joint-dual1e4/render-compare-bonsai.png", "runs/joint-dual1e4/render-compare-bonsai.jpg"),
]
for src, dst in pairs:
    im = Image.open(src).convert("RGB")
    im.save(dst, "JPEG", quality=92, subsampling=0)
    print(dst, os.path.getsize(dst), im.size)
PY
