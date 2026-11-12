#!/bin/bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/storage/user/zyuef/micromamba/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate main
cd "$HOME/budgetgs-m0b"
python - <<'PY'
import json, torch
from train_joint import quantized_splats, ATTR_ORDER, CONTAINER_NAME
torch.manual_seed(0)
with open("runs/joint-dual-room375k/joint-room-result.json", encoding="utf-8") as f:
    result = json.load(f)
il = result["integer_levels"]
mr = result["mulaw_mu"]
lm = {n: int(il[n]) for n in ATTR_ORDER}
mm = {n: torch.as_tensor(mr.get(CONTAINER_NAME[n], 0.0), dtype=torch.float32) for n in ATTR_ORDER}
print("levels:", lm)
print("mu:", {n: float(mm[n]) for n in ATTR_ORDER})
splats = {
    "means": torch.rand(100, 3),
    "scales": torch.rand(100, 3),
    "quats": torch.rand(100, 4),
    "opacities": torch.rand(100, 1),
    "sh0": torch.rand(100, 1),
    "shn": torch.rand(100, 15),
}
out = quantized_splats(splats, lm, noise=False, mu_per_attr=mm)
print("quantized_splats OK")
for n in ATTR_ORDER:
    print(n, tuple(out[n].shape), round(float(out[n].min()), 4), round(float(out[n].max()), 4))
print("SMOKE_OK")
PY
