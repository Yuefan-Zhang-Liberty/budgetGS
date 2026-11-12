"""Encode ONE group via write_mixed_container and time it. Usage: python test_groups.py <name>"""
import sys
import time
import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from directionB_scene import write_mixed_container
from budgetgs.modules.serialize import CODEC_RANGE

name = sys.argv[1]
ckpt = torch.load(
    "runs/joint/joint-bonsai.pt", map_location="cpu", weights_only=False
)
names = ["means", "scales", "quats", "opacity", "sh0", "shn"]
mu_map = {n: float(v) for n, v in zip(names, ckpt["mulaw_mu"].tolist())}
level_map = {"means": 1047, "scales": 136, "quats": 136,
             "opacity": 36, "sh0": 64, "shn": 10}
x = ckpt[name]
if x.dim() == 1:
    x = x.reshape(-1, 1)
if x.dim() > 2:
    x = x.reshape(x.shape[0], -1)
attr = {name: x.cpu().numpy().astype(float)}
groups = ((name, attr[name].shape[1]),)
print(f"[{name}] start n={len(x)} channels={attr[name].shape[1]} "
      f"lv={level_map[name]} mu={mu_map[name]:.4f}", flush=True)
t0 = time.time()
nb = write_mixed_container(
    f"runs/joint/test_{name}.bgs", attr, groups, {name: level_map[name]},
    codec=CODEC_RANGE, mu_map=mu_map,
)
print(f"[{name}] OK bytes={nb} secs={time.time()-t0:.1f}", flush=True)
