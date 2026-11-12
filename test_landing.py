"""Isolate landing bottleneck: encode/decode/dequant only (no render, no LPIPS)."""
import time
import torch
import sys

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from directionB_scene import write_mixed_container, dequant_mixed
from budgetgs.modules.serialize import CODEC_RANGE, read_container

ckpt = torch.load(
    "runs/joint/joint-bonsai.pt", map_location="cpu", weights_only=False
)
names = ["means", "scales", "quats", "opacity", "sh0", "shn"]
mu_map = {n: float(v) for n, v in zip(names, ckpt["mulaw_mu"].tolist())}
level_map = {"means": 1047, "scales": 136, "quats": 136,
             "opacity": 36, "sh0": 64, "shn": 10}
attributes = {}
for name in names:
    x = ckpt[name]
    if x.dim() == 1:
        x = x.reshape(-1, 1)
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    attributes[name] = x.cpu().numpy().astype(float)
groups = tuple((n, attributes[n].shape[1]) for n in names)

print("levels:", level_map)
print("mu:", {k: round(v, 4) for k, v in mu_map.items()}, flush=True)
t0 = time.time()
nb = write_mixed_container(
    "runs/joint/test_land.bgs", attributes, groups, level_map,
    codec=CODEC_RANGE, mu_map=mu_map,
)
print("encode OK bytes=%d secs=%.1f" % (nb, time.time() - t0), flush=True)
t0 = time.time()
decoded = read_container("runs/joint/test_land.bgs", groups=groups)
print("decode OK secs=%.1f" % (time.time() - t0), flush=True)
t0 = time.time()
recon = dequant_mixed(decoded, level_map)
print("dequant OK secs=%.1f" % (time.time() - t0), flush=True)
