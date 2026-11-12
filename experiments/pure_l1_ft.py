"""Discriminative fine-tune: continue from P02D ckpt with PURE L1 (no quant,
no prune, no noise, no rate term). If the bad views recover -> they were
under-trained at 30k; if they stay flat -> representation ceiling.
"""
import os
import random
import sys

import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image, _rasterize, _compose_bkgd
from budgetgs.evaluation.landing import _metrics

OUT = "/home/stud/zyuef/budgetgs-m0b/runs/joint-dual-room375k-p02d"
DEVICE = "cuda"
FT_STEPS = 8000
EVAL_EVERY = 1000
BAD = [168, 288, 40, 272]
GOOD = [144, 8, 200]

ckpt = torch.load(os.path.join(OUT, "joint-room.pt"),
                  map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
sh_degree = int(cfg.get("sh_degree", 3))
antialiased = bool(cfg.get("antialiased", False))

scene = load_360_scene("/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room",
                      factor=2)
n_views = len(scene.image_paths)
val_idcs = set(range(0, n_views, 8))
train_idcs = [i for i in range(n_views) if i not in val_idcs]
width, height = scene.width, scene.height
c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32,
                   device=DEVICE).view(n_views, 4, 4)
Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=DEVICE).view(n_views, 3, 3)
bkgd = torch.ones(3, device=DEVICE)
images = {i: _load_image(scene.image_paths[i], DEVICE, size=(width, height))
          for i in range(n_views)}

kd = (sh_degree + 1) ** 2 - 1
splats = {
    "means": torch.nn.Parameter(ckpt["means"].to(DEVICE)),
    "scales": torch.nn.Parameter(ckpt["scales"].to(DEVICE)),
    "quats": torch.nn.Parameter(ckpt["quats"].to(DEVICE)),
    "opacities": torch.nn.Parameter(ckpt["opacity"].to(DEVICE).reshape(-1)),
    "sh0": torch.nn.Parameter(ckpt["sh0"].to(DEVICE).reshape(-1, 1, 3)),
    "shn": torch.nn.Parameter(ckpt["shn"].to(DEVICE).reshape(-1, kd, 3)),
}
ss = scene.scene_scale
lrs = {"means": 1.6e-4 * ss, "scales": 1e-3, "quats": 2e-4,
       "opacities": 1e-2, "sh0": 5e-4, "shn": 2.5e-4}
opt = torch.optim.Adam([{"params": splats[k], "lr": lrs[k]} for k in splats],
                       eps=1e-15, betas=(0.9, 0.999))

def render(i):
    r, a, _ = _rasterize(splats, c2w, Ks, width, height, [i], DEVICE,
                         antialiased, sh_degree)
    return _compose_bkgd(r, a, bkgd)

def evaluate():
    out = {}
    for i in BAD + GOOD:
        out[i] = _metrics(render(i), images[i], None, DEVICE)["psnr"]
    return out

print("step  " + "".join("v%-5d" % i for i in BAD + GOOD))
base = evaluate()
print("%5d " % 0 + "".join("%6.2f" % base[i] for i in BAD + GOOD))

for step in range(1, FT_STEPS + 1):
    i = random.choice(train_idcs)
    r = render(i)
    loss = torch.abs(r - images[i]).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % EVAL_EVERY == 0:
        m = evaluate()
        print("%5d " % step + "".join("%6.2f" % m[i] for i in BAD + GOOD),
              flush=True)
print("PURE_L1_FT_DONE")
