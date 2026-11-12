"""Pure-L1 fine-tune v2: save compatible ckpt, full-39-view live eval, and a
real quantized landing, to quantify the total gain from removing compression.
"""
import os
import random
import sys

import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image, _rasterize, _compose_bkgd
from budgetgs.evaluation.landing import _metrics, analyze_landing

SRC = "/home/stud/zyuef/budgetgs-m0b/runs/joint-dual-room375k-p02d"
OUTDIR = "/home/stud/zyuef/budgetgs-m0b/runs/purel1-ft-room375k"
DEVICE = "cuda"
FT_STEPS = 8000
SCENE_DIR = "/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room"
os.makedirs(OUTDIR, exist_ok=True)

ckpt = torch.load(os.path.join(SRC, "joint-room.pt"),
                  map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
sh_degree = int(cfg.get("sh_degree", 3))
antialiased = bool(cfg.get("antialiased", False))

scene = load_360_scene(SCENE_DIR, factor=2)
n_views = len(scene.image_paths)
val_idcs = list(range(0, n_views, 8))
train_idcs = [i for i in range(n_views) if i not in set(val_idcs)]
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

for step in range(1, FT_STEPS + 1):
    i = random.choice(train_idcs)
    loss = torch.abs(render(i) - images[i]).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 2000 == 0:
        print("ft step %d" % step, flush=True)

# save compatible ckpt
out_ckpt = {
    "means": splats["means"].detach().cpu(),
    "scales": splats["scales"].detach().cpu(),
    "quats": splats["quats"].detach().cpu(),
    "opacity": splats["opacities"].detach().cpu(),
    "sh0": splats["sh0"].detach().cpu(),
    "shn": splats["shn"].detach().cpu(),
    "step": FT_STEPS,
    "config": {"sh_degree": sh_degree, "antialiased": antialiased, "factor": 2},
    "n_gaussians": int(splats["means"].shape[0]),
}
ckpt_path = os.path.join(OUTDIR, "joint-room.pt")
torch.save(out_ckpt, ckpt_path)
print("saved", ckpt_path, flush=True)

# full 39-view live eval
live_psnr = []
for i in val_idcs:
    live_psnr.append(_metrics(render(i), images[i], None, DEVICE)["psnr"])
print("LIVE 39-view mean PSNR %.4f" % (sum(live_psnr) / len(live_psnr)), flush=True)

# real quantized landing at 10MB
res = analyze_landing(ckpt_path, SCENE_DIR, budget_mb=10.0,
                      sh_degree=sh_degree, device=DEVICE)
print("LANDED keys:", sorted(res.keys()), flush=True)
print("LANDED result:", {k: res[k] for k in res if not isinstance(res[k], dict)},
      flush=True)
print("PURE_L1_V2_DONE")
