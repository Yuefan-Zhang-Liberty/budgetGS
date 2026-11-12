"""P0-A quantization-aware fine-tune.

Same starting ckpt and 8k-step budget as the pure-L1 v2 run, but the forward
keeps the QAT additive-noise quantization path (noise=True) at P02D's learned
per-attr levels + mu-law, and retains the opacity/scale regularizers. No
Lagrangian rate term, no pruning -- the only changed variable vs v2 is the
quantization-aware training signal, to test whether robustness to the
position / shn quantization perturbation can be recovered.

Outputs a compatible ckpt (with mulaw_mu) and a full 39-view live eval; the
real_landing pass is a separate job (land_qa.py).
"""
import json
import os
import random
import sys

import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import ATTR_ORDER, quantized_splats
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image, _rasterize, _compose_bkgd
from budgetgs.evaluation.landing import _metrics

SRC = "/home/stud/zyuef/budgetgs-m0b/runs/joint-dual-room375k-p02d"
OUTDIR = "/home/stud/zyuef/budgetgs-m0b/runs/qa-ft-room375k"
DEVICE = "cuda"
FT_STEPS = 8000
SCENE_DIR = "/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room"
os.makedirs(OUTDIR, exist_ok=True)

ckpt = torch.load(os.path.join(SRC, "joint-room.pt"),
                  map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
sh_degree = int(cfg.get("sh_degree", 3))
antialiased = bool(cfg.get("antialiased", False))

# fixed quantization config learned by P02D
mu_t = ckpt["mulaw_mu"]
hist = json.load(open(os.path.join(SRC, "joint-room-history.json")))
int_levels = dict(hist[-1]["integer_levels"])
levels_map = {name: float(int_levels[name]) for name in ATTR_ORDER}
mu_map = {name: mu_t[j].to(DEVICE) for j, name in enumerate(ATTR_ORDER)}
print("levels_map:", levels_map, flush=True)
print("mu_map:", {k: round(float(v), 4) for k, v in mu_map.items()}, flush=True)

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


def render_raw(i):
    r, a, _ = _rasterize(splats, c2w, Ks, width, height, [i], DEVICE,
                         antialiased, sh_degree)
    return _compose_bkgd(r, a, bkgd)


for step in range(1, FT_STEPS + 1):
    i = random.choice(train_idcs)
    # QAT forward: additive-noise quantized view at fixed P02D levels/mu.
    view = quantized_splats(splats, levels_map, noise=True, mu_per_attr=mu_map)
    r, a, _ = _rasterize(view, c2w, Ks, width, height, [i], DEVICE,
                         antialiased, sh_degree)
    render = _compose_bkgd(r, a, bkgd)

    loss = torch.nn.functional.l1_loss(render, images[i])
    # retain light opacity / scale regularizers (MCMC Eq.10 style), no rate term.
    reg_o = 0.01 * torch.sigmoid(splats["opacities"]).mean()
    reg_s = 0.01 * torch.exp(splats["scales"]).mean()
    loss = loss + reg_o + reg_s

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 2000 == 0:
        print("qa step %d  loss=%.5f  reg_o=%.5f reg_s=%.5f"
              % (step, loss.item(), reg_o.item(), reg_s.item()), flush=True)

# save compatible ckpt, carrying P02D's mulaw_mu for the landing pass
out_ckpt = {
    "means": splats["means"].detach().cpu(),
    "scales": splats["scales"].detach().cpu(),
    "quats": splats["quats"].detach().cpu(),
    "opacity": splats["opacities"].detach().cpu(),
    "sh0": splats["sh0"].detach().cpu(),
    "shn": splats["shn"].detach().cpu(),
    "mulaw_mu": mu_t,
    "step": FT_STEPS,
    "config": {"sh_degree": sh_degree, "antialiased": antialiased, "factor": 2},
    "n_gaussians": int(splats["means"].shape[0]),
}
ckpt_path = os.path.join(OUTDIR, "joint-room.pt")
torch.save(out_ckpt, ckpt_path)
print("saved", ckpt_path, flush=True)

# full 39-view live eval (noise-free raw splats)
live_psnr, live_ssim, live_lpips = [], [], []
for i in val_idcs:
    m = _metrics(render_raw(i), images[i], None, DEVICE)
    live_psnr.append(m["psnr"])
    live_ssim.append(m["ssim"])
    live_lpips.append(m["lpips"])
print("LIVE 39-view mean PSNR %.4f SSIM %.4f LPIPS %.4f"
      % (sum(live_psnr) / len(live_psnr),
         sum(live_ssim) / len(live_ssim),
         sum(live_lpips) / len(live_lpips)), flush=True)
print("QA_FT_DONE")
