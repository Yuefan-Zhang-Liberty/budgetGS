"""Real-landing P0-A QA fine-tuned ckpt under the MAIN pipeline (real_landing),
reusing P02D's per-attr levels; mu carried inside the QA ckpt itself."""
import json
import os
import sys

import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import ATTR_ORDER, real_landing
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image

ROOT = "/home/stud/zyuef/budgetgs-m0b"
QA = os.path.join(ROOT, "runs/qa-ft-room375k")
P02D = os.path.join(ROOT, "runs/joint-dual-room375k-p02d")
SCENE_DIR = "/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room"
DEVICE = "cuda"

ck = torch.load(os.path.join(QA, "joint-room.pt"), map_location="cpu", weights_only=False)
cfg = ck.get("config", {})
sh_degree = int(cfg.get("sh_degree", 3))
antialiased = bool(cfg.get("antialiased", False))

mu_t = ck["mulaw_mu"]
mu_per_attr = {name: mu_t[j] for j, name in enumerate(ATTR_ORDER)}

hist = json.load(open(os.path.join(P02D, "joint-room-history.json")))
int_levels = dict(hist[-1]["integer_levels"])


class _Levels:
    def __init__(self, lv): self._lv = lv
    def integer_targets(self): return dict(self._lv)


scene = load_360_scene(SCENE_DIR, factor=2)
n_views = len(scene.image_paths)
val_idcs = list(range(0, n_views, 8))
width, height = scene.width, scene.height
images = {i: _load_image(scene.image_paths[i], DEVICE, size=(width, height))
          for i in val_idcs}

splats = {k: ck[k].to(DEVICE) for k in ["means", "scales", "quats", "opacity", "sh0", "shn"]}
splats["opacities"] = splats.pop("opacity").reshape(-1)
kd = (sh_degree + 1) ** 2 - 1
splats["sh0"] = splats["sh0"].reshape(-1, 1, 3)
splats["shn"] = splats["shn"].reshape(-1, kd, 3)

result = real_landing(
    splats, _Levels(int_levels), scene, "room", None, None,
    width, height, images, val_idcs, DEVICE, antialiased,
    sh_degree, QA, 10.0, mu_per_attr=mu_per_attr,
)
print("RESULT", json.dumps(result, indent=2, default=str))
with open(os.path.join(QA, "joint-room-result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=str)
print("LAND_QA_DONE")
