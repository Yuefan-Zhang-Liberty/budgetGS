import os, sys, json
import numpy as np
import torch
sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from directionB_scene import write_mixed_container, dequant_mixed, split_shn_groups
from budgetgs.modules.serialize import CODEC_ZLIB, read_container
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image
from budgetgs.evaluation.landing import _render_views, _metrics, _splats_tensors

ROOT = "/home/stud/zyuef/budgetgs-m0b"
SCENE_DIR = "/home/stud/zyuef/storage/user/zyuef/datasets/360_v2/room"
DEVICE = "cuda"
ATTRS = ["means", "scales", "quats", "opacities", "sh0", "shn"]

p02d_ck = torch.load(os.path.join(ROOT, "runs/joint-dual-room375k-p02d/joint-room.pt"),
                     map_location="cpu", weights_only=False)
mu_t = p02d_ck["mulaw_mu"]
ATTR_ORDER = ["means", "scales", "quats", "opacities", "sh0", "shn"]
hist = json.load(open(os.path.join(ROOT, "runs/joint-dual-room375k-p02d/joint-room-history.json")))
int_levels = hist[-1]["integer_levels"]
level_map = {("opacity" if k == "opacities" else k): v for k, v in int_levels.items()}
mu_map = {("opacity" if ATTR_ORDER[j] == "opacities" else ATTR_ORDER[j]): float(mu_t[j])
          for j in range(6)}
groups = (("means", 3), ("scales", 3), ("quats", 4), ("opacity", 1), ("sh0", 3), ("shn", 45))

scene = load_360_scene(SCENE_DIR, factor=2)
n_views = len(scene.image_paths)
val_idcs = list(range(0, n_views, 8))
width, height = scene.width, scene.height
images = {i: _load_image(scene.image_paths[i], DEVICE, size=(width, height))
          for i in val_idcs}
lpips_model = None

def build_attrs(ck):
    a = {}
    a["means"] = ck["means"].reshape(-1, 3).numpy().astype(np.float64)
    a["scales"] = ck["scales"].reshape(-1, 3).numpy().astype(np.float64)
    a["quats"] = ck["quats"].reshape(-1, 4).numpy().astype(np.float64)
    a["opacity"] = ck["opacity"].reshape(-1, 1).numpy().astype(np.float64)
    a["sh0"] = ck["sh0"].reshape(-1, 3).numpy().astype(np.float64)
    a["shn"] = ck["shn"].reshape(-1, 45).numpy().astype(np.float64)
    return a

def live_splats(ck):
    return {
        "means": ck["means"].to(DEVICE),
        "scales": ck["scales"].to(DEVICE),
        "quats": ck["quats"].to(DEVICE),
        "opacities": ck["opacity"].to(DEVICE).reshape(-1),
        "sh0": ck["sh0"].to(DEVICE).reshape(-1, 1, 3),
        "shn": ck["shn"].to(DEVICE).reshape(-1, 15, 3),
    }

def score(splats):
    tot = 0.0
    for i in val_idcs:
        v = _render_views(splats, scene, [i], DEVICE, 3, False)
        tot += _metrics(v, images[i], lpips_model, DEVICE)["psnr"]
    return tot / len(val_idcs)

for tag, d in [("P02D", "runs/joint-dual-room375k-p02d"),
               ("V2FT", "runs/purel1-ft-room375k")]:
    ck = torch.load(os.path.join(ROOT, d, "joint-room.pt"), map_location="cpu", weights_only=False)
    attrs = build_attrs(ck)
    path = "/tmp/%s-sens.bgs" % tag
    nb = write_mixed_container(path, attrs, groups, level_map, codec=CODEC_ZLIB, mu_map=mu_map)
    decoded = read_container(path, groups=split_shn_groups(groups))
    recon = dequant_mixed(decoded, level_map)
    rec_splats = _splats_tensors(recon, DEVICE, 3)
    live = live_splats(ck)
    print("==== %s (%.4fMB)" % (tag, nb / 1e6))
    print("  ALL-LIVE  %.3f" % score(live))
    print("  ALL-RECON %.3f" % score(rec_splats))
    for attr in ATTRS:
        h = dict(live)
        h[attr] = rec_splats[attr]
        print("  only-%-9s quantized: %.3f" % (attr, score(h)))
