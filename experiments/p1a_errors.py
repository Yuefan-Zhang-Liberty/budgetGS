import os, sys, json
import numpy as np
import torch
sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from directionB_scene import write_mixed_container, dequant_mixed, group_errors, split_shn_groups
from budgetgs.modules.serialize import CODEC_ZLIB, read_container

ROOT = "/home/stud/zyuef/budgetgs-m0b"
P02D = os.path.join(ROOT, "runs/joint-dual-room375k-p02d")
V2 = os.path.join(ROOT, "runs/purel1-ft-room375k")

# shared per-attr levels + mu from P02D
p02d_ck = torch.load(os.path.join(P02D, "joint-room.pt"), map_location="cpu", weights_only=False)
mu_t = p02d_ck["mulaw_mu"]
ATTR_ORDER = ["means", "scales", "quats", "opacities", "sh0", "shn"]
hist = json.load(open(os.path.join(P02D, "joint-room-history.json")))
int_levels = hist[-1]["integer_levels"]
# attr-keyed -> container-keyed (opacities -> opacity)
level_map = {("opacity" if k == "opacities" else k): v for k, v in int_levels.items()}
mu_map = {("opacity" if ATTR_ORDER[j] == "opacities" else ATTR_ORDER[j]): float(mu_t[j])
          for j in range(6)}
print("level_map:", level_map)
print("mu_map:", {k: round(v, 4) for k, v in mu_map.items()})

def build_attributes(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = {}
    a["means"] = ck["means"].reshape(-1, 3).numpy().astype(np.float64)
    a["scales"] = ck["scales"].reshape(-1, 3).numpy().astype(np.float64)
    a["quats"] = ck["quats"].reshape(-1, 4).numpy().astype(np.float64)
    a["opacity"] = ck["opacity"].reshape(-1, 1).numpy().astype(np.float64)
    a["sh0"] = ck["sh0"].reshape(-1, 3).numpy().astype(np.float64)
    a["shn"] = ck["shn"].reshape(-1, 45).numpy().astype(np.float64)
    return a

groups = (("means", 3), ("scales", 3), ("quats", 4), ("opacity", 1), ("sh0", 3), ("shn", 45))
os.makedirs("/tmp/p1a", exist_ok=True)

for tag, d in [("P02D", P02D), ("V2FT", V2)]:
    attrs = build_attributes(os.path.join(d, "joint-room.pt"))
    path = "/tmp/p1a/%s.bgs" % tag
    n_bytes = write_mixed_container(path, attrs, groups, level_map,
                                    codec=CODEC_ZLIB, mu_map=mu_map)
    decoded = read_container(path, groups=split_shn_groups(groups))
    recon = dequant_mixed(decoded, level_map)
    errs = group_errors(attrs, recon)
    print("==== %s bytes=%.4fMB" % (tag, n_bytes / 1e6))
    for name, e in errs.items():
        print("  %-8s rmse=%.5f mse=%.6f" % (name, e["rmse"], e["mse"]))
