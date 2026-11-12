"""Time shn range-encoding at growing channel counts to classify hang (quadratic)
vs plain slowness (linear). Usage: python test_shn_scale.py"""
import sys
import time
import torch

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from directionB_scene import write_mixed_container
from budgetgs.modules.serialize import CODEC_RANGE

ckpt = torch.load(
    "runs/joint/joint-bonsai.pt", map_location="cpu", weights_only=False
)
names = ["means", "scales", "quats", "opacity", "sh0", "shn"]
mu_map = {n: float(v) for n, v in zip(names, ckpt["mulaw_mu"].tolist())}
x = ckpt["shn"]  # [375000, 15, 3]
x = x.reshape(x.shape[0], -1)  # [375000, 45]
attr_np = x.cpu().numpy().astype(float)
lv = 10

for ch in (3, 9, 15, 30, 45):
    attr = {"shn": attr_np[:, :ch]}
    groups = (("shn", ch),)
    t0 = time.time()
    nb = write_mixed_container(
        f"runs/joint/test_shn{ch}.bgs", attr, groups, {"shn": lv},
        codec=CODEC_RANGE, mu_map={"shn": mu_map["shn"]},
    )
    dt = time.time() - t0
    nsym = attr_np.shape[0] * ch
    print(f"shn {ch:2d}ch n_sym={nsym:>9d} bytes={nb:>9d} secs={dt:6.1f} "
          f"us/sym={dt*1e6/nsym:8.1f}", flush=True)
