"""Pinpoint where shn 45ch hangs: time quantize / model / searchsorted / encode
separately, print progress every 1M symbols inside the encode loop."""
import sys
import time
import torch
import numpy as np

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from budgetgs.modules.quantize import quantize_group
from budgetgs.modules.probability import CategoricalModel
from budgetgs.modules.serialize import RangeEncoder

ckpt = torch.load(
    "runs/joint/joint-bonsai.pt", map_location="cpu", weights_only=False
)
names = ["means", "scales", "quats", "opacity", "sh0", "shn"]
muv = {n: float(v) for n, v in zip(names, ckpt["mulaw_mu"].tolist())}
x = ckpt["shn"].reshape(ckpt["shn"].shape[0], -1).cpu().numpy().astype(float)
ch, lv, mu = 45, 10, muv["shn"]
attr = {"shn": x[:, :ch]}
groups = (("shn", ch),)

t0 = time.time()
scale_table, indices = quantize_group(attr["shn"], lv, mu=mu)
print(f"[quantize] secs={time.time()-t0:.1f} n_sym={len(indices)}", flush=True)

t0 = time.time()
model = CategoricalModel.from_symbols(indices, lv)
print(f"[model]    secs={time.time()-t0:.1f} n_distinct={model.n_distinct}", flush=True)

symbols, cum, total = model.cumulative_freq()
print(f"[cum]      symbols={symbols} total={total}", flush=True)

t0 = time.time()
i_arr = np.searchsorted(np.asarray(symbols, dtype=np.int64), indices)
print(f"[search]   secs={time.time()-t0:.1f} i_arr dtype={i_arr.dtype}", flush=True)

enc = RangeEncoder()
t0 = time.time()
last = t0
for k, i in enumerate(i_arr):
    enc.encode(cum[i], cum[i + 1], total)
    if k and k % 1000000 == 0:
        now = time.time()
        print(f"[encode] {k/1e6:.0f}M syms dt={now-last:.1f}s "
              f"cum={now-t0:.1f}s low={enc.low} range={enc.range} "
              f"cache_size={enc.cache_size} outlen={len(enc.out)}", flush=True)
        last = now
print(f"[encode]   done secs={time.time()-t0:.1f} outlen={len(enc.out)}", flush=True)
