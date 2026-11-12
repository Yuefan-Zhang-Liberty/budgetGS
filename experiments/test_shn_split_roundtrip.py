"""Verify the shn sub-group split end to end on the real 1975189 checkpoint:
write (range, mu-law) -> read (split groups) -> dequant (stitch back to 45ch),
assert byte-exact symbol round-trip and correct (n, 45) shape.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/budgetgs-m0b"))

from budgetgs.modules.serialize import CODEC_RANGE, read_container
from directionB_scene import (
    write_mixed_container,
    dequant_mixed,
    split_shn_groups,
)

CKPT = os.path.expanduser("~/budgetgs-m0b/runs/joint/joint-bonsai.pt")

def load_attributes():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    n = int(ckpt.get("n_gaussians", 0))
    if not n:
        n = len(ckpt["means"])
    attrs = {}
    for key, out_name in [("means", "means"), ("scales", "scales"),
                          ("quats", "quats"), ("opacity", "opacity"),
                          ("sh0", "sh0"), ("shn", "shn")]:
        t = ckpt[key].cpu().numpy().astype(np.float64)
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        attrs[out_name] = t
    return attrs, n

def main():
    attrs, n = load_attributes()
    print(f"n_gaussians={n} shn_shape={attrs['shn'].shape}", flush=True)
    groups = tuple((name, attrs[name].shape[1]) for name in
                   ["means", "scales", "quats", "opacity", "sh0", "shn"])
    level_map = {"means": 1047, "scales": 136, "quats": 136,
                 "opacity": 36, "sh0": 64, "shn": 10}
    mu_map = {"means": 6.0025, "scales": 0.0067, "quats": 6.0025,
              "opacity": 6.0025, "sh0": 0.0067, "shn": 6.0025}
    wgroups = split_shn_groups(groups)
    print("write groups:", wgroups, flush=True)

    path = "/tmp/test-split-roundtrip.bgs"
    t0 = time.time()
    n_bytes = write_mixed_container(path, attrs, groups, level_map,
                                   codec=CODEC_RANGE, mu_map=mu_map)
    print(f"write OK bytes={n_bytes} secs={time.time()-t0:.1f}", flush=True)

    t0 = time.time()
    decoded = read_container(path, groups=wgroups)
    print(f"read OK groups={len(decoded)} secs={time.time()-t0:.1f}", flush=True)
    for name in sorted(decoded):
        idx, st = decoded[name]
        print(f"  {name}: idx_len={len(idx)} ch={len(st.mins)}", flush=True)

    recon = dequant_mixed(decoded, level_map)
    print("dequant keys:", sorted(recon.keys()), flush=True)
    shn = recon.get("shn")
    assert shn is not None, "shn missing after dequant"
    assert shn.shape == (n, 45), f"shn shape {shn.shape} != ({n}, 45)"
    print(f"shn recon shape OK: {shn.shape}", flush=True)

    # byte-exact check: re-quantize the reconstructed values at same levels/mu
    # and compare indices with the decoded indices per group.
    from directionB_scene import quantize_group
    from budgetgs.modules.probability import CategoricalModel
    ok = True
    for name, (indices, scale_table) in decoded.items():
        base = name.split("#")[0] if name.startswith("shn#") else name
        lv = level_map[base]
        mu = mu_map.get(base, 0.0)
        _, idx2 = quantize_group(recon[base], lv, mu=mu) if not name.startswith("shn#") \
            else quantize_group(shn[:, int(name.split('#')[1])*3:int(name.split('#')[1])*3+3], lv, mu=mu)
        same = int(np.sum(np.asarray(indices) != idx2))
        print(f"  {name}: idx_mismatch={same}", flush=True)
        if same:
            ok = False
    print("ROUNDTRIP_OK" if ok else "ROUNDTRIP_FAIL", flush=True)

if __name__ == "__main__":
    main()
