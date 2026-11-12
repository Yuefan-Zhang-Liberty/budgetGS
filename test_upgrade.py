"""CPU-only unit test for the two upgrades:
  1. learned mu-law non-uniform quantization (torch + numpy parity, round trip)
  2. range coder (arithmetic coding) container write/read byte-exact, and a
     zlib-vs-range size comparison on the same data.
Run from the repo root with the `main` conda env: python test_upgrade.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

REPO = "/home/stud/zyuef/budgetgs-m0b"
sys.path.insert(0, REPO)

import train_joint as tj
from budgetgs.modules.quantize import (
    quantize_group,
    mulaw_compand,
    mulaw_expand,
)
from budgetgs.modules.serialize import (
    CODEC_RANGE,
    CODEC_ZLIB,
    read_container,
)
from directionB_scene import write_mixed_container, dequant_mixed

print("== T1: numpy mu-law round trip ==", flush=True)
u = np.linspace(0.0, 1.0, 100001)
for mu in [0.0, 0.5, 5.0, 50.0]:
    back = mulaw_expand(mulaw_compand(u, mu), mu)
    print(f"  mu={mu:6.1f} max|roundtrip|={np.max(np.abs(back - u)):.2e}", flush=True)

print("== T2: torch vs numpy mu-law parity ==", flush=True)
ut = torch.linspace(0.0, 1.0, 1001)
for mu in [0.0, 0.5, 5.0, 50.0]:
    vt = tj._mulaw_compand(ut, torch.tensor(mu)).numpy()
    vn = mulaw_compand(ut.numpy(), mu)
    print(f"  mu={mu:6.1f} max|torch-np|={np.max(np.abs(vt - vn)):.2e}", flush=True)

print("== T3: build synthetic central-heavy attributes ==", flush=True)
rng = np.random.default_rng(0)
N = 5000
attributes = {
    "means": rng.normal(0.0, 1.0, (N, 3)),
    "scales": rng.normal(-5.0, 1.0, (N, 3)),
    "quats": rng.normal(0.0, 1.0, (N, 4)),
    "opacity": rng.normal(0.0, 1.0, (N, 1)),
    "sh0": rng.normal(0.0, 1.0, (N, 3)),
    "shn": rng.normal(0.0, 1.0, (N, 3)),
}
groups = tuple((k, v.shape[1]) for k, v in attributes.items())
level_map = {"means": 1024, "scales": 128, "quats": 128,
             "opacity": 32, "sh0": 64, "shn": 10}
mu_map = {"means": 2.0, "scales": 1.0, "quats": 0.5,
          "opacity": 1.0, "sh0": 0.5, "shn": 0.0}

print("== T4: range container write/read, byte-exact + dequant ==", flush=True)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "t.bgs")
    nb = write_mixed_container(
        p, attributes, groups, level_map, codec=CODEC_RANGE, mu_map=mu_map
    )
    decoded = read_container(p, groups=groups)
    all_exact = True
    for name, (indices, st) in decoded.items():
        _, exp_idx = quantize_group(
            attributes[name], level_map[name], mu=mu_map[name]
        )
        exact = indices == exp_idx
        all_exact = all_exact and exact
        print(f"  {name:8s} idx_exact={exact} n={len(indices)} "
              f"mu_read={getattr(st, 'mu', 0.0)}", flush=True)
    recon = dequant_mixed(decoded, level_map)
    for name in attributes:
        err = np.max(np.abs(recon[name] - attributes[name]))
        print(f"  {name:8s} max dequant err={err:.3e}", flush=True)
    print(f"  ALL INDICES BYTE-EXACT: {all_exact}", flush=True)

print("== T5: zlib vs range size on identical data/mu ==", flush=True)
with tempfile.TemporaryDirectory() as td:
    pz = os.path.join(td, "z.bgs")
    pr = os.path.join(td, "r.bgs")
    nz = write_mixed_container(
        pz, attributes, groups, level_map, codec=CODEC_ZLIB, mu_map=mu_map
    )
    nr = write_mixed_container(
        pr, attributes, groups, level_map, codec=CODEC_RANGE, mu_map=mu_map
    )
    print(f"  zlib={nz} B  range={nr} B  range/zlib={nr / nz:.3f}", flush=True)

print("ALL TESTS DONE", flush=True)
