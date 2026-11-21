#!/usr/bin/env python
"""Unit test for prune_splats_relocate: mass conservation (transmittance) + shape."""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_colorfield import prune_splats_relocate, ATTR_ORDER  # noqa: E402


def main():
    torch.manual_seed(0)
    N = 2000
    n_keep = 1200
    splats = {
        "means": torch.nn.Parameter(torch.randn(N, 3) * 2.0),
        "scales": torch.nn.Parameter(torch.randn(N, 3)),
        "quats": torch.nn.Parameter(torch.randn(N, 4)),
        "opacities": torch.nn.Parameter(torch.randn(N) * 2.0),
    }
    optimizers = {name: torch.optim.Adam([splats[name]], lr=0.01) for name in ATTR_ORDER}
    importance = torch.sigmoid(splats["opacities"]).detach().clone()

    alpha_before = torch.sigmoid(splats["opacities"].detach())
    log_trans_before = torch.log1p(-alpha_before).sum().item()

    removed = prune_splats_relocate(splats, optimizers, n_keep, importance)

    alpha_after = torch.sigmoid(splats["opacities"].detach())
    log_trans_after = torch.log1p(-alpha_after).sum().item()

    print(f"removed={removed}  N_after={splats['means'].shape[0]} (expect {n_keep})")
    print(f"log_trans_before={log_trans_before:.6f}  log_trans_after={log_trans_after:.6f}")
    rel_err = abs(log_trans_after - log_trans_before) / max(abs(log_trans_before), 1e-9)
    print(f"rel_err={rel_err:.2e} (expect ~1e-6)")

    # sanity checks
    ok = (
        removed == N - n_keep
        and splats["means"].shape[0] == n_keep
        and all(splats[n].shape[0] == n_keep for n in ATTR_ORDER)
        and rel_err < 1e-3
    )
    print("TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
