#!/usr/bin/env python
"""Re-run the real landing for a finished joint checkpoint (no retraining).

Loads joint-<scene>.pt + the last integer levels from the history json, and
re-runs train_joint.real_landing() with streamed per-view evaluation. Only
validation ground-truth images are kept in RAM so total memory stays far below
the cgroup limit (the factor=1 training already saved its ckpt; only the
landing evaluation was OOM-killed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_joint import ATTR_ORDER, real_landing

from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image


class _Levels:
    """Minimal schedule duck-type: only integer_targets() is used by real_landing."""

    def __init__(self, levels: dict):
        self._levels = levels

    def integer_targets(self):
        return dict(self._levels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="360_v2 scene directory")
    ap.add_argument("--scene-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--budget-mb", type=float, default=10.0)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-every", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir
    ckpt = torch.load(
        os.path.join(out_dir, f"joint-{args.scene_name}.pt"),
        map_location="cpu", weights_only=False,
    )
    cfg = ckpt.get("config", {})
    sh_degree = int(cfg.get("sh_degree", args.sh_degree))

    hist_path = os.path.join(out_dir, f"joint-{args.scene_name}-history.json")
    with open(hist_path, encoding="utf-8") as f:
        history = json.load(f)
    int_levels = dict(history[-1]["integer_levels"])
    mu_t = ckpt["mulaw_mu"]  # length-6 vector, ordered by ATTR_ORDER
    mu_per_attr = {name: mu_t[j] for j, name in enumerate(ATTR_ORDER)}

    device = args.device
    scene = load_360_scene(args.scene, factor=args.factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, args.test_every))
    width, height = scene.width, scene.height
    # only validation GT images, not the full training set
    images = {
        i: _load_image(scene.image_paths[i], device, size=(width, height))
        for i in val_idcs
    }
    print(
        f"scene={args.scene_name} val={len(val_idcs)} size={width}x{height} "
        f"levels={int_levels} mu="
        + ", ".join(f"{k}={float(v):.4f}" for k, v in mu_per_attr.items()),
        flush=True,
    )

    splats = {
        k: ckpt[k].to(device)
        for k in ["means", "scales", "quats", "opacity", "sh0", "shn"]
    }
    splats["opacities"] = splats.pop("opacity").reshape(-1)
    k_dim = (sh_degree + 1) ** 2 - 1
    splats["sh0"] = splats["sh0"].reshape(-1, 1, 3)
    splats["shn"] = splats["shn"].reshape(-1, k_dim, 3)

    schedule = _Levels(int_levels)
    antialiased = bool(cfg.get("antialiased", False))
    result = real_landing(
        splats, schedule, scene, args.scene_name, None, None,
        width, height, images, val_idcs, device, antialiased,
        sh_degree, out_dir, args.budget_mb, mu_per_attr=mu_per_attr,
    )
    result["history_tail"] = history[-1]
    with open(
        os.path.join(out_dir, f"joint-{args.scene_name}-result.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)
    print("LANDING DONE", flush=True)


if __name__ == "__main__":
    main()
