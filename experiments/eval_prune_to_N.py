#!/usr/bin/env python
"""Ablation (a): prune the relocate ckpt down to topk's N, then land, to
separate "more Gaussians" from "better-trained Gaussians".

Loads joint-room.pt (relocate, N=757,599), hard-prunes (topk by opacity) to
--n-target (topk's 668,984), quantizes geometry with the ckpt's own levels/mu,
8-bits the color field, and renders landed metrics. Compare to topk's landed
25.490 @ N=668,984.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from budgetgs.data.colmap import load_360_scene  # noqa: E402
from budgetgs.modules.color_field import ColorField  # noqa: E402
from train_colorfield import ATTR_ORDER, CONTAINER_NAME, quantize_attr  # noqa: E402
from eval_cf_ckpt import (  # noqa: E402
    quantize_8bit,
    dequant_8bit,
    quantize_16bit,
    dequant_16bit,
    render_metrics,
)


def main():
    ap = argparse.ArgumentParser(prog="eval_prune_to_N")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-target", type=int, required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    ckpt = torch.load(args.ckpt, map_location="cpu")
    result = json.load(open(args.result_json, encoding="utf-8"))
    cfg = ckpt["color_field_cfg"]

    scene = load_360_scene(args.scene, factor=args.factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, 8))

    geo = {
        "means": ckpt["means"].to(device),
        "scales": ckpt["scales"].to(device),
        "quats": ckpt["quats"].to(device),
        "opacities": ckpt["opacity"].to(device),
    }

    # topk hard-prune to n_target (same N as topk run)
    n = geo["means"].shape[0]
    if args.n_target < n:
        importance = torch.sigmoid(geo["opacities"]).detach()
        keep = torch.argsort(importance, descending=True)[: args.n_target]
        keep, _ = torch.sort(keep)
        for name in ATTR_ORDER:
            geo[name] = geo[name][keep]
    print(f"pruned to N={geo['means'].shape[0]} (target {args.n_target})", flush=True)

    # quantize geometry with ckpt's own levels / mu
    levels_map = {name: result["integer_levels"][name] for name in ATTR_ORDER}
    mu_map = {
        name: torch.tensor(
            result["mulaw_mu"][CONTAINER_NAME[name]], dtype=torch.float32, device=device
        )
        for name in ATTR_ORDER
    }
    geo_q = {name: quantize_attr(geo[name], levels_map[name], noise=False, mu=mu_map[name])
             for name in ATTR_ORDER}

    color_field = ColorField(
        n_levels=cfg["n_levels"],
        log2_hashmap_size=cfg["log2_hashmap_size"],
        view_dependent=cfg.get("view_dependent", False),
    ).to(device)
    color_field.load_state_dict({k: v.to(device) for k, v in ckpt["color_field"].items()})

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc))

    # 8-bit hashgrid + 16-bit MLP (same landed color accounting)
    idx_h, lo_h, hi_h = quantize_8bit(color_field.recolor.params.detach())
    color_field.recolor.params.data.copy_(dequant_8bit(idx_h, lo_h, hi_h))
    idx_m, lo_m, hi_m = quantize_16bit(color_field.mlp_head.params.detach())
    color_field.mlp_head.params.data.copy_(dequant_16bit(idx_m, lo_m, hi_m))

    landed = render_metrics(geo_q, color_field, scene, val_idcs, device, False, lpips_model)
    print("landed (pruned to N):", landed, flush=True)

    out = {
        "source_ckpt": args.ckpt,
        "n_after_prune": int(geo["means"].shape[0]),
        "landed": landed,
    }
    out_path = os.path.join(os.path.dirname(args.result_json), "prune-to-N-eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
