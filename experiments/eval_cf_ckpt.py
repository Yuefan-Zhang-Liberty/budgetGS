#!/usr/bin/env python
"""Evaluate the P0 color-field ckpt with REAL 8-bit hashgrid / 16-bit MLP
quantization, closing the accounting gap (ledger said 8-bit but render used
float). Loads joint-room.pt, renders 39 val views twice:

  - ref   : quantized geometry + FLOAT color field
  - landed: quantized geometry + 8-bit hashgrid + 16-bit MLP (dequantized back)

and reports the true color loss from the 8-bit grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# run from ~/budgetgs-m0b/experiments so train_colorfield is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from budgetgs.data.colmap import load_360_scene  # noqa: E402
from budgetgs.modules.color_field import ColorField  # noqa: E402
from budgetgs.modules.trainer import _load_image  # noqa: E402
from budgetgs.evaluation.landing import _metrics  # noqa: E402
from train_colorfield import (  # noqa: E402
    ATTR_ORDER,
    CONTAINER_NAME,
    _compose_bkgd,
    _rasterize_cf,
    quantize_attr,
)


def quantize_8bit(t):
    lo = t.min()
    hi = t.max()
    span = (hi - lo).clamp_min(1e-12)
    idx = torch.round((t - lo) / span * 255.0).clamp(0, 255).to(torch.uint8)
    return idx, lo, hi


def dequant_8bit(idx, lo, hi):
    span = (hi - lo).clamp_min(1e-12)
    return lo + idx.float() / 255.0 * span


def quantize_16bit(t):
    lo = t.min()
    hi = t.max()
    span = (hi - lo).clamp_min(1e-12)
    idx = torch.round((t - lo) / span * 65535.0).clamp(0, 65535).to(torch.int32)
    return idx, lo, hi


def dequant_16bit(idx, lo, hi):
    span = (hi - lo).clamp_min(1e-12)
    return lo + idx.float() / 65535.0 * span


def render_metrics(geo, color_field, scene, val_idcs, device, antialiased, lpips_model):
    n_views = len(scene.image_paths)
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    images = {i: _load_image(scene.image_paths[i], device, size=(width, height)) for i in val_idcs}
    bkgd = torch.ones(3, device=device)
    totals = {"psnr": 0.0, "ssim": 0.0, "ssim_std": 0.0, "lpips": 0.0}
    for i in val_idcs:
        render, alphas, _ = _rasterize_cf(geo, color_field, c2w, Ks, width, height, [i], device, antialiased)
        composed = _compose_bkgd(render, alphas, bkgd).detach().cpu()
        m = _metrics(composed, images[i], lpips_model, device)
        for k in totals:
            totals[k] += m[k]
        del render, alphas, composed
    n = len(val_idcs)
    return {k: round(v / n, 4) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser(prog="eval_cf_ckpt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--antialiased", action="store_true")
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
    # quantize geometry (same levels / mu as the trained landing)
    levels_map = {name: result["integer_levels"][name] for name in ATTR_ORDER}
    mu_map = {
        name: torch.tensor(result["mulaw_mu"][CONTAINER_NAME[name]], dtype=torch.float32, device=device)
        for name in ATTR_ORDER
    }
    geo_q = {}
    for name in ATTR_ORDER:
        geo_q[name] = quantize_attr(geo[name], levels_map[name], noise=False, mu=mu_map[name])

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

    # hashgrid byte check (8-bit) + MLP byte check (16-bit)
    hash_params = int(color_field.recolor.params.numel())
    mlp_params = int(color_field.mlp_head.params.numel())
    hash_bytes = hash_params * 1
    mlp_bytes = mlp_params * 2
    print(f"hashgrid params={hash_params:,} (8-bit = {hash_bytes/1e6:.2f} MB)")
    print(f"mlp params={mlp_params:,} (16-bit = {mlp_bytes/1e6:.3f} MB)")
    print(f"color_field_bytes = {hash_bytes + mlp_bytes:,}")

    # ---- ref: float color field ----
    print("rendering ref (float color field) ...", flush=True)
    ref = render_metrics(geo_q, color_field, scene, val_idcs, device, args.antialiased, lpips_model)
    print("ref:", ref, flush=True)

    # ---- landed: 8-bit hashgrid + 16-bit MLP ----
    print("quantizing color field (8-bit hash + 16-bit mlp) ...", flush=True)
    idx_h, lo_h, hi_h = quantize_8bit(color_field.recolor.params.detach())
    color_field.recolor.params.data.copy_(dequant_8bit(idx_h, lo_h, hi_h))
    idx_m, lo_m, hi_m = quantize_16bit(color_field.mlp_head.params.detach())
    color_field.mlp_head.params.data.copy_(dequant_16bit(idx_m, lo_m, hi_m))

    print("rendering landed (8-bit color field) ...", flush=True)
    landed = render_metrics(geo_q, color_field, scene, val_idcs, device, args.antialiased, lpips_model)
    print("landed:", landed, flush=True)

    delta = {k: round(landed[k] - ref[k], 4) for k in ref}
    print("=== 8-bit color-field cost (landed - ref) ===")
    print(delta)

    out = {
        "ckpt": args.ckpt,
        "hashgrid_params": hash_params,
        "mlp_params": mlp_params,
        "color_field_bytes": hash_bytes + mlp_bytes,
        "ref": ref,
        "landed_8bit": landed,
        "delta": delta,
    }
    out_path = os.path.join(os.path.dirname(args.result_json), "cf-8bit-eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
