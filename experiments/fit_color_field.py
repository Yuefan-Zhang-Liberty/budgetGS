#!/usr/bin/env python
"""Smoke: fit a shared hashgrid color field against FROZEN geometry, and compare
its rendering PSNR against the per-Gaussian SH color (the ref).

This isolates the single question P0 depends on: can a fixed-size shared color
field express the scene color well enough to replace the 45-dim shn block?
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch


def load_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    geo = {
        "means": ck["means"].to(device),
        "scales": ck["scales"].to(device),
        "quats": ck["quats"].to(device),
        "opacities": ck["opacity"].to(device),
    }
    sh = {"sh0": ck["sh0"].to(device), "shn": ck["shn"].to(device)}
    return geo, sh


def rasterize(geo, colors, c2w, Ks, w, h, idxs, device, antialiased, sh_degree):
    from gsplat.rendering import rasterization

    mode = "antialiased" if antialiased else "classic"
    return rasterization(
        means=geo["means"],
        quats=geo["quats"],
        scales=torch.exp(geo["scales"]),
        opacities=torch.sigmoid(geo["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv_ex(c2w[idxs]).inverse,
        Ks=Ks[idxs],
        width=w,
        height=h,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=mode,
        distributed=False,
        sh_degree=sh_degree,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--view_dependent", action="store_true")
    ap.add_argument("--n_levels", type=int, default=10)
    ap.add_argument("--log2_hashmap_size", type=int, default=17)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import _load_image
    from budgetgs.modules.color_field import ColorField

    scene = load_360_scene(args.scene_dir, factor=args.factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, 8))
    train_idcs = [i for i in range(n_views) if i not in val_idcs]
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    cam_centers = c2w[:, :3, 3]
    images = {i: _load_image(scene.image_paths[i], device, size=(width, height)) for i in range(n_views)}

    geo, sh = load_ckpt(args.ckpt, device)
    N = geo["means"].shape[0]
    print(f"geometry N={N} | views={n_views} val={len(val_idcs)} size={width}x{height}", flush=True)

    cf = ColorField(
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        view_dependent=args.view_dependent,
    ).to(device)
    print(f"color field params={cf.num_params():,} (hashgrid {cf.hashgrid_params():,}) "
          f"view_dependent={args.view_dependent}", flush=True)

    opt = torch.optim.Adam(cf.parameters(), lr=args.lr)
    bkgd = torch.ones(3, device=device)

    def render_cf(idxs):
        if args.view_dependent:
            rgb = cf(geo["means"], cam_centers[idxs[0]])
        else:
            rgb = cf(geo["means"])
        rgb = rgb.float()  # tcnn FullyFusedMLP outputs half on CUDA; gsplat needs fp32
        r, a, _info = rasterize(geo, rgb, c2w, Ks, width, height, idxs, device, False, None)
        return r + bkgd * (1.0 - a)

    # ---- ref (SH) PSNR once, no grad ----
    with torch.no_grad():
        sh_colors = torch.cat([sh["sh0"], sh["shn"]], dim=1)
        ref_psnrs = []
        for i in val_idcs:
            r, a, _ = rasterize(geo, sh_colors, c2w, Ks, width, height, [i], device, False, 3)
            r = r + bkgd * (1.0 - a)
            gt = images[i]
            mse = torch.mean((r - gt) ** 2)
            ref_psnrs.append(float(-10.0 * torch.log10(mse)))
        ref_psnr = float(np.mean(ref_psnrs))
    print(f"ref (SH, sh_degree=3) val PSNR = {ref_psnr:.3f} dB", flush=True)

    for step in range(1, args.steps + 1):
        i = random.choice(train_idcs)
        render = render_cf([i])
        gt = images[i]
        loss = torch.nn.functional.l1_loss(render, gt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.val_every == 0 or step == 1:
            with torch.no_grad():
                psnrs = []
                for j in val_idcs:
                    r = render_cf([j])
                    mse = torch.mean((r - images[j]) ** 2)
                    psnrs.append(float(-10.0 * torch.log10(mse)))
                val = float(np.mean(psnrs))
            print(f"step {step:>5d} loss {loss.item():.4f} | color-field val PSNR {val:.3f} "
                  f"(ref {ref_psnr:.3f}, gap {ref_psnr - val:.3f} dB)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
