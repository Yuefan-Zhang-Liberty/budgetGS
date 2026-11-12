#!/usr/bin/env python3
"""render_compare.py - P1-3: render GT / unquantized ref / quantized-landed
comparison tiles for a trained joint checkpoint.

Reads runs/<out_dir>/joint-<scene>.pt + joint-<scene>-result.json, re-renders
the val views through the SAME path as real_landing (quantized splats via
quantized_splats(noise=False) == discrete landing; the range-coded write/read
round-trip is lossless so this is byte-identical to the landed model), and
writes per-view 3-tile PNGs plus one full grid image, together with
per-view PSNR/SSIM/LPIPS.

Usage:
    python render_compare.py --scene-name room --out-dir runs/joint \
        --factor 2 --sh-degree 3 --max-views 6 --grid-views 6
"""

import argparse
import json
import os

import numpy as np
import torch

ATTR_ORDER = ["means", "scales", "quats", "opacities", "sh0", "shn"]
CONTAINER_NAME = {
    "means": "means", "scales": "scales", "quats": "quats",
    "opacities": "opacity", "sh0": "sh0", "shn": "shn",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-name", required=True)
    ap.add_argument("--scene-dir", default=None,
                    help="defaults to $HOME/storage/user/zyuef/datasets/360_v2/<scene>")
    ap.add_argument("--out-dir", default="runs/joint")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--max-views", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import _load_image
    from budgetgs.evaluation.landing import _render_views, _metrics
    from train_joint import quantized_splats, _splats_from_live

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_name = args.scene_name
    scene_dir = args.scene_dir or os.path.join(
        os.path.expanduser("~"), "storage/user/zyuef/datasets/360_v2", scene_name
    )
    ckpt_path = os.path.join(args.out_dir, f"joint-{scene_name}.pt")
    result_path = os.path.join(args.out_dir, f"joint-{scene_name}-result.json")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"ckpt not found: {ckpt_path}")
    if not os.path.exists(result_path):
        raise SystemExit(f"result not found: {result_path}")

    # ---- scene + images (same protocol as train_joint) ---------------------
    scene = load_360_scene(scene_dir, factor=args.factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, 8))          # test_every=8 like training
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    images = {i: _load_image(scene.image_paths[i], device, size=(width, height))
              for i in val_idcs}
    print(f"scene={scene_name} views={n_views} val={len(val_idcs)} size={width}x{height}")

    # ---- ckpt + result ------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    splats = {
        "means": torch.nn.Parameter(ckpt["means"].to(device)),
        "scales": torch.nn.Parameter(ckpt["scales"].to(device)),
        "quats": torch.nn.Parameter(ckpt["quats"].to(device)),
        "opacities": torch.nn.Parameter(ckpt["opacity"].to(device)),
        "sh0": torch.nn.Parameter(ckpt["sh0"].to(device)),
        "shn": torch.nn.Parameter(ckpt["shn"].to(device)),
    }
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    int_levels = result["integer_levels"]      # keys = trainer attr names (opacities)
    mu_raw = result.get("mulaw_mu", {})        # keys = container names (opacity)
    levels_map = {
        name: int(int_levels[name]) for name in ATTR_ORDER
    }
    mu_map = {
        name: torch.as_tensor(mu_raw.get(CONTAINER_NAME[name], 0.0),
                              dtype=torch.float32, device=device)
        for name in ATTR_ORDER
    }
    print("levels:", levels_map)
    print("mu    :", mu_map)

    # ---- render ref (unquantized) + landed (discrete quantized) ------------
    antialiased = False
    ref_splats = _splats_from_live(splats, device, args.sh_degree)
    landed_splats = quantized_splats(splats, levels_map, mu_per_attr=mu_map)
    ref_views = _render_views(ref_splats, scene, val_idcs, device,
                              args.sh_degree, antialiased)
    landed_views = _render_views(landed_splats, scene, val_idcs, device,
                                 args.sh_degree, antialiased)
    print("rendered ref + landed views")

    # ---- metrics + tiles ----------------------------------------------------
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc))

    from PIL import Image, ImageDraw, ImageFont

    out_img_dir = os.path.join(args.out_dir, f"render-compare-{scene_name}")
    os.makedirs(out_img_dir, exist_ok=True)
    font = ImageFont.load_default()

    per_view = []
    for j, i in enumerate(val_idcs):
        gt = images[i]
        m = _metrics(landed_views[j:j + 1], gt, lpips_model, device)
        per_view.append({"view": int(i), **m})

    # tiles for the first `max_views` val views
    tile_paths = []
    for j, i in enumerate(val_idcs[:args.max_views]):
        gt = images[i]
        def _to_np(t):
            x = t.detach().float().cpu()
            if x.dim() == 4:            # (1,H,W,3)
                x = x[0]
            if x.shape[-1] != 3:
                x = x.permute(1, 2, 0).contiguous()
            return (x.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        a = _to_np(gt)
        b = _to_np(ref_views[j])
        c = _to_np(landed_views[j])
        pad = np.full((a.shape[0], 14, 3), 255, dtype=np.uint8)
        row = np.concatenate([a, pad, b, pad, c], axis=1)      # H x (3W+2pad) x 3
        im = Image.fromarray(row)
        draw = ImageDraw.Draw(im)
        draw.text((8, 2), f"GT view{i}", font=font, fill=(0, 0, 0))
        draw.text((a.shape[1] + 14 + 8, 2), "ref full-precision", font=font, fill=(0, 0, 0))
        draw.text((2 * (a.shape[1] + 14) + 8, 2),
                  f"landed {result.get('actual_mb', 0):.2f}MB PSNR {m['psnr']:.2f}",
                  font=font, fill=(0, 0, 0))
        tp = os.path.join(out_img_dir, f"view-{i:03d}.png")
        im.save(tp)
        tile_paths.append(tp)

    # ---- full grid: one row per selected view ------------------------------
    sel = list(val_idcs[:min(6, len(val_idcs))])
    rows = []
    for j, i in enumerate(sel):
        gt = images[i]
        a = _to_np(gt); b = _to_np(ref_views[j]); c = _to_np(landed_views[j])
        pad = np.full((a.shape[0], 14, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([a, pad, b, pad, c], axis=1))
    grid = np.concatenate(rows, axis=0)
    grid_path = os.path.join(args.out_dir, f"render-compare-{scene_name}.png")
    Image.fromarray(grid).save(grid_path)

    # ---- json ---------------------------------------------------------------
    out = {
        "scene": scene_name,
        "n_val": len(val_idcs),
        "integer_levels": int_levels,
        "mulaw_mu": mu_raw,
        "landed_actual_mb": result.get("actual_mb"),
        "per_view": per_view,
        "mean": {
            k: round(sum(p[k] for p in per_view) / len(per_view), 4)
            for k in ("psnr", "ssim", "lpips")
        },
        "tile_paths": tile_paths,
        "grid_path": grid_path,
    }
    with open(os.path.join(args.out_dir, f"render-compare-{scene_name}.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("grid saved:", grid_path)
    print("mean:", out["mean"])
    print("DONE")


if __name__ == "__main__":
    main()
