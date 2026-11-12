#!/usr/bin/env python3
"""mem_probe.py - 峰值显存探针：375k 高斯 x factor=1 全分辨率训练一步
（量化噪声路径 + 渲染 + L1 反传 + optimizer step），打印峰值显存与可用卡判断。"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-name", default="room")
    ap.add_argument("--scene-dir", default=None)
    ap.add_argument("--out-dir", default="runs/joint-dual-room375k")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    scene_name = args.scene_name
    scene_dir = args.scene_dir or os.path.join(
        os.path.expanduser("~"), "storage/user/zyuef/datasets/360_v2", scene_name)
    ckpt_path = os.path.join(args.out_dir, f"joint-{scene_name}.pt")
    result_path = os.path.join(args.out_dir, f"joint-{scene_name}-result.json")
    for p in (ckpt_path, result_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import _rasterize, _compose_bkgd, _load_image
    from train_joint import quantized_splats, ATTR_ORDER, CONTAINER_NAME

    scene = load_360_scene(scene_dir, factor=args.factor)
    n_views = len(scene.image_paths)
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32,
                       device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32,
                      device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    print(f"scene={scene_name} factor={args.factor} views={n_views} size={width}x{height}")

    ckpt = torch.load(ckpt_path, map_location=device)
    splats = {
        "means": torch.nn.Parameter(ckpt["means"].to(device)),
        "scales": torch.nn.Parameter(ckpt["scales"].to(device)),
        "quats": torch.nn.Parameter(ckpt["quats"].to(device)),
        "opacities": torch.nn.Parameter(ckpt["opacity"].to(device)),
        "sh0": torch.nn.Parameter(ckpt["sh0"].to(device)),
        "shn": torch.nn.Parameter(ckpt["shn"].to(device)),
    }
    n_g = splats["means"].shape[0]
    print(f"N={n_g}")

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    il = result["integer_levels"]
    mr = result["mulaw_mu"]
    levels_map = {name: int(il[name]) for name in ATTR_ORDER}
    mu_map = {name: torch.as_tensor(mr.get(CONTAINER_NAME[name], 0.0),
                                    dtype=torch.float32, device=device)
              for name in ATTR_ORDER}
    print("levels:", levels_map)
    print("mu    :", {n: round(float(mu_map[n]), 4) for n in ATTR_ORDER})

    gt = _load_image(scene.image_paths[0], device, size=(width, height))
    bkgd = torch.ones(3, device=device)
    antialiased = False

    opt = torch.optim.SGD(list(splats.values()), lr=1e-3)
    torch.cuda.reset_peak_memory_stats()
    n_steps = 3
    for si in range(n_steps):
        vi = si % n_views
        gt = _load_image(scene.image_paths[vi], device, size=(width, height))
        view = quantized_splats(splats, levels_map, noise=True, mu_per_attr=mu_map)
        render, alphas, info = _rasterize(view, c2w, Ks, width, height, [vi],
                                          device, antialiased, args.sh_degree)
        render = _compose_bkgd(render, alphas, bkgd)
        loss = torch.nn.functional.l1_loss(render, gt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        try:
            n_ren = int(info.num_rendered.item()) if hasattr(info, "num_rendered") else "?"
            print(f"step{si}: view={vi} rendered_px={n_ren}")
        except Exception:
            print(f"step{si}: view={vi} rendered=ok")

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU_NAME={torch.cuda.get_device_name(0)}")
    print(f"GPU_TOTAL_GB={total:.2f}")
    print(f"PEAK_GB={peak:.2f}")
    print(f"STEPS={n_steps}")
    print(f"FIT_48G={peak < 46.0}")
    print(f"FIT_96G={peak < 92.0}")
    print("MEM_PROBE_DONE")


if __name__ == "__main__":
    main()
