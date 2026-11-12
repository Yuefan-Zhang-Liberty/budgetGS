#!/usr/bin/env python3
"""Direction A offline validation (no training).

On a fixed trained checkpoint, search the per-attribute quantization precision
(levels) allocation that maximizes landed PSNR under a HARD byte budget. This is
the offline / exhaustive counterpart of the planned train-time coordinate
descent (M0'). It reuses the exact production landing path proven in
directionB_scene.py (real zlib write/read -> dequant -> render -> score).

Experiments per scene:
  1. one-hot: bring ONE attribute group to 4096, leave all others at 23 ->
     which attribute carries image quality.
  2. means sweep: means 23..4096 with everything else at 23 -> how much
     position precision suffices.
  3. greedy marginal-gain allocation: start all groups at 23; every step raise
     the group whose next precision level gives the most PSNR gain per extra MB.
     Record the full RD path; report the best allocation with bytes <= budget
     (the Direction-A starting dict).

Outputs:
  runs/analysis/directionA-<scene>.json
  runs/analysis/directionA-<scene>.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

# Reuse the exact helpers proven in directionB_scene.py.
from budgetgs.modules.quantize import ATTRIBUTE_GROUPS
from budgetgs.utils.ckpt import load_attributes
from budgetgs.data.colmap import load_360_scene
from budgetgs.modules.trainer import _load_image
from budgetgs.evaluation.landing import (
    _metrics,
    _render_views,
    _splats_tensors,
)
from directionB_scene import (
    groups_with_shn,
    write_mixed_container,
    dequant_mixed,
    real_roundtrip,
)

DEVICE = "cuda"
VAL_EVERY = 8
SCAN_LEVELS = [23, 64, 128, 256, 512, 1024, 2048, 4096]
GREEDY_LEVELS = [23, 128, 512, 2048, 4096]
CURVE_CAP_MB = 60.0


def score_psnr(splats, scene, val_idcs, images, lpips_model, sh_degree):
    """Render all val views; return mean PSNR/SSIM/LPIPS."""
    rec_views = _render_views(splats, scene, val_idcs, DEVICE, sh_degree)
    tot = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for j, i in enumerate(val_idcs):
        m = _metrics(rec_views[j:j + 1], images[i], lpips_model, DEVICE)
        for k in tot:
            tot[k] += m[k]
    n = max(len(val_idcs), 1)
    return {k: round(v / n, 4) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser(prog="directionA_offline")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-name", default=None)
    ap.add_argument("--out-dir", default="runs/analysis")
    ap.add_argument("--budget-mb", type=float, default=10.0)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    args = ap.parse_args()

    scene_name = args.scene_name or os.path.basename(os.path.normpath(args.scene))
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "tmp_containers")
    os.makedirs(tmp_dir, exist_ok=True)

    # ---- load (identical to directionB) ------------------------------------
    attributes = load_attributes(args.ckpt)
    n_g = len(attributes["means"])
    raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        raw_ckpt = raw_ckpt["state_dict"]
    cfg = raw_ckpt.get("config", {}) if isinstance(raw_ckpt, dict) else {}
    sh_degree = int(cfg.get("sh_degree", 3)) if isinstance(cfg, dict) else 3

    groups = groups_with_shn(3)
    group_names = [g[0] for g in groups]
    print(f"[{scene_name}] n={n_g:,} sh_degree={sh_degree} groups={group_names}", flush=True)

    scene = load_360_scene(args.scene, factor=2)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, args.val_every))
    width, height = scene.width, scene.height
    print(f"scene views={n_views} val views={len(val_idcs)} size={width}x{height}", flush=True)
    images = {i: _load_image(scene.image_paths[i], DEVICE, size=(width, height)) for i in val_idcs}

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc), flush=True)

    torch.manual_seed(0)
    ref_splats = _splats_tensors(attributes, DEVICE, sh_degree)
    ref_metrics = score_psnr(ref_splats, scene, val_idcs, images, lpips_model, sh_degree)
    print("REF metrics:", ref_metrics, flush=True)
    del ref_splats
    torch.cuda.empty_cache()

    counter = {"n": 0}
    cache = {}
    base = None
    onehot = {}
    means_sweep = {}
    path = []
    jpath = os.path.join(args.out_dir, f"directionA-{scene_name}.json")

    def dump(note="", complete=False):
        """Incrementally write the json so a wall-time kill never loses results."""
        feasible = [p for p in path if p["mb"] <= args.budget_mb]
        best = max(feasible, key=lambda p: p["psnr"]) if feasible else (path[0] if path else None)
        result = {
            "scene": scene_name,
            "n_gaussians": n_g,
            "budget_mb": args.budget_mb,
            "group_order": group_names,
            "ref": ref_metrics,
            "all_23": base,
            "onehot": onehot,
            "means_sweep": means_sweep,
            "greedy_path": path,
            "best_within_budget": best,
            "complete": complete,
            "note": note,
        }
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"    [checkpoint] {note} -> {jpath} ({len(path)} greedy pts)", flush=True)

    def key_of(gl):
        return tuple(int(gl[g]) for g in group_names)

    def evaluate(gl, tag=""):
        k = key_of(gl)
        if k in cache:
            return cache[k]
        counter["n"] += 1
        t0 = time.time()
        n_bytes, recon, dt = real_roundtrip(
            attributes, groups, 0, tmp_dir, level_map=dict(gl), tag=f"a{counter['n']}"
        )
        splats = _splats_tensors(recon, DEVICE, sh_degree)
        m = score_psnr(splats, scene, val_idcs, images, lpips_model, sh_degree)
        del splats
        torch.cuda.empty_cache()
        rec = dict(levels=dict(gl), bytes=int(n_bytes), mb=round(n_bytes / 1e6, 3),
                   psnr=round(m["psnr"], 2), ssim=round(m["ssim"], 4),
                   lpips=round(m["lpips"], 4), sec=round(time.time() - t0, 1))
        cache[k] = rec
        print(f"    {tag:20s} {n_bytes/1e6:7.2f}MB  PSNR {m['psnr']:6.2f}  "
              f"SSIM {m['ssim']:.4f} ({rec['sec']}s)", flush=True)
        return rec

    # ---- 1. one-hot max contribution ---------------------------------------
    print("  [1] one-hot contribution (one group=4096, others=23)", flush=True)
    base = evaluate({g: 23 for g in group_names}, tag="all-23")
    for g in group_names:
        gl = {k: 23 for k in group_names}
        gl[g] = 4096
        onehot[g] = evaluate(gl, tag=f"onehot-{g}")
    dump("after-onehot")

    # ---- 2. means precision sweep ------------------------------------------
    print("  [2] means precision sweep (others=23)", flush=True)
    for lv in SCAN_LEVELS:
        gl = {k: 23 for k in group_names}
        gl["means"] = lv
        means_sweep[str(lv)] = evaluate(gl, tag=f"means-{lv}")
    dump("after-means-sweep")

    # ---- 3. greedy marginal-gain allocation --------------------------------
    print("  [3] greedy marginal-gain allocation (dPSNR/dMB)", flush=True)
    state = {g: 23 for g in group_names}
    path.append(cache[key_of(state)])  # == all-23 base, already evaluated
    step = 0
    stall = 0
    stall_limit = 2
    stall_gain = 0.05
    while True:
        if all(state[g] >= GREEDY_LEVELS[-1] for g in group_names):
            break
        prev = cache[key_of(state)]
        cands = []
        for g in group_names:
            idx = GREEDY_LEVELS.index(state[g])
            if idx == len(GREEDY_LEVELS) - 1:
                continue
            nxt = GREEDY_LEVELS[idx + 1]
            trial = dict(state)
            trial[g] = nxt
            r = evaluate(trial, tag=f"try-{g}-{nxt}")
            dmb = r["mb"] - prev["mb"]
            dp = r["psnr"] - prev["psnr"]
            gain = dp / dmb if dmb > 1e-6 else -1e9
            cands.append((gain, g, nxt, r))
        cands.sort(key=lambda x: -x[0])
        gain, g, nxt, r = cands[0]
        if r["mb"] > CURVE_CAP_MB:
            print(f"    reached curve cap {CURVE_CAP_MB} MB, stop", flush=True)
            break
        state[g] = nxt
        step += 1
        path.append(cache[key_of(state)])
        print(f"  step {step:2d}: raise {g:9s} -> {nxt:4d} | "
              f"{r['mb']:6.2f}MB  PSNR {r['psnr']:6.2f}  gain {gain:.3f} dB/MB", flush=True)
        dump(f"greedy-step-{step}")
        # early-stop: once well past the budget and PSNR gains have plateaued,
        # the remaining Pareto tail adds no decision-relevant info.
        if r["mb"] > 2 * args.budget_mb and gain < stall_gain:
            stall += 1
            if stall >= stall_limit:
                print(f"    saturated past {2*args.budget_mb:.0f}MB "
                      f"(gain<{stall_gain} dB/MB for {stall_limit} steps); stop", flush=True)
                break
        else:
            stall = 0

    feasible = [p for p in path if p["mb"] <= args.budget_mb]
    best = max(feasible, key=lambda p: p["psnr"]) if feasible else path[0]
    dump("complete", complete=True)
    print(f"\n  wrote {jpath}", flush=True)

    print(f"\n=== BEST <= {args.budget_mb} MB allocation ===", flush=True)
    for g in group_names:
        print(f"    {g:10s} levels = {best['levels'][g]}", flush=True)
    print(f"    {best['mb']} MB  PSNR {best['psnr']}  SSIM {best['ssim']}  "
          f"LPIPS {best['lpips']}   (ref {ref_metrics['psnr']})", flush=True)

    # ---- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].set_title(f"{scene_name} greedy allocation: landed PSNR vs MB")
    mb = [p["mb"] for p in path]
    pp = [p["psnr"] for p in path]
    ax[0].plot(mb, pp, "-o", color="tab:blue", label="greedy per-attribute")
    ms = [means_sweep[str(lv)]["mb"] for lv in SCAN_LEVELS]
    mps = [means_sweep[str(lv)]["psnr"] for lv in SCAN_LEVELS]
    ax[0].plot(ms, mps, "--s", color="tab:green", label="means-only (others=23)")
    ax[0].axhline(ref_metrics["psnr"], color="black", ls=":", label=f"ref {ref_metrics['psnr']:.2f}")
    ax[0].axhline(20, color="tab:orange", ls=":", label="advisor floor 20")
    ax[0].axvline(args.budget_mb, color="tab:red", ls=":", label=f"{args.budget_mb:.0f} MB budget")
    ax[0].plot(best["mb"], best["psnr"], "*", color="red", ms=20)
    ax[0].annotate(f"best {best['mb']}MB / {best['psnr']}dB",
                   (best["mb"], best["psnr"]), textcoords="offset points",
                   xytext=(8, -14), color="red", fontsize=9)
    ax[0].set_xlabel("container bytes (MB, zlib)")
    ax[0].set_ylabel("landed PSNR (dB)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].set_title("one-hot: single attribute at 4096 (others=23)")
    vals = [onehot[g]["psnr"] for g in group_names]
    sizes = [onehot[g]["mb"] for g in group_names]
    colors = ["tab:red" if g == "means" else "tab:blue" for g in group_names]
    bars = ax[1].bar(group_names, vals, color=colors, alpha=0.8)
    ax[1].axhline(base["psnr"], color="gray", ls=":", label=f"all-23 ({base['psnr']})")
    ax[1].axhline(20, color="tab:orange", ls=":")
    for b, v, sz in zip(bars, vals, sizes):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}\n{sz:.1f}MB",
                   ha="center", fontsize=8)
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    ppath = os.path.join(args.out_dir, f"directionA-{scene_name}.png")
    fig.savefig(ppath, dpi=130)
    plt.close(fig)
    print(f"  wrote {ppath}", flush=True)
    print("DIRECTION A DONE", flush=True)


if __name__ == "__main__":
    main()
