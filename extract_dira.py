#!/usr/bin/env python3
import json
base="/home/stud/zyuef/budgetgs-m0b/runs/analysis"
for scene in ["room","bonsai"]:
    d=json.load(open(f"{base}/directionA-{scene}.json"))
    ref=d["ref"]
    print("="*24, scene, "="*24)
    print(f"complete={d['complete']}  n_gauss={d['n_gaussians']}  ref={ref['psnr']:.2f}/{ref['ssim']:.4f}/{ref['lpips']:.4f}")
    b=d["best_within_budget"]
    print(f"  BEST <= {d['budget_mb']}MB:")
    for g in d["group_order"]:
        print(f"      {g:9s} levels = {b['levels'][g]}")
    print(f"      -> {b['mb']} MB | PSNR {b['psnr']} | SSIM {b['ssim']} | LPIPS {b['lpips']}")
    print("  greedy path (MB -> PSNR), every point:")
    for p in d["greedy_path"]:
        print(f"      {p['mb']:6.2f} MB -> {p['psnr']:6.2f}")
    print()
