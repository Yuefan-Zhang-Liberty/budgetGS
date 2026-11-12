"""Landing-table evaluation: fit a checkpoint onto a byte budget, serialize it,
decode it back, re-render, and report epsilon / PSNR / SSIM / LPIPS / write time.

Week-1 plan Day 5: 3 scenes x 10 MB -> epsilon, PSNR, SSIM, LPIPS, write time,
with no retraining and no per-scene lambda search.
"""

from __future__ import annotations

import os
import tempfile
import time

from ..modules.quantize import dequantize_channel
from ..modules.serialize import CODEC_ZLIB, read_container, write_container
from ..utils.ckpt import load_attributes
from .fit import fit_levels

_BUDGET_BYTES = 1_000_000  # 1 MB = 10^6 bytes per the CLI contract


class _FixedLevels:
    """Duck-type stand-in for fit_levels() when the user pins a levels value,
    so we skip the (hours-long) bisection and serialize directly."""

    def __init__(self, levels: int):
        self.levels = int(levels)
        self.met_tolerance = None  # not evaluated when levels is pinned
        self.evals = 0


def _dequant_attributes(
    decoded: dict[str, tuple[list[int], object]],
    levels: int,
    original: dict[str, list],
) -> dict[str, object]:
    """Rebuild attribute values as (n, channels) numpy arrays (channel-major
    index stream is de-interleaved per the writer's layout)."""
    import numpy as np

    out: dict[str, object] = {}
    for name, (indices, scale_table) in decoded.items():
        channels = len(scale_table.mins)
        n = len(indices) // channels
        idx_arr = np.asarray(indices, dtype=np.float64)
        rows = np.empty((n, channels), dtype=np.float64)
        for c in range(channels):
            lo = scale_table.mins[c]
            hi = scale_table.maxs[c]
            inv = (hi - lo) / (levels - 1) if levels > 1 else 0.0
            # writer(quantize.py) 是 channel-major: flat=[ch0全部n个, ch1全部n个,...]
            # 必须取 idx[c*n:(c+1)*n]; 旧代码 idx[c::channels] 是逐行交错, 会把
            # channel 打乱 -> 解压出的高斯属性错乱 -> landed 质量崩成噪声(坑⑪后新增的根因修复)
            rows[:, c] = lo + idx_arr[c * n : (c + 1) * n] * inv
        out[name] = rows
    return out


def _splats_tensors(attributes: dict[str, object], device: str, sh_degree: int = 3):
    """Build the renderer's splat dict from attribute arrays; SH blocks are
    stored flat (sh0 (N,3), shn (N,K*3)) and reshaped to (N,1,3)/(N,K,3)."""
    import torch

    k = (sh_degree + 1) ** 2 - 1
    return {
        "means": torch.tensor(attributes["means"], dtype=torch.float32, device=device),
        "scales": torch.tensor(attributes["scales"], dtype=torch.float32, device=device),
        "quats": torch.tensor(attributes["quats"], dtype=torch.float32, device=device),
        "opacities": torch.tensor(attributes["opacity"], dtype=torch.float32, device=device).reshape(-1),
        "sh0": torch.tensor(attributes["sh0"], dtype=torch.float32, device=device).reshape(-1, 1, 3),
        "shn": torch.tensor(attributes["shn"], dtype=torch.float32, device=device).reshape(-1, k, 3),
    }


def _render_views(splats, scene, val_idcs, device, sh_degree=3, antialiased=False):
    """Render validation views over a white background; returns [len(val), H, W, 3].

    Renders one view at a time and IMMEDIATELY moves each result to CPU, so peak
    VRAM stays at a single view (~15GB for 1M gaussians) instead of accumulating
    all views on GPU (which OOMs a 48GB card at 25 views ~41GB).
    """
    import torch

    from ..modules.trainer import _compose_bkgd, _rasterize

    n_views = len(scene.image_paths)
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    out_views = []
    for i in val_idcs:
        render, alphas, _info = _rasterize(
            splats, c2w, Ks, width, height, [i], device, antialiased, sh_degree
        )
        composed = _compose_bkgd(render, alphas, torch.ones(3, device=device))
        out_views.append(composed.detach().cpu())  # 立即移回CPU, GPU只保留当前视图
    return torch.cat(out_views, dim=0)  # cat在CPU上进行, 不再占GPU显存


def _metrics(render, gt, lpips_model, device) -> dict:
    """PSNR / SSIM / LPIPS between a rendered and a ground-truth view.

    render 已由 _render_views 移回 CPU; 这里把 gt 也转到 CPU, 统一设备并省显存.
    LPIPS 计算时单独移回 GPU(模型在 GPU 上).
    """
    import torch
    from skimage.metrics import structural_similarity

    render = render.cpu()
    gt = gt.cpu()
    mse = torch.mean((render - gt) ** 2).item()
    import math

    psnr = float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))
    arr_r = render.squeeze(0).detach().cpu().numpy()
    arr_g = gt.squeeze(0).detach().cpu().numpy()
    ssim = float(structural_similarity(arr_r, arr_g, channel_axis=2, data_range=1.0))
    # 论文口径 SSIM: 11x11 Gaussian window, sigma=1.5 (3DGS / gsplat 官方 metrics.py 默认)
    try:
        ssim_std = float(
            structural_similarity(
                arr_r,
                arr_g,
                channel_axis=2,
                data_range=1.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                win_size=11,
            )
        )
    except Exception:
        ssim_std = float("nan")
    lpips = float("nan")
    if lpips_model is not None:
        a = render.permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
        b = gt.permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
        with torch.no_grad():
            lpips = float(lpips_model(a, b).item())
    return {
        "psnr": round(psnr, 3),
        "ssim": round(ssim, 4),
        "ssim_std": round(ssim_std, 4),
        "lpips": round(lpips, 4),
    }


def analyze_landing(
    ckpt_path: str,
    scene_dir: str,
    budget_mb: float = 10.0,
    tolerance: float = 0.01,
    val_every: int = 8,
    sh_degree: int = 3,
    device: str = "cuda",
    levels: int | None = None,
) -> dict:
    """Fit one checkpoint onto ``budget_mb`` MB and evaluate the landed container.

    If ``levels`` is given (non-None), the fit bisection is skipped entirely and
    that precision level is used directly -- this avoids the hours-long search
    when the landing level is already known (e.g. from a prior fit pass).
    """
    import torch

    from ..data.colmap import load_360_scene
    from ..modules.trainer import _load_image

    attributes = load_attributes(ckpt_path)
    import torch as _torch

    _ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _cfg = _ckpt.get("config", {}) if isinstance(_ckpt, dict) else {}
    sh_degree = int(_cfg.get("sh_degree", sh_degree))
    budget = int(budget_mb * _BUDGET_BYTES)
    if levels is not None:
        fit = _FixedLevels(levels)
    else:
        fit = fit_levels(attributes, budget, tolerance=tolerance)

    with tempfile.NamedTemporaryFile(suffix=".bgs", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        t0 = time.time()
        actual = write_container(tmp_path, attributes, levels=fit.levels, codec=CODEC_ZLIB)
        t_write = time.time() - t0
        decoded = read_container(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    recon = _dequant_attributes(decoded, fit.levels, attributes)
    scene = load_360_scene(scene_dir, factor=2)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, val_every))
    width, height = scene.width, scene.height
    images = {
        i: _load_image(scene.image_paths[i], device, size=(width, height))
        for i in val_idcs
    }

    torch.manual_seed(0)
    try:
        lpips_model = None
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            lpips_model = None
        ref_splats = _splats_tensors(attributes, device, sh_degree)
        rec_splats = _splats_tensors(recon, device, sh_degree)
        ref_views = _render_views(ref_splats, scene, val_idcs, device, sh_degree)
        rec_views = _render_views(rec_splats, scene, val_idcs, device, sh_degree)
        metric_names = ["psnr", "ssim", "ssim_std", "lpips"]
        ref_tot = {k: 0.0 for k in metric_names}
        rec_tot = {k: 0.0 for k in metric_names}
        for j, i in enumerate(val_idcs):
            gt = images[i]
            rm = _metrics(rec_views[j : j + 1], gt, lpips_model, device)
            fm = _metrics(ref_views[j : j + 1], gt, lpips_model, device)
            for k in metric_names:
                rec_tot[k] += rm[k]
                ref_tot[k] += fm[k]
        n = max(len(val_idcs), 1)
        ref_avg = {k: round(ref_tot[k] / n, 4) for k in metric_names}
        rec_avg = {k: round(rec_tot[k] / n, 4) for k in metric_names}
    finally:
        del images, ref_splats, rec_splats
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "ckpt": os.path.basename(ckpt_path),
        "n_gaussians": len(attributes["means"]),
        "budget_mb": budget_mb,
        "levels": fit.levels,
        "actual_bytes": actual,
        "epsilon_bytes": abs(actual - budget),
        "epsilon_pct": round(abs(actual - budget) / budget * 100.0, 4),
        "tolerance_met": fit.met_tolerance,
        "fit_evals": fit.evals,
        "write_seconds": round(t_write, 2),
        "ref_psnr": ref_avg["psnr"],
        "ref_ssim": ref_avg["ssim"],
        "ref_lpips": ref_avg["lpips"],
        "landed_psnr": rec_avg["psnr"],
        "landed_ssim": rec_avg["ssim"],
        "ref_ssim_std": ref_avg["ssim_std"],
        "landed_ssim_std": rec_avg["ssim_std"],
        "landed_lpips": rec_avg["lpips"],
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="budgetgs landing", description=__doc__)
    parser.add_argument("--ckpt", required=True, help="trained checkpoint (.pt)")
    parser.add_argument("--scene", required=True, help="COLMAP scene directory")
    parser.add_argument("--budget-mb", type=float, default=10.0)
    parser.add_argument("--val-every", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="skip fit bisection and serialize at this fixed precision level "
        "(use a known landing level to save hours)",
    )
    args = parser.parse_args(argv)

    result = analyze_landing(
        args.ckpt, args.scene, budget_mb=args.budget_mb, levels=args.levels
    )
    print(
        f"{result['ckpt']}: N={result['n_gaussians']} budget={args.budget_mb}MB "
        f"levels={result['levels']} actual={result['actual_bytes']}B "
        f"eps={result['epsilon_bytes']}B ({result['epsilon_pct']}%) "
        f"tol={result['tolerance_met']} write={result['write_seconds']}s"
    )
    print(
        f"  PSNR {result['landed_psnr']} (ref {result['ref_psnr']}) | "
        f"SSIM {result['landed_ssim']} (ref {result['ref_ssim']}) | "
        f"LPIPS {result['landed_lpips']} (ref {result['ref_lpips']})"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
