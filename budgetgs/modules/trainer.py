"""`budgetgs train`: minimal gsplat 3DGS trainer.

Packs the gsplat ``simple_trainer`` loop into a CLI (plan Day 3) and stores
checkpoints under the exact attribute keys the byte ledger consumes
(means/scales/quats/opacity/sh0/shn), so K1 and fit can run off raw
checkpoints without retraining.

Design notes
------------
- Densification uses ``gsplat.strategy.DefaultStrategy`` (original 3DGS
  heuristics); N grows over the run, which is precisely the signal K1
  regresses bytes on.
- Checkpoints every ``--ckpt-every`` steps hold the raw splat parameters plus
  step/seed/config, in a flat dict whose keys the ledger's alias table maps.
- ``runs.jsonl`` appends one JSON object per run: scene, seed, commit hash,
  config, and the per-checkpoint (step, N, val PSNR, ckpt path) curve.
- Image loading uses PIL (pulled in by matplotlib, which the env pins).

The module imports torch/gsplat lazily so ``budgetgs --help`` and the
pure-python test suite keep working without the conda environment.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys

_SH_C0 = 0.28209479177387814
_SSIM_LAMBDA = 0.2


def rgb_to_sh(rgb):
    """RGB in [0, 1] to SH0 coefficients (3DGS convention)."""
    return (rgb - 0.5) / _SH_C0


def _init_scales(points, k: int = 4, chunk: int = 512) -> object:
    """Log-scales uniform across points, sized by the mean distance to the k
    nearest neighbours on a 20k subsample (chunked cdist to bound memory)."""
    import torch

    n = points.shape[0]
    if n <= k:
        return torch.full((n, 3), math.log(0.01), device=points.device)
    m = min(n, 20000)
    perm = torch.randperm(n, device=points.device)[:m]
    sub = points[perm]
    dists = []
    for i in range(0, m, chunk):
        j = min(i + chunk, m)
        d = torch.cdist(sub[i:j], sub)
        d[torch.arange(j - i), torch.arange(i, j)] = float("inf")
        top = torch.topk(d, k, largest=False).values  # [rows, k]
        dists.append(top.mean(dim=-1))
    med = torch.median(torch.cat(dists))
    return torch.full((n, 3), math.log(max(med.item(), 1e-6)), device=points.device)


def create_splats(scene, device: str = "cuda", sh_degree: int = 3, init_num_pts: int = 100_000):
    """Build the splat ``ParameterDict`` and one Adam optimizer per attribute,
    mirroring gsplat's ``create_splats_with_optimizers`` defaults."""
    import torch

    torch.manual_seed(0)  # overridden by the caller's seed before this runs
    pts = torch.tensor(scene.points, dtype=torch.float32, device=device)
    rgbs = torch.tensor(scene.points_rgb, dtype=torch.float32, device=device) / 255.0
    if pts.shape[0] < 1024:
        extent = 2.0 * scene.scene_scale
        pts = (torch.rand(init_num_pts, 3, device=device) * 2 - 1) * extent
        rgbs = torch.rand(init_num_pts, 3, device=device)
    n = pts.shape[0]

    scales = _init_scales(pts)
    quats = torch.rand(n, 4, device=device)
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    colors = torch.zeros(n, (sh_degree + 1) ** 2, 3, device=device)
    colors[:, 0] = rgb_to_sh(rgbs)

    params = {
        "means": (pts, 1.6e-4 * scene.scene_scale),
        "scales": (scales, 5e-3),
        "quats": (quats, 1e-3),
        "opacity": (opacities, 5e-2),
        "sh0": (colors[:, :1], 2.5e-3),
        "shn": (colors[:, 1:], 2.5e-3 / 20),
    }
    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.clone()) for name, (value, _lr) in params.items()}
    )
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr}], eps=1e-15, betas=(0.9, 0.999)
        )
        for name, (_value, lr) in params.items()
    }
    return splats, optimizers


def _load_image(path: str, device: str):
    """Load one JPEG/PNG as a float tensor in [0, 1] with shape [1, H, W, 3]."""
    import numpy as np
    import torch
    from PIL import Image

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _rasterize(splats, camtoworlds, Ks, width, height, view_idcs, device, antialiased):
    """Rasterize one or more views; returns ``(render_colors, render_alphas, info)``
    with ``render_colors`` in [C, H, W, 3]."""
    import torch
    from gsplat.rendering import RasterizeMode, rasterization

    colors = torch.cat([splats["sh0"], splats["shn"]], dim=1)
    mode = RasterizeMode.ANTIALIASED if antialiased else RasterizeMode.CLASSIC
    render_colors, render_alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv_ex(camtoworlds[view_idcs]).inverse,
        Ks=Ks[view_idcs],
        width=width,
        height=height,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=mode,
        distributed=False,
    )
    return render_colors, render_alphas, info


def _compose_bkgd(colors, alphas, bkgd) -> object:
    """Over alpha on a solid background, out-of-place (no grads to bkgd)."""
    return colors + bkgd * (1.0 - alphas)


def _psnr(render: object, gt: object) -> float:
    import torch

    mse = torch.mean((render - gt) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def _evaluate(splats, camtoworlds, Ks, width, height, images, val_idcs, device, antialiased):
    """Mean val PSNR over the held-out views."""
    import torch

    render, alphas, _info = _rasterize(
        splats, camtoworlds, Ks, width, height, val_idcs, device, antialiased
    )
    bkgd = torch.ones(3, device=device)
    render = _compose_bkgd(render, alphas, bkgd)
    psnrs = []
    with torch.no_grad():
        for idx, i in enumerate(val_idcs):
            gt = images[i]
            psnrs.append(_psnr(render[idx : idx + 1], gt))
    return sum(psnrs) / len(psnrs)


def _commit_hash(repo_root: str | None) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def train(
    scene_dir: str,
    steps: int = 30_000,
    ckpt_every: int = 1_000,
    out: str = "runs",
    seed: int = 0,
    factor: int = 2,
    sh_degree: int = 3,
    device: str = "cuda",
    antialiased: bool = False,
    test_every: int = 8,
) -> str:
    """Train one scene; returns the path of the run directory."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    from budgetgs.data.colmap import load_360_scene

    scene = load_360_scene(scene_dir, factor=factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, test_every))
    train_idcs = [i for i in range(n_views) if i not in val_idcs]

    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device)
    width, height = scene.width, scene.height
    images = {i: _load_image(scene.image_paths[i], device) for i in range(n_views)}

    splats, optimizers = create_splats(scene, device=device, sh_degree=sh_degree)
    from gsplat.strategy import DefaultStrategy
    from gsplat.losses import l1_loss, ssim_loss

    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)

    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(steps, 1))
    )

    scene_name = os.path.basename(os.path.normpath(scene_dir))
    run_dir = os.path.join(out, f"{scene_name}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    commit = _commit_hash(repo_root)

    ckpt_rows: list[dict] = []
    final_psnr = float("nan")
    bkgd = torch.ones(3, device=device)

    for step in range(1, steps + 1):
        i = random.choice(train_idcs)
        gt = images[i]
        render, alphas, info = _rasterize(
            splats, c2w, Ks, width, height, [i], device, antialiased
        )
        render = _compose_bkgd(render, alphas, bkgd)

        strategy.step_pre_backward(
            params=splats, optimizers=optimizers, state=strategy_state, step=step, info=info
        )
        l1 = l1_loss(render, gt).mean()
        ssim = ssim_loss(render.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = (1.0 - _SSIM_LAMBDA) * l1 + _SSIM_LAMBDA * (1.0 - ssim)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=strategy_state,
            step=step, info=info, packed=True,
        )
        means_scheduler.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:>6d} | loss {loss.item():.4f} | N {splats['means'].shape[0]}", flush=True)

        if step % ckpt_every == 0:
            val_psnr = _evaluate(
                splats, c2w, Ks, width, height, images, val_idcs, device, antialiased
            )
            final_psnr = val_psnr
            ckpt_path = os.path.join(run_dir, f"ckpt-{step:07d}.pt")
            ckpt = {
                "means": splats["means"].detach().cpu(),
                "scales": splats["scales"].detach().cpu(),
                "quats": splats["quats"].detach().cpu(),
                "opacity": splats["opacity"].detach().cpu(),
                "sh0": splats["sh0"].detach().cpu(),
                "shn": splats["shn"].detach().cpu(),
                "step": step,
                "seed": seed,
                "config": {
                    "steps": steps, "ckpt_every": ckpt_every, "factor": factor,
                    "sh_degree": sh_degree, "antialiased": antialiased,
                },
                "n_gaussians": int(splats["means"].shape[0]),
            }
            torch.save(ckpt, ckpt_path)
            ckpt_rows.append(
                {
                    "step": step,
                    "n_gaussians": int(splats["means"].shape[0]),
                    "val_psnr": round(float(val_psnr), 4),
                    "ckpt": os.path.relpath(ckpt_path, out),
                }
            )
            print(
                f"step {step:>6d} | ckpt {os.path.relpath(ckpt_path, out)} | "
                f"N {ckpt_rows[-1]['n_gaussians']} | val PSNR {val_psnr:.3f}",
                flush=True,
            )

    run = {
        "scene": scene_name,
        "seed": seed,
        "commit": commit,
        "config": {
            "steps": steps, "ckpt_every": ckpt_every, "factor": factor,
            "sh_degree": sh_degree, "antialiased": antialiased,
        },
        "ckpt_dir": run_dir,
        "final_val_psnr": round(float(final_psnr), 4),
        "checkpoints": ckpt_rows,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")
    print(
        f"done: {scene_name} seed {seed} | final val PSNR {final_psnr:.3f} | "
        f"{len(ckpt_rows)} checkpoints -> {run_dir}",
        flush=True,
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """Parse ``budgetgs train`` arguments and run one training job."""
    import argparse

    parser = argparse.ArgumentParser(prog="budgetgs train", description=__doc__)
    parser.add_argument("--scene", required=True, help="path to a 360_v2/COLMAP scene directory")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--ckpt-every", type=int, default=1_000)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factor", type=int, default=2, help="image downscale factor (images_{factor})")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--antialiased", action="store_true", help="antialiased rasterization")
    args = parser.parse_args(argv)
    train(
        args.scene, steps=args.steps, ckpt_every=args.ckpt_every, out=args.out,
        seed=args.seed, factor=args.factor, sh_degree=args.sh_degree,
        device=args.device, antialiased=args.antialiased,
    )


if __name__ == "__main__":
    main()
