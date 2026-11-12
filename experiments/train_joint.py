"""BudgetGS joint training: compress *during* training instead of after.

The offline Direction-A/B sweeps proved that post-hoc uniform quantization is
doomed at a 10 MB budget: the position (means) axis is the bottleneck, and with
N fixed by standard densification there is no way to land <=10 MB at quality.
This script implements the advisor's 10/29 direction -- "put the quantizer
inside the training loop" -- with four pieces:

  1. Two-phase schedule.
       Phase 1 (steps 0..warmup, default 15k): standard 3DGS training with
       gsplat DefaultStrategy densification, NO quantization -- let the
       geometry grow and converge.
       Phase 2 (warmup..steps): quantization turned on and gradually annealed,
       densification stopped, active pruning drives N down to a target.

  2. Differentiable per-attribute quantization. Each of the six attribute
     groups (means/scales/quats/opacities/sh0/shn) is quantized per-channel
     against its own min/max with a straight-through estimator; the renderer
     sees quantized values, raw parameters keep receiving gradients.

  3. Learnable per-attribute levels. One raw scalar per group (6 parameters)
     maps through a sigmoid to a soft level in [L_min, L_max]; the global
     quantization progress g in [0,1] log-anneals from 4096 (lossless) down
     to the learned target. A soft rate penalty only pushes back once the
     surrogate byte count exceeds the budget, so levels stay as high as the
     budget allows (no hard fixed boundary that destroys gradients).

  4. MCMC-style budget control. From Phase 2, every --prune-every steps the
     least opaque Gaussians are pruned along a linear N schedule to
     --target-n; light opacity / scale regularizers (MCMC Eq.10 style) and
     optional anisotropic position noise (SGLD, means only) are included.

After training it performs a REAL landing: per-group integer levels, a real
zlib container written and read back through the production serializer, then
re-rendered and scored (PSNR/SSIM/LPIPS). The acceptance target is
<=10 MB AND PSNR > 20.

Usage (see train-joint.sbatch):
  python train_joint.py --scene <dir> --scene-name room --target-n 400000 \
      --out-dir runs/joint --budget-mb 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

# Gaussian attribute groups in the order they are optimized / serialized.
# Trainer ParameterDict keys vs. container keys differ only for opacity.
ATTR_ORDER = ["means", "scales", "quats", "opacities", "sh0", "shn"]
# Container name for each trainer key (opacities -> opacity).
CONTAINER_NAME = {
    "means": "means",
    "scales": "scales",
    "quats": "quats",
    "opacities": "opacity",
    "sh0": "sh0",
    "shn": "shn",
}
# Channel count per group at sh_degree 3 (shn = 15 rest orders * 3).
def channel_counts(sh_degree: int) -> dict:
    k = (sh_degree + 1) ** 2 - 1
    return {
        "means": 3,
        "scales": 3,
        "quats": 4,
        "opacities": 1,
        "sh0": 3,
        "shn": k * 3,
    }


# ---------------------------------------------------------------------------
# differentiable per-channel quantization
# ---------------------------------------------------------------------------
def _mulaw_compand(u, mu):
    """mu-law compression on normalized ``u`` in [0,1]; ``mu`` >= 0 -> v in
    [0,1]. The dense central region is expanded so uniform bins in the v
    domain become finer bins around the center in the u domain. ``mu = 0`` is
    the identity. Fully differentiable in both ``u`` and ``mu``."""
    import torch

    t = 2.0 * u - 1.0
    log_mu = torch.log1p(mu)
    t_c = torch.sign(t) * torch.log1p(mu * torch.abs(t)) / log_mu
    v = (t_c + 1.0) * 0.5
    return torch.where(mu > 1e-6, v, u)


def _mulaw_expand(v, mu):
    """Inverse of :func:`_mulaw_compand`. Fully differentiable."""
    import torch

    t = 2.0 * v - 1.0
    log_mu = torch.log1p(mu)
    t_e = torch.sign(t) * torch.expm1(torch.abs(t) * log_mu) / mu
    u = (t_e + 1.0) * 0.5
    return torch.where(mu > 1e-6, u, v)


def _mulaw_compand_np(u, mu):
    """Numpy mu-law compression, matching :func:`_mulaw_compand` (``mu`` is a
    fixed float here). ``mu <= 0`` returns ``u`` unchanged."""
    import numpy as np

    if mu <= 1e-6:
        return u
    t = 2.0 * u - 1.0
    log_mu = np.log1p(mu)
    t_c = np.sign(t) * np.log1p(mu * np.abs(t)) / log_mu
    return (t_c + 1.0) * 0.5


def quantize_attr(x, levels, noise=False, mu=0.0):
    """Per-channel quantize/dequantize with a straight-through estimator,
    optionally non-uniform via learned mu-law companding.

    ``levels`` is a scalar (float, differentiable) and ``mu`` is the companding
    strength (>= 0, differentiable; 0 = uniform). Values are normalized to
    [0,1], mu-law compressed, quantized into uniform bins in the compressed
    domain, then inverse-companded, so the real reconstruction bins are finer
    around the data's dense center. Two forward paths:

    * ``noise=False`` (real landing / probing): forward uses the *discrete*
      value; backward flows through a continuous proxy so ``x``, ``levels`` and
      ``mu`` all receive gradients (QAT-style learnable quantization).
    * ``noise=True`` (Phase-2 training): forward uses additive uniform noise in
      the code domain, an unbiased dequant whose variance matches the true
      rounding error. Because the noise is sampled independently of the level,
      *levels* and *mu* receive a real quality signal (low levels / weak mu ->
      large noise -> higher distortion) instead of a zero gradient.
    """
    import torch

    orig_shape = x.shape
    if x.dim() == 1:
        x = x.reshape(-1, 1)
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    lo = x.amin(dim=0, keepdim=True).detach()
    hi = x.amax(dim=0, keepdim=True).detach()
    degenerate = (hi - lo) < 1e-12
    hi = torch.where(degenerate, lo + 1e-6, hi)
    rng = hi - lo
    l_minus_1 = levels - 1.0
    qmin = torch.zeros((), device=x.device, dtype=x.dtype)
    qmax = l_minus_1
    # normalize to [0,1], then non-linear compand (mu=0 -> identity)
    u = (x - lo) / rng
    v = _mulaw_compand(u, mu)
    if noise:
        # additive-uniform-noise in the code domain: unbiased dequant, variance
        # matches the true rounding error at the current (companded) levels.
        qn = v * l_minus_1 + (torch.rand_like(v) - 0.5)
        qn = torch.clamp(qn, qmin, qmax)
        vq = qn / l_minus_1
        uq = _mulaw_expand(vq, mu)
        xq = lo + uq * rng
        return xq.reshape(orig_shape)
    # discrete forward: round in the companded domain
    q = torch.clamp(torch.round(v * l_minus_1), qmin, qmax)
    vq = q / l_minus_1
    uq = _mulaw_expand(vq, mu)
    xq = lo + uq * rng
    # continuous proxy (same formula, no rounding) so gradients flow to x,
    # levels, and mu. The forward stays DISCRETE; backward flows through this
    # proxy (the (xq - xq_c).detach() order is load-bearing -- swapping it
    # collapses the gradient through round and freezes every attribute).
    vc = torch.clamp(v * l_minus_1, qmin, qmax) / l_minus_1
    uc = _mulaw_expand(vc, mu)
    xq_c = lo + uc * rng
    out = xq_c + (xq - xq_c).detach()
    return out.reshape(orig_shape)


def quantized_splats(splats, levels_per_attr, noise=False, mu_per_attr=None):
    """Return a dict of quantized attribute tensors with the same shapes."""
    out = {}
    for name in ATTR_ORDER:
        mu = mu_per_attr[name] if mu_per_attr is not None else 0.0
        out[name] = quantize_attr(
            splats[name], levels_per_attr[name], noise=noise, mu=mu
        )
    return out


# ---------------------------------------------------------------------------
# learnable per-attribute levels
# ---------------------------------------------------------------------------
class LevelSchedule:
    """Six learnable target levels via sigmoid; log-annealed from 4096.

    init_targets: initial per-group target level (a sensible Direction-A prior);
    min_levels: per-attribute floor (position must never fall below the level
    at which quantization destroys geometry).
    """

    L_MAX = 4096.0
    L_START = 4096.0

    def __init__(self, device, init_targets, min_levels):
        import torch

        self.min_levels = torch.tensor(
            list(min_levels), device=device, dtype=torch.float32
        )
        raws = []
        for t, mn in zip(init_targets, min_levels):
            t = min(max(t, mn + 1.0), self.L_MAX - 1.0)
            p = (t - mn) / (self.L_MAX - mn)
            raws.append(math.log(p / (1.0 - p)))
        self.raw = torch.nn.Parameter(
            torch.tensor(raws, device=device, dtype=torch.float32)
        )

    def target_levels(self):
        import torch

        s = torch.sigmoid(self.raw)
        return self.min_levels + (self.L_MAX - self.min_levels) * s

    def levels_at(self, g):
        """Log-space interpolation L_START -> target at global progress g."""
        import torch

        tgt = self.target_levels()
        log_l = (1.0 - g) * math.log(self.L_START) + g * torch.log(tgt)
        return torch.exp(log_l)

    def integer_targets(self):
        """Final integer levels for the real landing (dict keyed by trainer)."""
        tgt = self.target_levels().detach().cpu().tolist()
        mn = self.min_levels.detach().cpu().tolist()
        return {
            name: max(int(round(tgt[i])), int(mn[i]))
            for i, name in enumerate(ATTR_ORDER)
        }


def surrogate_bytes(levels, n, ch):
    """Upper-bound byte estimate: sum over groups of N*C*log2(L)/8.

    This is the uniform-distribution entropy (an upper bound on real zlib
    output); it is used only relatively, for gradients, and is calibrated
    against a real zlib probe during Phase 2.
    """
    import torch

    total = None
    for i, name in enumerate(ATTR_ORDER):
        bits = n * ch[name] * torch.log2(levels[i])
        total = bits if total is None else total + bits
    return total / 8.0


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------
def prune_splats(splats, optimizers, n_keep, importance):
    """Prune every attribute to the top-``n_keep`` rows by importance.

    Rebuilds the per-attribute Adam optimizers (state is reset; it re-warms in
    a few steps). All groups share the first (Gaussian) axis, so one index
    vector applies to every tensor.
    """
    import torch

    n = splats["means"].shape[0]
    if n_keep >= n:
        return 0
    idx = torch.topk(importance, int(n_keep), largest=True).indices
    idx, _ = torch.sort(idx)
    for name in ATTR_ORDER:
        p = splats[name]
        new_p = torch.nn.Parameter(p.data[idx])
        splats[name] = new_p
        old_lr = optimizers[name].param_groups[0]["lr"]
        optimizers[name] = torch.optim.Adam(
            [{"params": new_p, "lr": old_lr}], eps=1e-15, betas=(0.9, 0.999)
        )
    return int(n - n_keep)


# ---------------------------------------------------------------------------
# real zlib probe (used to calibrate the surrogate and to log real bytes)
# ---------------------------------------------------------------------------
def probe_real_bytes(splats, ch, device, mu_per_attr=None):
    """Range-coder byte estimate at the CURRENT learned integer levels.

    Each group is mu-law companded (if learned), quantized exactly like the
    serializer, and priced with the same Laplace-smoothed categorical entropy
    the range coder pays for (``code_bits``), plus the transmitted frequency
    table overhead. Returns (real_bytes, integer_levels).
    """
    import numpy as np

    from budgetgs.modules.probability import CategoricalModel

    int_levels = _LAST_INTEGER_LEVELS
    return_bytes = 0
    for name in ATTR_ORDER:
        x = splats[name].detach()
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        arr = x.cpu().numpy().astype(np.float64)
        mu = (
            float(mu_per_attr[name].detach())
            if mu_per_attr is not None
            else 0.0
        )
        c = arr.shape[1]
        L = int_levels[name]
        per_channel = []
        for cc in range(c):
            col = arr[:, cc]
            lo = float(col.min())
            hi = float(col.max())
            if hi - lo < 1e-12:
                hi = lo + 1e-6
            u = (col - lo) / (hi - lo)
            v = _mulaw_compand_np(u, mu)
            idx = np.rint(v * (L - 1)).astype(np.int64)
            np.clip(idx, 0, L - 1, out=idx)
            per_channel.append(idx)
        flat = np.concatenate(per_channel).tolist()
        model = CategoricalModel.from_symbols(flat, L)
        coded = max(1, math.ceil(model.code_bits() / 8.0))
        # n_distinct field (4) + coded_len field (4) + 8 bytes per table entry
        return_bytes += coded + 4 + 4 + 8 * model.n_distinct
    return return_bytes, int_levels


_LAST_INTEGER_LEVELS = {name: 4096 for name in ATTR_ORDER}


# ---------------------------------------------------------------------------
# main training
# ---------------------------------------------------------------------------
def train_joint(
    scene_dir,
    scene_name,
    out_dir,
    steps=30_000,
    warmup=15_000,
    target_n=400_000,
    budget_mb=10.0,
    prune_every=250,
    sh_degree=3,
    seed=0,
    factor=2,
    device="cuda",
    antialiased=False,
    test_every=8,
    position_noise=True,
    lambda_mode="dual",
    lambda_eta=1e-4,
    n_hold_step=None,
):
    import numpy as np
    import torch

    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import (
        _compose_bkgd,
        _load_image,
        _rasterize,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ch = channel_counts(sh_degree)

    # ---- data -------------------------------------------------------------
    scene = load_360_scene(scene_dir, factor=factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, test_every))
    train_idcs = [i for i in range(n_views) if i not in val_idcs]
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    images = {i: _load_image(scene.image_paths[i], device, size=(width, height)) for i in range(n_views)}
    print(f"scene={scene_name} views={n_views} val={len(val_idcs)} size={width}x{height}", flush=True)

    # ---- splats / optimizers / strategy -----------------------------------
    from budgetgs.modules.trainer import create_splats
    splats, optimizers = create_splats(scene, device=device, sh_degree=sh_degree)

    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(verbose=False)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)

    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(steps, 1))
    )

    # learnable levels, initialized from a Direction-A prior (position-rich)
    init_targets = {
        "means": 1024.0,
        "scales": 128.0,
        "quats": 128.0,
        "opacities": 32.0,
        "sh0": 64.0,
        "shn": 10.0,
    }
    # no hard floors: with noise-path training the level that destroys
    # geometry hurts loss directly (means stay high on their own), and the
    # marginal probe re-allocates budget every 1000 steps. Floors are only
    # there to keep logits finite / quantization meaningful (>= 2 levels).
    min_levels = {
        "means": 4.0,
        "scales": 4.0,
        "quats": 4.0,
        "opacities": 4.0,
        "sh0": 2.0,
        "shn": 2.0,
    }
    schedule = LevelSchedule(
        device, [init_targets[n] for n in ATTR_ORDER],
        [min_levels[n] for n in ATTR_ORDER],
    )
    # SGD, not Adam: Adam's normalization turns the tiny rate gradient into a
    # constant +/-lr step and slams the levels to the floor. SGD keeps the
    # level updates proportional to the true (small) gradients.
    level_opt = torch.optim.SGD([{"params": schedule.raw, "lr": 1e-3}], momentum=0.0)

    # Learnable mu-law companding strength per attribute (6 parameters).
    # softplus keeps mu >= 0. raw starts at -1.0 (mu ~ 0.31), a small but
    # already-non-uniform value: the mu-law L1 gradient at the old init -3.0
    # (mu ~ 0.049, near-uniform) measured only ~1e-5..1e-4, so plain SGD at
    # lr=1e-3 moved mu_raw by ~1e-7/step -> mu never left 0.0486 in the
    # 1974260 run (grad-check confirmed).
    # AdamW lr=1e-2 DID learn in 1975189 but slammed every mu_raw to the
    # clamp edges (+6.0/-5.0): Adam's sign-level updates have no "gradient
    # zero => stop" fixed point, so 15000 Phase2 steps pushed means/quats/
    # opacity/shn to mu~403 (pathological) and scales/sh0 to mu~0.0067.
    # SGD lr=1e0 keeps updates proportional to the true small gradients
    # (~1e-4/step), reaches the optimum range, and naturally settles when
    # the gradient decays or flips -- no sign-level floor-slam.
    mu_raw = torch.nn.Parameter(
        torch.full((len(ATTR_ORDER),), -1.0, device=device, dtype=torch.float32)
    )
    mu_opt = torch.optim.SGD([{"params": mu_raw, "lr": 1e0}], momentum=0.0)

    def mu_values():
        # clamp raw to keep mu in [~1e-6, ~403]; avoids runaway expm1 scale
        # if Adam ever drives raw to extremes
        raw_c = mu_raw.clamp(-5.0, 6.0)
        return torch.nn.functional.softplus(raw_c) + 1e-6

    def mu_map():
        muv = mu_values()
        return {name: muv[j] for j, name in enumerate(ATTR_ORDER)}

    budget_bytes = budget_mb * 1_000_000.0
    bkgd = torch.ones(3, device=device)

    # Lagrangian lambda for the rate term: loss = D + lam * R/B.
    # Dual ascent by default (lambda_mode="dual"): lambda += eta * (R/B - 1),
    # driven by the surplus/deficit of the (calibrated) byte estimate vs budget;
    # the real probe re-anchors lambda once per probe interval. Multiplicative
    # heuristic kept as lambda_mode="sched" for ablation. R/B normalization
    # keeps lam interpretable (lam ~ fraction of the distortion loss
    # contributed by the rate term).
    lam = torch.tensor(1e-2, device=device)
    # P0-2b: count consecutive real-byte probes that sit comfortably under
    # budget; used to trigger a fast multiplicative release of lambda.
    _relax_streak = 0
    # P0-2c: once the budget has been comfortably met for two probes, freeze
    # N pruning and anneal the opacity regulariser, so late-stage training
    # becomes pure L1 quality refinement instead of continued compression.
    _budget_satisfied = False
    reg_o_scale = 1.0

    # surrogate calibration: real = k * surrogate (filled on first Phase-2 probe)
    calib = None
    history = []
    n_pruned_total = 0

    def render_view(splats_view, i):
        render, alphas, info = _rasterize(
            splats_view, c2w, Ks, width, height, [i], device, antialiased, sh_degree
        )
        render = _compose_bkgd(render, alphas, bkgd)
        return render, info

    for step in range(1, steps + 1):
        phase2 = step > warmup
        i = random.choice(train_idcs)
        gt = images[i]

        if phase2:
            # global quantization progress 0..1 over Phase 2
            g = (step - warmup) / max(steps - warmup, 1)
            levels = schedule.levels_at(g)
            levels_map = {name: levels[j] for j, name in enumerate(ATTR_ORDER)}
            # noise-path QAT: additive uniform noise whose variance exactly
            # matches the true rounding error at the current levels; levels
            # finally receive a real quality gradient (dxq/ds = -eps/s^2).
            # P0-2d: keep additive-noise QAT while the budget is still being
            # pursued, but once it is satisfied switch to the true discrete
            # round path so late-stage fine-tuning optimises exactly what the
            # landing/evaluation renders (no noise-path vs round-path drift).
            view = quantized_splats(
                splats, levels_map, noise=not _budget_satisfied, mu_per_attr=mu_map()
            )
            render, info = render_view(view, i)
        else:
            g = 0.0
            render, info = render_view(splats, i)

        # Phase 1 densification pre-backward hook (must run before backward).
        if not phase2:
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=step, info=info,
            )

        # ---- distortion loss ----
        loss = torch.nn.functional.l1_loss(render, gt)

        # ---- regularizers (Phase 2 only), MCMC Eq.10 style ----
        if phase2:
            reg_o = 0.01 * reg_o_scale * torch.sigmoid(splats["opacities"]).mean()
            reg_s = 0.01 * torch.exp(splats["scales"]).mean()
            loss = loss + reg_o + reg_s

        # ---- Lagrangian rate term: loss = L1 + regs + lam * R/B ----
        # Continuous (differentiable) rate pressure that never vanishes: the
        # surrogate bytes are normalized by the budget so lam controls the
        # distortion-vs-rate trade-off directly. Adapted every 250 steps.
        if phase2:
            n_now = splats["means"].shape[0]
            rate = surrogate_bytes(levels, n_now, ch)
            if calib is not None:
                rate_real = rate * calib
            else:
                rate_real = rate
            loss = loss + lam * (rate_real / budget_bytes)

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if phase2:
            level_opt.step()
            level_opt.zero_grad(set_to_none=True)
            mu_opt.step()
            mu_opt.zero_grad(set_to_none=True)

        # ---- lambda adaptation (every 250 steps, Phase 2 only) ----
        if phase2 and step % 250 == 0:
            with torch.no_grad():
                r_b = (rate_real / budget_bytes).item()
                if lambda_mode == "dual":
                    # dual ascent on the budget constraint:
                    # above budget -> lambda rises (more rate pressure),
                    # comfortably below -> lambda falls (relax).
                    lam.add_(lambda_eta * (r_b - 1.0))
                else:
                    if r_b > 1.02:
                        lam.mul_(1.15)
                    elif r_b < 0.90:
                        lam.mul_(0.85)
                lam.clamp_(1e-5, 1.0)

        # ---- position noise (SGLD, means only, anisotropic) ----
        if phase2 and position_noise:
            with torch.no_grad():
                # MCMC: noise follows each Gaussian's covariance, scaled by opacity.
                # Amplitude anneals to zero as g -> 1 so late-stage geometry settles.
                noise_scale = 5e-5 * scene.scene_scale * (1.0 - g)
                geo = torch.exp(splats["scales"].detach())  # (N,3)
                eps = torch.randn_like(splats["means"])
                # opaque Gaussians get less noise (MCMC Eq.8 style)
                o = torch.sigmoid(splats["opacities"].detach()).unsqueeze(1)
                damp = 1.0 / (1.0 + 100.0 * o)
                splats["means"].add_(noise_scale * eps * geo * damp)

        # ---- densification post-backward (Phase 1) / pruning (Phase 2) ----
        if not phase2:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=step, info=info, packed=True,
            )
        else:
            # linear N schedule from current N at warmup down to target_n
            if step == warmup + 1:
                n_at_warmup = splats["means"].shape[0]
            if step % prune_every == 0:
                if _budget_satisfied:
                    # P0-2c: bytes already comfortably within budget -> stop
                    # pruning N and freeze the current Gaussian set.
                    n_sched = splats["means"].shape[0]
                else:
                    hold = steps if n_hold_step is None else min(n_hold_step, steps)
                    frac = (min(step, hold) - warmup) / max(steps - warmup, 1)
                    n_sched = int(round(n_at_warmup + (target_n - n_at_warmup) * frac))
                importance = torch.sigmoid(splats["opacities"]).detach()
                removed = prune_splats(splats, optimizers, n_sched, importance)
                n_pruned_total += removed

        means_scheduler.step()

        if step % 250 == 0 or step == 1:
            n_now = splats["means"].shape[0]
            msg = f"step {step:>6d} g={g:.2f} | loss {loss.item():.4f} | N {n_now}"
            if phase2:
                intl = schedule.integer_targets()
                msg += f" | lam {lam.item():.1e} | tgtL means={intl['means']} shn={intl['shn']}"
            print(msg, flush=True)

        # ---- periodic real-byte probe + validation ----
        if phase2 and (step % 1000 == 0 or step == warmup + 1):
            global _LAST_INTEGER_LEVELS
            _LAST_INTEGER_LEVELS = schedule.integer_targets()
            real_b, intl = probe_real_bytes(
                splats, ch, device, mu_per_attr=mu_map()
            )
            if calib is None:
                with torch.no_grad():
                    sur = surrogate_bytes(schedule.levels_at(g), splats["means"].shape[0], ch)
                calib = real_b / sur.item()
            # quick val PSNR at current quantized levels
            val_psnr = _quick_val_psnr(
                splats, schedule, g, c2w, Ks, width, height, images, val_idcs,
                device, antialiased, sh_degree, bkgd, mu_per_attr=mu_map()
            )
            row = {
                "step": step,
                "n": int(splats["means"].shape[0]),
                "real_bytes": int(real_b),
                "real_mb": round(real_b / 1e6, 4),
                "calib": round(calib, 4),
                "val_psnr": round(val_psnr, 3),
                "lam": round(lam.item(), 6),
                "lambda_mode": lambda_mode,
                "integer_levels": intl,
            }
            history.append(row)
            # C: online R-D marginal probe (from 1000 steps into Phase 2 on)
            if step >= warmup + 1000:
                mres = probe_marginal(
                    splats, schedule, c2w, Ks, width, height, images, val_idcs,
                    device, antialiased, sh_degree, bkgd, ch, budget_bytes, calib,
                    mu_per_attr=mu_map()
                )
                row.update(mres)
                print(
                    f"  [probe] real {real_b/1e6:.3f} MB (calib {calib:.3f}) | "
                    f"val PSNR {val_psnr:.2f} | N {row['n']} | lam {lam.item():.1e} | "
                    f"MR moved={mres['probe_moved']} tgt={mres['probe_new_tgt']}",
                    flush=True,
                )
            else:
                print(
                    f"  [probe] real {real_b/1e6:.3f} MB (calib {calib:.3f}) | "
                    f"val PSNR {val_psnr:.2f} | N {row['n']} | lam {lam.item():.1e}",
                    flush=True,
                )
            with open(os.path.join(out_dir, f"joint-{scene_name}-history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            # dual-ascent real anchor: re-point lambda at the TRUE probed bytes
            # once per probe interval (the surrogate updates above are 4x more
            # frequent and only calibrated; this one uses the ground truth).
            if lambda_mode == "dual":
                with torch.no_grad():
                    excess_r = real_b / budget_bytes - 1.0
                    lam.add_(lambda_eta * excess_r)
                    # Fast release: once real bytes sit comfortably under
                    # budget for two consecutive probes, stop dribbling
                    # lambda down step by step and cut it multiplicatively
                    # so late-stage gradient effort goes back to distortion.
                    if real_b / budget_bytes < 0.95:
                        _relax_streak += 1
                    else:
                        _relax_streak = 0
                    if _relax_streak >= 2:
                        lam.mul_(0.5)
                        # P0-2c: lock in budget satisfaction, freeze N pruning
                        # and anneal the opacity regulariser (once per probe).
                        _budget_satisfied = True
                        reg_o_scale *= 0.8
                    lam.clamp_(1e-5, 1.0)

    # ---- save final joint checkpoint --------------------------------------
    final_ckpt = {
        "means": splats["means"].detach().cpu(),
        "scales": splats["scales"].detach().cpu(),
        "quats": splats["quats"].detach().cpu(),
        "opacity": splats["opacities"].detach().cpu(),
        "sh0": splats["sh0"].detach().cpu(),
        "shn": splats["shn"].detach().cpu(),
        "step": steps,
        "seed": seed,
        "config": {
            "steps": steps, "warmup": warmup, "target_n": target_n,
            "budget_mb": budget_mb, "factor": factor, "sh_degree": sh_degree,
            "antialiased": antialiased, "lambda_mode": lambda_mode,
            "lambda_eta": lambda_eta,
        },
        "n_gaussians": int(splats["means"].shape[0]),
        "mulaw_mu": mu_values().detach().cpu(),
    }
    ckpt_path = os.path.join(out_dir, f"joint-{scene_name}.pt")
    torch.save(final_ckpt, ckpt_path)
    print("joint ckpt saved:", ckpt_path, flush=True)

    # ---- real landing -----------------------------------------------------
    result = real_landing(
        splats, schedule, scene, scene_name, c2w, Ks, width, height, images,
        val_idcs, device, antialiased, sh_degree, out_dir, budget_mb,
        mu_per_attr=mu_map()
    )
    result["n_pruned_total"] = n_pruned_total
    result["history"] = history
    with open(os.path.join(out_dir, f"joint-{scene_name}-result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("JOINT TRAINING DONE", flush=True)
    return result


def _quick_val_psnr(splats, schedule, g, c2w, Ks, width, height, images, val_idcs,
                    device, antialiased, sh_degree, bkgd, mu_per_attr=None):
    import torch
    from budgetgs.modules.trainer import _rasterize, _compose_bkgd
    levels = schedule.levels_at(min(g, 1.0))
    levels_map = {name: levels[j] for j, name in enumerate(ATTR_ORDER)}
    view = quantized_splats(splats, levels_map, mu_per_attr=mu_per_attr)
    psnrs = []
    with torch.no_grad():
        for i in val_idcs:
            r, a, _ = _rasterize(view, c2w, Ks, width, height, [i], device, antialiased, sh_degree)
            r = _compose_bkgd(r, a, bkgd)
            mse = torch.mean((r - images[i]) ** 2)
            psnrs.append(float(-10.0 * torch.log10(mse)))
    return sum(psnrs) / len(psnrs)


def _probe_psnr_at_levels(splats, levels_map, c2w, Ks, width, height, images,
                          val_idcs, device, antialiased, sh_degree, bkgd,
                          mu_per_attr=None):
    """PSNR over ``val_idcs`` (a subset) at explicit integer levels, using the
    true (discrete) quantization path — what the marginal probe should measure."""
    import torch
    from budgetgs.modules.trainer import _rasterize, _compose_bkgd
    view = quantized_splats(splats, levels_map, mu_per_attr=mu_per_attr)
    psnrs = []
    with torch.no_grad():
        for i in val_idcs:
            r, a, _ = _rasterize(view, c2w, Ks, width, height, [i], device, antialiased, sh_degree)
            r = _compose_bkgd(r, a, bkgd)
            mse = torch.mean((r - images[i]) ** 2)
            psnrs.append(float(-10.0 * torch.log10(mse)))
    return sum(psnrs) / len(psnrs)


def probe_marginal(splats, schedule, c2w, Ks, width, height, images, val_idcs,
                   device, antialiased, sh_degree, bkgd, ch, budget_bytes, calib,
                   n_probe_views=2, n_steps=2, mu_per_attr=None):
    """Online R-D marginal-gain probe (video-codec style budget allocation).

    At the current integer levels, estimate dPSNR/dBytes for each attribute by
    re-rendering a few val views at level+1 and level-1 (true discrete path).
    Then greedily re-allocate: if over budget, drop the level with the SMALLEST
    PSNR loss per saved byte; if well under budget, raise the level with the
    LARGEST PSNR gain per extra byte. schedule.raw is rewritten so the learned
    levels track the probe (the SGD path then fine-tunes around it). This does
    not depend on the zero first-order gradient of the round path — it measures
    the actual PSNR-vs-bytes trade-off, i.e. the user's 1MB marginal heuristic
    automated inside training.
    """
    import math
    import torch

    intl = schedule.integer_targets()
    mn = {name: float(schedule.min_levels[j]) for j, name in enumerate(ATTR_ORDER)}
    n = splats["means"].shape[0]
    vi = list(val_idcs[:n_probe_views])

    base_map = {name: float(intl[name]) for name in ATTR_ORDER}
    base_psnr = _probe_psnr_at_levels(
        splats, base_map, c2w, Ks, width, height, images, vi,
        device, antialiased, sh_degree, bkgd, mu_per_attr=mu_per_attr
    )
    sur = surrogate_bytes(schedule.levels_at(1.0), n, ch)
    rate_real = (sur * calib).item() if calib is not None else sur.item()

    def dbytes(name, lv_old, lv_new):
        return n * ch[name] * math.log2(lv_new / lv_old) / 8.0 * (calib or 1.0)

    moves_up, moves_down = [], []
    for name in ATTR_ORDER:
        lv = intl[name]
        if lv < 4095:
            up_map = dict(base_map)
            up_map[name] = lv + 1
            up_psnr = _probe_psnr_at_levels(
                splats, up_map, c2w, Ks, width, height, images, vi,
                device, antialiased, sh_degree, bkgd, mu_per_attr=mu_per_attr
            )
            gain = up_psnr - base_psnr
            if gain > 0:
                moves_up.append((name, gain / max(dbytes(name, lv, lv + 1), 1e-9), gain))
        if lv > mn[name] + 1:
            dn_map = dict(base_map)
            dn_map[name] = lv - 1
            dn_psnr = _probe_psnr_at_levels(
                splats, dn_map, c2w, Ks, width, height, images, vi,
                device, antialiased, sh_degree, bkgd, mu_per_attr=mu_per_attr
            )
            loss = base_psnr - dn_psnr
            if loss > 0:
                moves_down.append((name, loss / max(dbytes(name, lv, lv - 1), 1e-9), loss))

    new_tgt = {name: float(intl[name]) for name in ATTR_ORDER}
    n_moved = 0
    if rate_real > budget_bytes * 1.01 and moves_down:
        # over budget: drop levels with the smallest PSNR loss per saved byte
        for _ in range(n_steps):
            cand = [m for m in moves_down if new_tgt[m[0]] > mn[m[0]] + 1]
            if not cand:
                break
            best = min(cand, key=lambda m: m[1])
            new_tgt[best[0]] = max(new_tgt[best[0]] - 1, mn[best[0]] + 1)
            n_moved += 1
    elif rate_real < budget_bytes * 0.95 and moves_up:
        # under budget: raise the level with the best PSNR per extra byte
        for _ in range(n_steps):
            cand = [m for m in moves_up if new_tgt[m[0]] < 4095]
            if not cand:
                break
            best = max(cand, key=lambda m: m[1])
            new_tgt[best[0]] = min(new_tgt[best[0]] + 1, 4095)
            n_moved += 1

    if n_moved:
        with torch.no_grad():
            for j, name in enumerate(ATTR_ORDER):
                t = min(max(new_tgt[name], mn[name] + 1.0), schedule.L_MAX - 1.0)
                p = (t - mn[name]) / (schedule.L_MAX - mn[name])
                p = min(max(p, 1e-6), 1.0 - 1e-6)
                schedule.raw[j] = math.log(p / (1.0 - p))

    return {
        "probe_base_psnr": round(base_psnr, 3),
        "probe_rate_mb": round(rate_real / 1e6, 3),
        "probe_moved": n_moved,
        "probe_new_tgt": {name: int(round(new_tgt[name])) for name in ATTR_ORDER},
    }


# ---------------------------------------------------------------------------
# real landing through the production serializer
# ---------------------------------------------------------------------------
def real_landing(splats, schedule, scene, scene_name, c2w, Ks, width, height,
                 images, val_idcs, device, antialiased, sh_degree, out_dir,
                 budget_mb, mu_per_attr=None):
    import numpy as np
    import torch

    from budgetgs.modules.serialize import CODEC_RANGE
    from budgetgs.evaluation.landing import _metrics, _render_views

    int_levels = schedule.integer_targets()
    level_map = {CONTAINER_NAME[name]: int_levels[name] for name in ATTR_ORDER}
    # container-keyed fixed mu values for the numpy writer (None/0 -> uniform)
    if mu_per_attr is not None:
        mu_level_map = {
            CONTAINER_NAME[name]: float(mu_per_attr[name].detach())
            for name in ATTR_ORDER
        }
    else:
        mu_level_map = {CONTAINER_NAME[name]: 0.0 for name in ATTR_ORDER}

    # build numpy attributes in container layout
    attributes = {}
    for name in ATTR_ORDER:
        cname = CONTAINER_NAME[name]
        x = splats[name].detach()
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        attributes[cname] = x.cpu().numpy().astype(np.float64)

    # real write/read via the driver mixed-container path (per-group levels)
    from directionB_scene import (
        write_mixed_container,
        dequant_mixed,
        split_shn_groups,
    )
    tmp_dir = os.path.join(out_dir, "tmp_containers")
    os.makedirs(tmp_dir, exist_ok=True)
    container_path = os.path.join(tmp_dir, f"joint-{scene_name}.bgs")
    groups = tuple((cname, attributes[cname].shape[1]) for cname in
                  ["means", "scales", "quats", "opacity", "sh0", "shn"])
    n_bytes = write_mixed_container(
        container_path, attributes, groups, level_map,
        codec=CODEC_RANGE, mu_map=mu_level_map
    )

    from budgetgs.modules.serialize import read_container
    decoded = read_container(container_path, groups=split_shn_groups(groups))
    recon = dequant_mixed(decoded, level_map)

    # LPIPS
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc), flush=True)

    landed_splats = _splats_from_recon(recon, device, sh_degree)

    # streamed per-view evaluation: render a single view, score it, free it.
    # (factor=1 full-res landing previously OOM-killed the 9G cgroup when all
    #  val views were cat'ed into one tensor: landed 3GB + ref 3GB + images 6GB)
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    per_view = []
    for j, i in enumerate(val_idcs):
        lv = _render_views(landed_splats, scene, [i], device, sh_degree, antialiased)
        m = _metrics(lv, images[i], lpips_model, device)
        per_view.append({"view": int(i), **m})
        for k in totals:
            totals[k] += m[k]
        del lv
    n = len(val_idcs)
    landed = {k: round(v / n, 4) for k, v in totals.items()}

    # also score the unquantized joint model for reference (streamed too)
    ref_splats = _splats_from_live(splats, device, sh_degree)
    rt = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for j, i in enumerate(val_idcs):
        rv = _render_views(ref_splats, scene, [i], device, sh_degree, antialiased)
        m = _metrics(rv, images[i], lpips_model, device)
        for k in rt:
            rt[k] += m[k]
        del rv
    ref = {k: round(v / n, 4) for k, v in rt.items()}

    try:
        os.unlink(container_path)
    except OSError:
        pass

    result = {
        "scene": scene_name,
        "n_gaussians": int(splats["means"].shape[0]),
        "integer_levels": int_levels,
        "mulaw_mu": {k: round(v, 4) for k, v in mu_level_map.items()},
        "actual_bytes": int(n_bytes),
        "actual_mb": round(n_bytes / 1e6, 4),
        "within_budget": bool(n_bytes <= budget_mb * 1_000_000),
        "landed": landed,
        "ref": ref,
        "per_view": per_view,
        "target_met": bool(n_bytes <= budget_mb * 1_000_000 and landed["psnr"] > 20.0),
    }
    print(
        f"[landed] {n_bytes/1e6:.3f} MB within={result['within_budget']} | "
        f"PSNR {landed['psnr']} SSIM {landed['ssim']} LPIPS {landed['lpips']} "
        f"(ref {ref['psnr']}) | TARGET_MET={result['target_met']}",
        flush=True,
    )
    return result


def _splats_from_recon(recon, device, sh_degree):
    from budgetgs.evaluation.landing import _splats_tensors
    return _splats_tensors(recon, device, sh_degree)


def _splats_from_live(splats, device, sh_degree):
    import torch
    k = (sh_degree + 1) ** 2 - 1
    return {
        "means": splats["means"].detach(),
        "scales": splats["scales"].detach(),
        "quats": splats["quats"].detach(),
        "opacities": splats["opacities"].detach(),
        "sh0": splats["sh0"].detach(),
        "shn": splats["shn"].detach(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="train_joint", description=__doc__)
    ap.add_argument("--scene", required=True, help="360_v2 scene directory")
    ap.add_argument("--scene-name", required=True)
    ap.add_argument("--out-dir", default="runs/joint")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--warmup", type=int, default=15_000)
    ap.add_argument("--target-n", type=int, default=400_000)
    ap.add_argument("--budget-mb", type=float, default=10.0)
    ap.add_argument("--prune-every", type=int, default=250)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--no-position-noise", action="store_true")
    ap.add_argument(
        "--lambda-mode", choices=("dual", "sched"), default="dual",
        help="lambda adaptation: 'dual' = dual ascent on the budget constraint "
             "(default); 'sched' = legacy multiplicative heuristic (ablation).",
    )
    ap.add_argument("--lambda-eta", type=float, default=1e-4,
                    help="dual-ascent step size for lambda (R/B surplus signal); "
                         "1e-4 measured best (dual vs sched ablation).")
    ap.add_argument(
        "--n-hold-step", type=int, default=None,
        help="freeze the Gaussian-count schedule at this step (e.g. 27000). "
             "Default None = anneal all the way to steps.",
    )
    args = ap.parse_args(argv)
    train_joint(
        args.scene, args.scene_name, args.out_dir, steps=args.steps,
        warmup=args.warmup, target_n=args.target_n, budget_mb=args.budget_mb,
        prune_every=args.prune_every, sh_degree=args.sh_degree, seed=args.seed,
        factor=args.factor, position_noise=not args.no_position_noise,
        lambda_mode=args.lambda_mode, lambda_eta=args.lambda_eta,
        n_hold_step=args.n_hold_step,
    )


if __name__ == "__main__":
    main()
