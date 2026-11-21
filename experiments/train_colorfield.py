#!/usr/bin/env python
"""BudgetGS P0: shared hashgrid color field replaces per-Gaussian SH color.

Same joint train/quantize/prune/landing pipeline as ``train_joint.py``, but the
color lives in a shared ``ColorField`` (hash grid + MLP) instead of 48 float SH
coefficients per Gaussian.  The byte ledger then has only 4 geometric attribute
groups plus a FIXED hashgrid+MLP term, so bytes no longer grow with the number
of Gaussians through the color block — the saved budget goes to geometry.

Derived from train_joint.py; kept as a separate script so the supervisor's
original stays untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# geometry-only attribute groups (no sh0/sh2)
# ---------------------------------------------------------------------------
ATTR_ORDER = ["means", "scales", "quats", "opacities"]
CONTAINER_NAME = {
    "means": "means",
    "scales": "scales",
    "quats": "quats",
    "opacities": "opacity",
}


def channel_counts() -> dict:
    return {"means": 3, "scales": 3, "quats": 4, "opacities": 1}


# ---------------------------------------------------------------------------
# differentiable per-channel quantization (unchanged from train_joint.py)
# ---------------------------------------------------------------------------
def _mulaw_compand(u, mu):
    import torch

    t = 2.0 * u - 1.0
    log_mu = torch.log1p(mu)
    t_c = torch.sign(t) * torch.log1p(mu * torch.abs(t)) / log_mu
    v = (t_c + 1.0) * 0.5
    return torch.where(mu > 1e-6, v, u)


def _mulaw_expand(v, mu):
    import torch

    t = 2.0 * v - 1.0
    log_mu = torch.log1p(mu)
    t_e = torch.sign(t) * torch.expm1(torch.abs(t) * log_mu) / mu
    u = (t_e + 1.0) * 0.5
    return torch.where(mu > 1e-6, u, v)


def _mulaw_compand_np(u, mu):
    import numpy as np

    if mu <= 1e-6:
        return u
    t = 2.0 * u - 1.0
    log_mu = np.log1p(mu)
    t_c = np.sign(t) * np.log1p(mu * np.abs(t)) / log_mu
    return (t_c + 1.0) * 0.5


def quantize_attr(x, levels, noise=False, mu=0.0):
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
    u = (x - lo) / rng
    v = _mulaw_compand(u, mu)
    if noise:
        qn = v * l_minus_1 + (torch.rand_like(v) - 0.5)
        qn = torch.clamp(qn, qmin, qmax)
        vq = qn / l_minus_1
        uq = _mulaw_expand(vq, mu)
        xq = lo + uq * rng
        return xq.reshape(orig_shape)
    q = torch.clamp(torch.round(v * l_minus_1), qmin, qmax)
    vq = q / l_minus_1
    uq = _mulaw_expand(vq, mu)
    xq = lo + uq * rng
    vc = torch.clamp(v * l_minus_1, qmin, qmax) / l_minus_1
    uc = _mulaw_expand(vc, mu)
    xq_c = lo + uc * rng
    out = xq_c + (xq - xq_c).detach()
    return out.reshape(orig_shape)


def quantized_splats(splats, levels_per_attr, noise=False, mu_per_attr=None):
    out = {}
    for name in ATTR_ORDER:
        mu = mu_per_attr[name] if mu_per_attr is not None else 0.0
        out[name] = quantize_attr(
            splats[name], levels_per_attr[name], noise=noise, mu=mu
        )
    return out


class LevelSchedule:
    L_MAX = 4096.0
    L_START = 4096.0

    def __init__(self, device, init_targets, min_levels):
        self.min_levels = torch.tensor(list(min_levels), device=device, dtype=torch.float32)
        raws = []
        for t, mn in zip(init_targets, min_levels):
            t = min(max(t, mn + 1.0), self.L_MAX - 1.0)
            p = (t - mn) / (self.L_MAX - mn)
            raws.append(math.log(p / (1.0 - p)))
        self.raw = torch.nn.Parameter(torch.tensor(raws, device=device, dtype=torch.float32))

    def target_levels(self):
        s = torch.sigmoid(self.raw)
        return self.min_levels + (self.L_MAX - self.min_levels) * s

    def levels_at(self, g):
        tgt = self.target_levels()
        log_l = (1.0 - g) * math.log(self.L_START) + g * torch.log(tgt)
        return torch.exp(log_l)

    def integer_targets(self):
        tgt = self.target_levels().detach().cpu().tolist()
        mn = self.min_levels.detach().cpu().tolist()
        return {name: max(int(round(tgt[i])), int(mn[i])) for i, name in enumerate(ATTR_ORDER)}


# ---------------------------------------------------------------------------
# byte ledger: 4 geometric groups + a fixed hashgrid/MLP term
# ---------------------------------------------------------------------------
def surrogate_bytes(levels, n, ch, fixed_bytes):
    import torch

    total = None
    for i, name in enumerate(ATTR_ORDER):
        bits = n * ch[name] * torch.log2(levels[i])
        total = bits if total is None else total + bits
    return total / 8.0 + fixed_bytes


def prune_splats(splats, optimizers, n_keep, importance):
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


def _nearest_keep(means_remove, means_keep, chunk_r=512, chunk_k=8192):
    """For each remove point, the index (into means_keep) of the nearest keep
    point. Chunked cdist keeps memory bounded. Returns (R,) long tensor."""
    import torch

    R = means_remove.shape[0]
    K = means_keep.shape[0]
    device = means_remove.device
    best_idx = torch.zeros(R, dtype=torch.long, device=device)
    best_dist = torch.full((R,), float("inf"), device=device, dtype=means_remove.dtype)
    for i in range(0, R, chunk_r):
        mr = means_remove[i : i + chunk_r]
        i_end = i + mr.shape[0]
        for j in range(0, K, chunk_k):
            mk = means_keep[j : j + chunk_k]
            d = torch.cdist(mr, mk)
            min_d, min_j = d.min(dim=1)
            upd = min_d < best_dist[i:i_end]
            best_idx[i:i_end] = torch.where(upd, j + min_j, best_idx[i:i_end])
            best_dist[i:i_end] = torch.minimum(best_dist[i:i_end], min_d)
    return best_idx


def prune_splats_relocate(splats, optimizers, n_keep, importance, chunk_r=512, chunk_k=8192):
    """MCMC-style mass-conserving prune (P1).

    Instead of hard-dropping the low-opacity tail, each removed Gaussian's
    opacity is merged into its nearest surviving neighbor (transmittance
    conservation), then the tail is dropped. Multiple removals mapping to the
    same survivor multiply their transmittances (scatter_reduce prod).
    """
    import torch

    n = splats["means"].shape[0]
    if n_keep >= n:
        return 0

    means = splats["means"].detach()
    opacity = splats["opacities"].detach()

    idx_sorted = torch.argsort(importance, descending=True)
    keep_idx = idx_sorted[:n_keep]
    remove_idx = idx_sorted[n_keep:]

    # nearest surviving neighbor for each removed Gaussian
    nearest = _nearest_keep(means[remove_idx], means[keep_idx], chunk_r, chunk_k)  # (R,) -> keep-local
    keep_global = keep_idx[nearest]  # (R,) -> global index

    # transmittance conservation: t_keep <- t_keep * prod_{d->keep} t_remove
    alpha_keep = torch.sigmoid(opacity[keep_idx])  # (K,)
    trans_keep = 1.0 - alpha_keep
    trans_remove = 1.0 - torch.sigmoid(opacity[remove_idx])  # (R,)

    contrib = torch.ones_like(trans_keep)
    contrib.scatter_reduce_(0, nearest, trans_remove, reduce="prod", include_self=False)
    final_trans = trans_keep * contrib
    final_alpha = (1.0 - final_trans).clamp(min=1e-6, max=1.0 - 1e-6)
    opacity_new = torch.logit(final_alpha)

    new_opacity = opacity.clone()
    new_opacity[keep_idx] = opacity_new
    splats["opacities"] = torch.nn.Parameter(new_opacity)

    # drop the removed tail (keep_idx is sorted ascending because idx_sorted
    # came from argsort over importance, but keep_idx itself must be sorted for
    # clean indexing)
    keep_sorted, _ = torch.sort(keep_idx)
    for name in ATTR_ORDER:
        p = splats[name]
        new_p = torch.nn.Parameter(p.data[keep_sorted])
        splats[name] = new_p
        old_lr = optimizers[name].param_groups[0]["lr"]
        optimizers[name] = torch.optim.Adam(
            [{"params": new_p, "lr": old_lr}], eps=1e-15, betas=(0.9, 0.999)
        )
    return int(n - n_keep)


_LAST_INTEGER_LEVELS = {name: 4096 for name in ATTR_ORDER}


def probe_real_bytes(splats, ch, device, mu_per_attr, fixed_bytes):
    import numpy as np

    from budgetgs.modules.probability import CategoricalModel

    int_levels = _LAST_INTEGER_LEVELS
    return_bytes = fixed_bytes
    for name in ATTR_ORDER:
        x = splats[name].detach()
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        arr = x.cpu().numpy().astype(np.float64)
        mu = float(mu_per_attr[name].detach()) if mu_per_attr is not None else 0.0
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
        return_bytes += coded + 4 + 4 + 8 * model.n_distinct
    return return_bytes, int_levels


# ---------------------------------------------------------------------------
# color-field rasterizer (view-independent: sh_degree=None RGB passthrough)
# ---------------------------------------------------------------------------
def _rasterize_cf(geo, color_field, c2w, Ks, width, height, view_idcs, device, antialiased):
    import torch
    from gsplat.rendering import rasterization

    rgb = color_field(geo["means"]).float()
    mode = "antialiased" if antialiased else "classic"
    return rasterization(
        means=geo["means"],
        quats=geo["quats"],
        scales=torch.exp(geo["scales"]),
        opacities=torch.sigmoid(geo["opacities"]),
        colors=rgb,
        viewmats=torch.linalg.inv_ex(c2w[view_idcs]).inverse,
        Ks=Ks[view_idcs],
        width=width,
        height=height,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=mode,
        distributed=False,
        sh_degree=None,
    )


def _compose_bkgd(colors, alphas, bkgd):
    return colors + bkgd * (1.0 - alphas)


# ---------------------------------------------------------------------------
# main training
# ---------------------------------------------------------------------------
def train_colorfield(
    scene_dir,
    scene_name,
    out_dir,
    steps=30_000,
    warmup=15_000,
    target_n=700_000,
    budget_mb=10.0,
    prune_every=250,
    seed=0,
    factor=2,
    device="cuda",
    antialiased=False,
    test_every=8,
    position_noise=True,
    lambda_mode="dual",
    lambda_eta=1e-4,
    n_hold_step=None,
    qat=True,
    rate=True,
    probe=True,
    n_levels=10,
    log2_hashmap_size=17,
    view_dependent=False,
    hash_lr=1e-3,
    relocate=False,
):
    import numpy as np
    import torch

    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import _load_image, create_splats
    from budgetgs.modules.color_field import ColorField

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ch = channel_counts()

    scene = load_360_scene(scene_dir, factor=factor)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, test_every))
    train_idcs = [i for i in range(n_views) if i not in val_idcs]
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    images = {i: _load_image(scene.image_paths[i], device, size=(width, height)) for i in range(n_views)}
    print(f"scene={scene_name} views={n_views} val={len(val_idcs)} size={width}x{height}", flush=True)

    # ---- geometry splats + shared color field ---------------------------------
    splats, optimizers = create_splats(scene, device=device, sh_degree=3)
    del splats["sh0"], splats["shn"]
    del optimizers["sh0"], optimizers["shn"]

    color_field = ColorField(
        n_levels=n_levels,
        log2_hashmap_size=log2_hashmap_size,
        view_dependent=view_dependent,
    ).to(device)
    optimizers["color_field"] = torch.optim.Adam(color_field.parameters(), lr=hash_lr)
    fixed_bytes = color_field.storage_bytes(bits_per_hash=8, bits_per_mlp=16)
    print(f"color field params={color_field.num_params():,} fixed_bytes={fixed_bytes:,} "
          f"({fixed_bytes/1e6:.2f} MB)", flush=True)

    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(verbose=False)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)

    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(steps, 1))
    )

    init_targets = {"means": 1024.0, "scales": 128.0, "quats": 128.0, "opacities": 32.0}
    min_levels = {"means": 4.0, "scales": 4.0, "quats": 4.0, "opacities": 4.0}
    schedule = LevelSchedule(
        device, [init_targets[n] for n in ATTR_ORDER],
        [min_levels[n] for n in ATTR_ORDER],
    )
    level_opt = torch.optim.SGD([{"params": schedule.raw, "lr": 1e-3}], momentum=0.0)

    mu_raw = torch.nn.Parameter(torch.full((len(ATTR_ORDER),), -1.0, device=device, dtype=torch.float32))
    mu_opt = torch.optim.SGD([{"params": mu_raw, "lr": 1e0}], momentum=0.0)

    def mu_values():
        raw_c = mu_raw.clamp(-5.0, 6.0)
        return torch.nn.functional.softplus(raw_c) + 1e-6

    def mu_map():
        muv = mu_values()
        return {name: muv[j] for j, name in enumerate(ATTR_ORDER)}

    budget_bytes = budget_mb * 1_000_000.0
    bkgd = torch.ones(3, device=device)

    lam = torch.tensor(1e-2, device=device)
    _relax_streak = 0
    _budget_satisfied = False
    reg_o_scale = 1.0
    calib = None
    history = []
    n_pruned_total = 0

    def render_view(splats_view, i):
        render, alphas, info = _rasterize_cf(
            splats_view, color_field, c2w, Ks, width, height, [i], device, antialiased
        )
        render = _compose_bkgd(render, alphas, bkgd)
        return render, info

    for step in range(1, steps + 1):
        phase2 = step > warmup
        i = random.choice(train_idcs)
        gt = images[i]

        if phase2:
            g = (step - warmup) / max(steps - warmup, 1)
            levels = schedule.levels_at(g)
            levels_map = {name: levels[j] for j, name in enumerate(ATTR_ORDER)}
            if qat:
                view = quantized_splats(
                    splats, levels_map, noise=not _budget_satisfied, mu_per_attr=mu_map()
                )
                render, info = render_view(view, i)
            else:
                render, info = render_view(splats, i)
        else:
            g = 0.0
            render, info = render_view(splats, i)

        if not phase2:
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=step, info=info,
            )

        loss = torch.nn.functional.l1_loss(render, gt)

        if phase2:
            reg_o = 0.01 * reg_o_scale * torch.sigmoid(splats["opacities"]).mean()
            reg_s = 0.01 * torch.exp(splats["scales"]).mean()
            loss = loss + reg_o + reg_s

        if phase2 and rate:
            n_now = splats["means"].shape[0]
            rate_val = surrogate_bytes(levels, n_now, ch, fixed_bytes)
            rate_real = rate_val * calib if calib is not None else rate_val
            loss = loss + lam * (rate_real / budget_bytes)

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if phase2:
            if rate:
                level_opt.step()
                level_opt.zero_grad(set_to_none=True)
            mu_opt.step()
            mu_opt.zero_grad(set_to_none=True)

        if phase2 and rate and step % 250 == 0:
            with torch.no_grad():
                r_b = (rate_real / budget_bytes).item()
                if lambda_mode == "dual":
                    lam.add_(lambda_eta * (r_b - 1.0))
                else:
                    if r_b > 1.02:
                        lam.mul_(1.15)
                    elif r_b < 0.90:
                        lam.mul_(0.85)
                lam.clamp_(1e-5, 1.0)

        if phase2 and position_noise:
            with torch.no_grad():
                noise_scale = 5e-5 * scene.scene_scale * (1.0 - g)
                geo = torch.exp(splats["scales"].detach())
                eps = torch.randn_like(splats["means"])
                o = torch.sigmoid(splats["opacities"].detach()).unsqueeze(1)
                damp = 1.0 / (1.0 + 100.0 * o)
                splats["means"].add_(noise_scale * eps * geo * damp)

        if not phase2:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=step, info=info, packed=True,
            )
        else:
            if step == warmup + 1:
                n_at_warmup = splats["means"].shape[0]
            if step % prune_every == 0:
                if _budget_satisfied:
                    n_sched = splats["means"].shape[0]
                else:
                    hold = steps if n_hold_step is None else min(n_hold_step, steps)
                    frac = (min(step, hold) - warmup) / max(steps - warmup, 1)
                    n_sched = int(round(n_at_warmup + (target_n - n_at_warmup) * frac))
                importance = torch.sigmoid(splats["opacities"]).detach()
                if relocate:
                    removed = prune_splats_relocate(splats, optimizers, n_sched, importance)
                else:
                    removed = prune_splats(splats, optimizers, n_sched, importance)
                n_pruned_total += removed

        means_scheduler.step()

        if step % 250 == 0 or step == 1:
            n_now = splats["means"].shape[0]
            msg = f"step {step:>6d} g={g:.2f} | loss {loss.item():.4f} | N {n_now}"
            if phase2:
                intl = schedule.integer_targets()
                msg += f" | lam {lam.item():.1e} | tgtL means={intl['means']}"
            print(msg, flush=True)

        if phase2 and (step % 1000 == 0 or step == warmup + 1):
            global _LAST_INTEGER_LEVELS
            _LAST_INTEGER_LEVELS = schedule.integer_targets()
            real_b, intl = probe_real_bytes(splats, ch, device, mu_map(), fixed_bytes)
            if calib is None:
                with torch.no_grad():
                    sur = surrogate_bytes(schedule.levels_at(g), splats["means"].shape[0], ch, fixed_bytes)
                calib = real_b / sur.item()
            val_psnr = _quick_val_psnr_cf(
                splats, color_field, schedule, g, c2w, Ks, width, height, images, val_idcs,
                device, antialiased, bkgd, mu_per_attr=mu_map()
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
            if probe and step >= warmup + 1000:
                mres = probe_marginal_cf(
                    splats, color_field, schedule, c2w, Ks, width, height, images, val_idcs,
                    device, antialiased, bkgd, ch, budget_bytes, calib, fixed_bytes,
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

            if rate and lambda_mode == "dual":
                with torch.no_grad():
                    excess_r = real_b / budget_bytes - 1.0
                    lam.add_(lambda_eta * excess_r)
                    if real_b / budget_bytes < 0.95:
                        _relax_streak += 1
                    else:
                        _relax_streak = 0
                    if _relax_streak >= 2:
                        lam.mul_(0.5)
                        _budget_satisfied = True
                        reg_o_scale *= 0.8
                    lam.clamp_(1e-5, 1.0)

    # ---- save final checkpoint (4 geometry groups + color field) ----
    final_ckpt = {
        "means": splats["means"].detach().cpu(),
        "scales": splats["scales"].detach().cpu(),
        "quats": splats["quats"].detach().cpu(),
        "opacity": splats["opacities"].detach().cpu(),
        "color_field": {k: v.detach().cpu() for k, v in color_field.state_dict().items()},
        "color_field_cfg": {
            "n_levels": n_levels, "log2_hashmap_size": log2_hashmap_size,
            "view_dependent": view_dependent,
        },
        "step": steps,
        "seed": seed,
        "config": {
            "steps": steps, "warmup": warmup, "target_n": target_n,
            "budget_mb": budget_mb, "factor": factor,
            "antialiased": antialiased, "lambda_mode": lambda_mode,
            "lambda_eta": lambda_eta,
        },
        "n_gaussians": int(splats["means"].shape[0]),
        "mulaw_mu": mu_values().detach().cpu(),
    }
    ckpt_path = os.path.join(out_dir, f"joint-{scene_name}.pt")
    torch.save(final_ckpt, ckpt_path)
    print("joint ckpt saved:", ckpt_path, flush=True)

    result = real_landing_cf(
        splats, color_field, schedule, scene, scene_name, c2w, Ks, width, height,
        images, val_idcs, device, antialiased, out_dir, budget_mb, fixed_bytes,
        mu_per_attr=mu_map()
    )
    result["n_pruned_total"] = n_pruned_total
    result["history"] = history
    with open(os.path.join(out_dir, f"joint-{scene_name}-result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("JOINT TRAINING DONE", flush=True)
    return result


def _quick_val_psnr_cf(splats, color_field, schedule, g, c2w, Ks, width, height,
                       images, val_idcs, device, antialiased, bkgd, mu_per_attr=None):
    import torch

    levels = schedule.levels_at(min(g, 1.0))
    levels_map = {name: levels[j] for j, name in enumerate(ATTR_ORDER)}
    view = quantized_splats(splats, levels_map, mu_per_attr=mu_per_attr)
    psnrs = []
    with torch.no_grad():
        for i in val_idcs:
            r, a, _ = _rasterize_cf(view, color_field, c2w, Ks, width, height, [i], device, antialiased)
            r = _compose_bkgd(r, a, bkgd)
            mse = torch.mean((r - images[i]) ** 2)
            psnrs.append(float(-10.0 * torch.log10(mse)))
    return sum(psnrs) / len(psnrs)


def _probe_psnr_at_levels_cf(splats, color_field, levels_map, c2w, Ks, width, height,
                             images, val_idcs, device, antialiased, bkgd, mu_per_attr=None):
    import torch

    view = quantized_splats(splats, levels_map, mu_per_attr=mu_per_attr)
    psnrs = []
    with torch.no_grad():
        for i in val_idcs:
            r, a, _ = _rasterize_cf(view, color_field, c2w, Ks, width, height, [i], device, antialiased)
            r = _compose_bkgd(r, a, bkgd)
            mse = torch.mean((r - images[i]) ** 2)
            psnrs.append(float(-10.0 * torch.log10(mse)))
    return sum(psnrs) / len(psnrs)


def probe_marginal_cf(splats, color_field, schedule, c2w, Ks, width, height, images,
                      val_idcs, device, antialiased, bkgd, ch, budget_bytes, calib,
                      fixed_bytes, n_probe_views=2, n_steps=2, mu_per_attr=None):
    import math
    import torch

    intl = schedule.integer_targets()
    mn = {name: float(schedule.min_levels[j]) for j, name in enumerate(ATTR_ORDER)}
    n = splats["means"].shape[0]
    vi = list(val_idcs[:n_probe_views])

    base_map = {name: float(intl[name]) for name in ATTR_ORDER}
    base_psnr = _probe_psnr_at_levels_cf(
        splats, color_field, base_map, c2w, Ks, width, height, images, vi,
        device, antialiased, bkgd, mu_per_attr=mu_per_attr
    )
    sur = surrogate_bytes(schedule.levels_at(1.0), n, ch, fixed_bytes)
    rate_real = (sur * calib).item() if calib is not None else sur.item()

    def dbytes(name, lv_old, lv_new):
        return n * ch[name] * math.log2(lv_new / lv_old) / 8.0 * (calib or 1.0)

    moves_up, moves_down = [], []
    for name in ATTR_ORDER:
        lv = intl[name]
        if lv < 4095:
            up_map = dict(base_map)
            up_map[name] = lv + 1
            up_psnr = _probe_psnr_at_levels_cf(
                splats, color_field, up_map, c2w, Ks, width, height, images, vi,
                device, antialiased, bkgd, mu_per_attr=mu_per_attr
            )
            gain = up_psnr - base_psnr
            if gain > 0:
                moves_up.append((name, gain / max(dbytes(name, lv, lv + 1), 1e-9), gain))
        if lv > mn[name] + 1:
            dn_map = dict(base_map)
            dn_map[name] = lv - 1
            dn_psnr = _probe_psnr_at_levels_cf(
                splats, color_field, dn_map, c2w, Ks, width, height, images, vi,
                device, antialiased, bkgd, mu_per_attr=mu_per_attr
            )
            loss = base_psnr - dn_psnr
            if loss > 0:
                moves_down.append((name, loss / max(dbytes(name, lv, lv - 1), 1e-9), loss))

    new_tgt = {name: float(intl[name]) for name in ATTR_ORDER}
    n_moved = 0
    if rate_real > budget_bytes * 1.01 and moves_down:
        for _ in range(n_steps):
            cand = [m for m in moves_down if new_tgt[m[0]] > mn[m[0]] + 1]
            if not cand:
                break
            best = min(cand, key=lambda m: m[1])
            new_tgt[best[0]] = max(new_tgt[best[0]] - 1, mn[best[0]] + 1)
            n_moved += 1
    elif rate_real < budget_bytes * 0.95 and moves_up:
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
# real landing: 4 geometry groups via the serializer + fixed color-field bytes
# ---------------------------------------------------------------------------
def real_landing_cf(splats, color_field, schedule, scene, scene_name, c2w, Ks,
                    width, height, images, val_idcs, device, antialiased, out_dir,
                    budget_mb, fixed_bytes, mu_per_attr=None):
    import numpy as np
    import torch

    from budgetgs.modules.serialize import CODEC_RANGE
    from budgetgs.evaluation.landing import _metrics

    int_levels = schedule.integer_targets()
    level_map = {CONTAINER_NAME[name]: int_levels[name] for name in ATTR_ORDER}
    if mu_per_attr is not None:
        mu_level_map = {CONTAINER_NAME[name]: float(mu_per_attr[name].detach()) for name in ATTR_ORDER}
    else:
        mu_level_map = {CONTAINER_NAME[name]: 0.0 for name in ATTR_ORDER}

    attributes = {}
    for name in ATTR_ORDER:
        cname = CONTAINER_NAME[name]
        x = splats[name].detach()
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        attributes[cname] = x.cpu().numpy().astype(np.float64)

    from directionB_scene import write_mixed_container, dequant_mixed
    tmp_dir = os.path.join(out_dir, "tmp_containers")
    os.makedirs(tmp_dir, exist_ok=True)
    container_path = os.path.join(tmp_dir, f"joint-{scene_name}.bgs")
    groups = tuple((cname, attributes[cname].shape[1]) for cname in
                   ["means", "scales", "quats", "opacity"])
    n_bytes = write_mixed_container(
        container_path, attributes, groups, level_map,
        codec=CODEC_RANGE, mu_map=mu_level_map
    )
    total_bytes = n_bytes + fixed_bytes

    from budgetgs.modules.serialize import read_container
    decoded = read_container(container_path, groups=groups)
    recon = dequant_mixed(decoded, level_map)

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc), flush=True)

    landed_splats = _splats_from_recon_cf(recon, device)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    per_view = []
    for j, i in enumerate(val_idcs):
        lv = _render_views_cf(landed_splats, color_field, scene, [i], device, antialiased)
        m = _metrics(lv, images[i], lpips_model, device)
        per_view.append({"view": int(i), **m})
        for k in totals:
            totals[k] += m[k]
        del lv
    n = len(val_idcs)
    landed = {k: round(v / n, 4) for k, v in totals.items()}

    ref_splats = _splats_from_live_cf(splats, device)
    rt = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for j, i in enumerate(val_idcs):
        rv = _render_views_cf(ref_splats, color_field, scene, [i], device, antialiased)
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
        "color_field_bytes": int(fixed_bytes),
        "geometry_bytes": int(n_bytes),
        "actual_bytes": int(total_bytes),
        "actual_mb": round(total_bytes / 1e6, 4),
        "within_budget": bool(total_bytes <= budget_mb * 1_000_000),
        "landed": landed,
        "ref": ref,
        "per_view": per_view,
        "target_met": bool(total_bytes <= budget_mb * 1_000_000 and landed["psnr"] > 20.0),
    }
    print(
        f"[landed] {total_bytes/1e6:.3f} MB within={result['within_budget']} | "
        f"PSNR {landed['psnr']} SSIM {landed['ssim']} LPIPS {landed['lpips']} "
        f"(ref {ref['psnr']}) | TARGET_MET={result['target_met']}",
        flush=True,
    )
    return result


def _splats_from_recon_cf(recon, device):
    import torch

    return {
        "means": torch.tensor(recon["means"], dtype=torch.float32, device=device),
        "scales": torch.tensor(recon["scales"], dtype=torch.float32, device=device),
        "quats": torch.tensor(recon["quats"], dtype=torch.float32, device=device),
        "opacities": torch.tensor(recon["opacity"], dtype=torch.float32, device=device).reshape(-1),
    }


def _splats_from_live_cf(splats, device):
    return {
        "means": splats["means"].detach(),
        "scales": splats["scales"].detach(),
        "quats": splats["quats"].detach(),
        "opacities": splats["opacities"].detach(),
    }


def _render_views_cf(splats, color_field, scene, val_idcs, device, antialiased=False):
    import torch

    n_views = len(scene.image_paths)
    c2w = torch.tensor(scene.camtoworlds, dtype=torch.float32, device=device).view(n_views, 4, 4)
    Ks = torch.tensor(scene.Ks, dtype=torch.float32, device=device).view(n_views, 3, 3)
    width, height = scene.width, scene.height
    out_views = []
    for i in val_idcs:
        render, alphas, _info = _rasterize_cf(
            splats, color_field, c2w, Ks, width, height, [i], device, antialiased
        )
        composed = _compose_bkgd(render, alphas, torch.ones(3, device=device))
        out_views.append(composed.detach().cpu())
    return torch.cat(out_views, dim=0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="train_colorfield", description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-name", required=True)
    ap.add_argument("--out-dir", default="runs/colorfield")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--warmup", type=int, default=15_000)
    ap.add_argument("--target-n", type=int, default=700_000)
    ap.add_argument("--budget-mb", type=float, default=10.0)
    ap.add_argument("--prune-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--no-position-noise", action="store_true")
    ap.add_argument("--lambda-mode", choices=("dual", "sched"), default="dual")
    ap.add_argument("--lambda-eta", type=float, default=1e-4)
    ap.add_argument("--n-hold-step", type=int, default=None)
    ap.add_argument("--ablate-a", action="store_true")
    ap.add_argument("--ablate-d", action="store_true")
    ap.add_argument("--ablate-c", action="store_true")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--log2-hashmap-size", type=int, default=17)
    ap.add_argument("--view-dependent", action="store_true")
    ap.add_argument("--hash-lr", type=float, default=1e-3)
    ap.add_argument("--relocate", action="store_true", help="P1 mass-conserving prune (merge into nearest survivor)")
    args = ap.parse_args(argv)

    train_colorfield(
        args.scene, args.scene_name, args.out_dir, steps=args.steps,
        warmup=args.warmup, target_n=args.target_n, budget_mb=args.budget_mb,
        prune_every=args.prune_every, seed=args.seed, factor=args.factor,
        position_noise=not args.no_position_noise,
        lambda_mode=args.lambda_mode, lambda_eta=args.lambda_eta,
        n_hold_step=args.n_hold_step,
        qat=not args.ablate_a, rate=not args.ablate_d, probe=not args.ablate_c,
        n_levels=args.n_levels, log2_hashmap_size=args.log2_hashmap_size,
        view_dependent=args.view_dependent, hash_lr=args.hash_lr,
        relocate=args.relocate,
    )


if __name__ == "__main__":
    main()
