"""Direction B: pin the global block precision on a scene checkpoint and sweep
levels, to isolate the B1 landing-quality collapse root cause.

Questions answered (advisor §7):
  H1 (main cause): global levels forced to 23?  Sweep levels in
      {23, 64, 128, 256, 512, 4096} through the exact production landing path
      (quantize -> zlib real write/read -> dequantize -> render -> score).
  H2 (second cause): shn 3-ch wide-range ruler erases high-order coefficients?
      shn45-independent-channel variants at levels 256/4096.
  Structural ablations (driver-only mixed-level writes; package untouched):
      mix-geo : means=4096, everything else 23 (geometry-only recovery)
      mix-app : shn=4096, everything else 23 (appearance-only recovery)
      mix-plan: plan default per-attribute dict (means1024/scales256/quats256/
                opacity64/sh0 128/shn128) — a data point for direction A.
  Q5 (bytes vs N really linear?): per-group fixed/coded byte breakdown at every
      levels, showing bytes also depend on the precision axis.
  Q4 (look at images): same-view GT / L23 / L256 / L4096 render grids.

Outputs (in --out-dir):
  directionB-<scene>.json, directionB-<scene>.png,
  directionB-<scene>-renders.png, directionB-<scene>-mix-renders.png
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import struct
import tempfile
import time
import zlib

import numpy as np
import torch

from budgetgs.modules import format as fmt
from budgetgs.modules.probability import CategoricalModel
from budgetgs.modules.quantize import (
    ATTRIBUTE_GROUPS,
    mulaw_expand,
    quantize_group,
)
from budgetgs.modules.serialize import (
    CODEC_ZLIB,
    _write_group,
    read_container,
    write_container,
)
from budgetgs.utils.ckpt import load_attributes
from budgetgs.evaluation.landing import (
    _dequant_attributes,
    _metrics,
    _render_views,
    _splats_tensors,
)

SWEEP_LEVELS = [23, 64, 128, 256, 512, 4096]
VARIANT_LEVELS = [256, 4096]
MIX_LEVEL_MAPS = {
    "mix-geo": {"means": 4096, "scales": 23, "quats": 23, "opacity": 23, "sh0": 23, "shn": 23},
    "mix-app": {"means": 23, "scales": 23, "quats": 23, "opacity": 23, "sh0": 23, "shn": 4096},
    "mix-plan": {"means": 1024, "scales": 256, "quats": 256, "opacity": 64, "sh0": 128, "shn": 128},
}
VAL_EVERY = 8
DEVICE = "cuda"
VIEW_POSITIONS = [0, "mid", "last"]  # positions inside val_idcs


def groups_with_shn(channels: int):
    return tuple((n, channels) if n == "shn" else (n, c) for n, c in ATTRIBUTE_GROUPS)


# ---------------------------------------------------------------------------
# shn sub-group splitting
# ---------------------------------------------------------------------------
# The range coder requires total symbols per coded block < 2^24 = 16,777,216
# (TOP): after renormalization range >= TOP, and if total >= TOP a leftover
# range in [TOP, total) makes `range //= total` == 0, range 0, and the
# renormalize loop spins forever (observed: shn 45ch x 375k = 16.875M symbols
# hung the 1975189 landing; test_groups shn exit=124; fixed by splitting shn
# into 15 blocks of 3 channels, each < 2^24 symbols).
SHN_SPLIT = 3  # channels per shn sub-block

def split_shn_groups(groups):
    """Expand a ("shn", 45) group into SHN_SPLIT-channel sub-groups named
    shn#k. Other groups pass through unchanged."""
    out = []
    for name, channels in groups:
        if name == "shn" and channels > SHN_SPLIT:
            k = channels // SHN_SPLIT
            for i in range(k):
                out.append((f"shn#{i}", SHN_SPLIT))
        else:
            out.append((name, channels))
    return tuple(out)


def split_attributes(attributes, groups):
    """Return a copy of ``attributes`` with shn split into shn#k blocks, and
    the expanded groups to write with. ``groups`` is the caller's list."""
    wgroups = split_shn_groups(groups)
    attrs = dict(attributes)
    if any(n.startswith("shn#") for n, _ in wgroups):
        shn = attrs["shn"]
        arr = np.asarray(shn, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        del attrs["shn"]
        for i in range(arr.shape[1] // SHN_SPLIT):
            attrs[f"shn#{i}"] = arr[:, i * SHN_SPLIT:(i + 1) * SHN_SPLIT]
    return attrs, wgroups


# ---------------------------------------------------------------------------
# byte composition (exact replica of the writer's per-group accounting)
# ---------------------------------------------------------------------------
def byte_breakdown(attributes, groups, level_map):
    """Return header bytes + per-group (fixed_bytes, coded_bytes, n_symbols),
    computed exactly the way serialize._write_group lays bytes out."""
    groups_rows = []
    pos = 28  # header size (magic 8 + 5 uint32)
    for name, channels in groups:
        lv = level_map[name]
        scale_table, indices = quantize_group(attributes[name], lv)
        seg = bytearray()
        nb = name.encode("utf-8")
        seg += struct.pack("<I", len(nb)) + nb
        seg += struct.pack("<I", channels)
        for lo, hi in zip(scale_table.mins, scale_table.maxs):
            seg += struct.pack("<dd", lo, hi)
        seg += struct.pack("<Q", len(indices))
        seg += struct.pack("<I", CODEC_ZLIB)
        packed = array.array("I", indices).tobytes()
        coded = zlib.compress(packed, level=6)
        coded_len = len(coded)
        seg += struct.pack("<I", coded_len)
        total_after = pos + len(seg) + coded_len
        padding = (fmt.ALIGNMENT - (total_after % fmt.ALIGNMENT)) % fmt.ALIGNMENT
        fixed = len(seg) + padding  # coded_len is not in seg yet; fields+padding
        groups_rows.append(
            {
                "group": name,
                "channels": channels,
                "levels": lv,
                "fixed_bytes": fixed,
                "coded_bytes": coded_len,
                "n_symbols": len(indices),
            }
        )
        pos += fixed + coded_len
    return 28, groups_rows


def write_mixed_container(path, attributes, groups, level_map, codec=CODEC_ZLIB,
                          mu_map=None):
    """Driver-only variant of write_container that accepts a per-group levels
    dict and an optional per-group mu-law ``mu_map`` (missing/0 = uniform).
    The header levels field is a placeholder (the readers do not use it).
    shn groups wider than SHN_SPLIT are automatically split into sub-blocks
    (see split_shn_groups) so every range-coded block stays below 2^24
    symbols."""
    attrs, wgroups = split_attributes(attributes, groups)
    n_gaussians = len(attrs[wgroups[0][0]])
    out = bytearray()
    out += fmt.MAGIC
    out += struct.pack("<IIIII", fmt.VERSION, n_gaussians, len(wgroups), 0, codec)
    for name, channels in wgroups:
        base = name.split("#")[0] if name.startswith("shn#") else name
        lv = level_map[base]
        mu = (mu_map or {}).get(base, 0.0)
        scale_table, indices = quantize_group(attrs[name], lv, mu=mu)
        model = CategoricalModel.from_symbols(indices, lv)
        _write_group(out, name, channels, scale_table, indices, model, codec=codec)
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def dequant_mixed(decoded, level_map):
    """Dequantize a decoded container where each group may have its own levels.
    shn sub-blocks (shn#k) are stitched back into a single (n, 45) array."""
    shn_parts = []
    out = {}
    for name, (indices, scale_table) in decoded.items():
        base = name.split("#")[0] if name.startswith("shn#") else name
        lv = int(level_map[base])
        mu = getattr(scale_table, "mu", 0.0)
        channels = len(scale_table.mins)
        n = len(indices) // channels
        idx = np.asarray(indices, dtype=np.float64)
        rows = np.empty((n, channels), dtype=np.float64)
        for c in range(channels):
            lo, hi = scale_table.mins[c], scale_table.maxs[c]
            vq = idx[c * n: (c + 1) * n] / (lv - 1) if lv > 1 else 0.0
            uq = mulaw_expand(vq, mu)
            rows[:, c] = lo + uq * (hi - lo)
        if name.startswith("shn#"):
            shn_parts.append((int(name.split("#")[1]), rows))
        else:
            out[name] = rows
    if shn_parts:
        shn_parts.sort(key=lambda kv: kv[0])
        out["shn"] = np.hstack([rows for _, rows in shn_parts])
    return out


def real_roundtrip(attributes, groups, levels, tmp_dir, level_map=None, tag="x"):
    """Real zlib write/read round trip. level_map=None -> uniform ``levels``
    through the production write_container; else the driver mixed writer."""
    path = os.path.join(tmp_dir, f"probe-{tag}.bgs")
    try:
        t0 = time.time()
        if level_map is None:
            n_bytes = write_container(path, attributes, groups=groups, levels=levels, codec=CODEC_ZLIB)
            rgroups = groups
        else:
            n_bytes = write_mixed_container(path, attributes, groups, level_map, codec=CODEC_ZLIB)
            rgroups = split_shn_groups(groups)
        dt = time.time() - t0
        decoded = read_container(path, groups=rgroups)
        if level_map is None:
            recon = _dequant_attributes(decoded, levels, attributes)
        else:
            recon = dequant_mixed(decoded, level_map)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return n_bytes, recon, dt


def group_errors(orig, recon):
    out = {}
    for name, a0 in orig.items():
        a = np.asarray(a0, dtype=np.float64)
        b = np.asarray(recon[name], dtype=np.float64)
        if a.shape != b.shape:
            mse = float(np.mean((a.reshape(-1) - b.reshape(-1)) ** 2))
            shape_note = f"{a.shape} vs {b.shape}"
        else:
            mse = float(np.mean((a - b) ** 2))
            shape_note = str(a.shape)
        out[name] = {
            "shape": shape_note,
            "mse": mse,
            "rmse": float(math.sqrt(mse)) if mse >= 0 else float("nan"),
        }
    return out


def shn_order_stats(orig_shn, recon_shn):
    a = np.asarray(orig_shn, dtype=np.float64).reshape(-1, 15, 3)
    b = np.asarray(recon_shn, dtype=np.float64).reshape(-1, 15, 3)
    mse_list, norm_list = [], []
    for k in range(15):
        mse = float(np.mean((a[:, k, :] - b[:, k, :]) ** 2))
        std = float(np.std(a[:, k, :]))
        mse_list.append(mse)
        norm_list.append(float(math.sqrt(mse)) / std if std > 1e-12 else float("nan"))
    return mse_list, norm_list


def score_recon(splats, scene, val_idcs, images, lpips_model, sh_degree):
    """Returns (avg metrics, per-view metrics, all rendered views on CPU)."""
    rec_views = _render_views(splats, scene, val_idcs, DEVICE, sh_degree)
    per_view = []
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for j, i in enumerate(val_idcs):
        m = _metrics(rec_views[j: j + 1], images[i], lpips_model, DEVICE)
        per_view.append({"view": int(i), **m})
        for k in totals:
            totals[k] += m[k]
    n = max(len(val_idcs), 1)
    avg = {k: round(v / n, 4) for k, v in totals.items()}
    return avg, per_view, rec_views


def main():
    ap = argparse.ArgumentParser(prog="directionB_scene")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-name", default=None,
                    help="label for output file names/titles; default = scene dir basename")
    ap.add_argument("--out-dir", default="runs/analysis")
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    args = ap.parse_args()

    scene_name = args.scene_name or os.path.basename(os.path.normpath(args.scene))
    print(f"scene_name={scene_name}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "tmp_containers")
    os.makedirs(tmp_dir, exist_ok=True)

    # ---- load ---------------------------------------------------------------
    attributes = load_attributes(args.ckpt)
    n_g = len(attributes["means"])
    loaded_shapes = {k: list(np.asarray(v).shape) for k, v in attributes.items()}
    print("loaded attribute shapes:", loaded_shapes, flush=True)

    raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        raw_ckpt = raw_ckpt["state_dict"]
    cfg = raw_ckpt.get("config", {}) if isinstance(raw_ckpt, dict) else {}
    sh_degree = int(cfg.get("sh_degree", 3)) if isinstance(cfg, dict) else 3
    k_orders = (sh_degree + 1) ** 2 - 1
    print(f"n_gaussians={n_g} sh_degree={sh_degree} shn_orders={k_orders}", flush=True)

    raw_shn = None
    for key in ("shn", "sh_rest", "shN", "features_rest"):
        if key in raw_ckpt:
            v = raw_ckpt[key]
            raw_shn = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            break
    raw_shn = np.asarray(raw_shn, dtype=np.float64)
    print("raw shn shape:", list(raw_shn.shape), flush=True)

    groups3 = groups_with_shn(3)
    groups45 = groups_with_shn(45)
    attributes45 = dict(attributes)
    attributes45["shn"] = raw_shn.reshape(n_g, k_orders * 3)

    # ---- scene / views / ref -------------------------------------------------
    from budgetgs.data.colmap import load_360_scene
    from budgetgs.modules.trainer import _load_image

    scene = load_360_scene(args.scene, factor=2)
    n_views = len(scene.image_paths)
    val_idcs = list(range(0, n_views, args.val_every))
    width, height = scene.width, scene.height
    print(f"scene views={n_views}, val views={len(val_idcs)}, size={width}x{height}", flush=True)
    images = {i: _load_image(scene.image_paths[i], DEVICE, size=(width, height)) for i in val_idcs}

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()
    except Exception as exc:
        print("LPIPS unavailable:", repr(exc), flush=True)

    torch.manual_seed(0)
    ref_splats = _splats_tensors(attributes, DEVICE, sh_degree)
    ref_metrics, ref_per_view, ref_views = score_recon(
        ref_splats, scene, val_idcs, images, lpips_model, sh_degree
    )
    print("REF metrics:", ref_metrics, flush=True)
    del ref_splats
    torch.cuda.empty_cache()

    rows = []
    stored_views = {}  # tag -> rendered views (CPU), for image grids
    STORE_TAGS = {"shn3ch-L23", "shn3ch-L256", "shn3ch-L4096", "mix-plan"}

    def run_one(tag, layout, groups, attrs, levels=None, level_map=None, store=False):
        n_bytes, recon, dt = real_roundtrip(
            attrs, groups, levels if levels is not None else 0, tmp_dir,
            level_map=level_map, tag=tag,
        )
        errs = group_errors(attrs, recon)
        order_mse, order_norm = shn_order_stats(attrs["shn"], recon["shn"])
        header_b, breakdown = byte_breakdown(attrs, groups, level_map or {g[0]: levels for g in groups})
        fixed_total = header_b + sum(b["fixed_bytes"] for b in breakdown)
        coded_total = sum(b["coded_bytes"] for b in breakdown)
        rec_splats = _splats_tensors(recon, DEVICE, sh_degree)
        metrics, per_view, views = score_recon(
            rec_splats, scene, val_idcs, images, lpips_model, sh_degree
        )
        if store:
            stored_views[tag] = views.detach().cpu()
        del rec_splats, views
        torch.cuda.empty_cache()
        row = {
            "tag": tag,
            "layout": layout,
            "levels": levels if levels is not None else level_map,
            "actual_bytes": int(n_bytes),
            "actual_mb": round(n_bytes / 1e6, 4),
            "write_seconds": round(dt, 2),
            "landed": metrics,
            "per_view": per_view,
            "group_rmse": {k: round(v["rmse"], 8) for k, v in errs.items()},
            "group_mse": {k: v["mse"] for k, v in errs.items()},
            "shn_order_mse": order_mse,
            "shn_order_rmse_over_std": order_norm,
            "byte_fixed": int(fixed_total),
            "byte_coded": int(coded_total),
            "byte_fixed_pct": round(100.0 * fixed_total / n_bytes, 2),
            "byte_coded_pct": round(100.0 * coded_total / n_bytes, 2),
            "byte_breakdown": breakdown,
        }
        rows.append(row)
        lm = levels if levels is not None else "mixed"
        print(
            f"[{tag}] {n_bytes/1e6:.3f} MB (fixed {fixed_total/1e6:.3f} / "
            f"coded {coded_total/1e6:.3f}) write={dt:.1f}s | "
            f"PSNR {metrics['psnr']} SSIM {metrics['ssim']} LPIPS {metrics['lpips']} | "
            f"means RMSE {errs['means']['rmse']:.4f} shn RMSE {errs['shn']['rmse']:.2e}",
            flush=True,
        )
        return row

    # ---- H1 sweep: production 3-ch layout, pinned levels --------------------
    for L in SWEEP_LEVELS:
        run_one(
            f"shn3ch-L{L}", "shn3ch", groups3, attributes, levels=L,
            store=f"shn3ch-L{L}" in STORE_TAGS,
        )

    # ---- H2: shn 45 independent channels at 256 / 4096 ----------------------
    for L in VARIANT_LEVELS:
        run_one(f"shn45ch-L{L}", "shn45ch", groups45, attributes45, levels=L)

    # ---- structural ablations (driver-only mixed levels) -------------------
    for tag, lmap in MIX_LEVEL_MAPS.items():
        run_one(tag, tag, groups3, attributes, level_map=dict(lmap), store=tag in STORE_TAGS)

    # ---- baseline check ------------------------------------------------------
    base = next(r for r in rows if r["tag"] == "shn3ch-L23")
    # Generic collapse reproduction: pinned L23 is expected to be badly broken.
    # The exact room value (~5.33 dB) is room-specific, so just flag collapse.
    l23_psnr = base["landed"]["psnr"]
    collapse_reproduced = l23_psnr < 12.0
    print(
        f"BASELINE CHECK L23 PSNR={l23_psnr} -> "
        f"{'collapse reproduced (<12 dB)' if collapse_reproduced else 'no collapse - investigate'}",
        flush=True,
    )

    payload = {
        "scene": scene_name,
        "ckpt": os.path.basename(args.ckpt),
        "ckpt_path": args.ckpt,
        "scene_dir": args.scene,
        "n_gaussians": n_g,
        "sh_degree": sh_degree,
        "val_every": args.val_every,
        "val_views": len(val_idcs),
        "val_idcs": val_idcs,
        "loaded_attribute_shapes": loaded_shapes,
        "raw_shn_shape": list(raw_shn.shape),
        "budget_mb": 10.0,
        "ref": ref_metrics,
        "ref_per_view": ref_per_view,
        "rows": rows,
        "l23_collapse_reproduced": collapse_reproduced,
    }
    json_path = os.path.join(args.out_dir, f"directionB-{scene_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("JSON written:", json_path, flush=True)

    # ---- curve plot ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s3 = [r for r in rows if r["layout"] == "shn3ch"]
    s45 = [r for r in rows if r["layout"] == "shn45ch"]
    mixes = [r for r in rows if r["layout"].startswith("mix-")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot([r["levels"] for r in s3], [r["landed"]["psnr"] for r in s3],
            "o-", color="#1f77b4", label="shn 3ch (production)")
    ax.plot([r["levels"] for r in s45], [r["landed"]["psnr"] for r in s45],
            "s--", color="#d62728", label="shn 45ch variant")
    ax.axhline(ref_metrics["psnr"], color="green", ls=":", label=f"factor2 ref {ref_metrics['psnr']:.2f} dB")
    ax.axhline(20.0, color="orange", ls=":", label="advisor first-iter floor 20 dB")
    for r in s3:
        ax.annotate(f"{r['landed']['psnr']:.2f}", (r["levels"], r["landed"]["psnr"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for r in mixes:
        ax.axhline(r["landed"]["psnr"], color="purple", ls="--", alpha=0.45)
        ax.text(24, r["landed"]["psnr"] + 0.4, f"{r['tag']} {r['landed']['psnr']:.2f} dB "
                f"({r['actual_mb']:.1f} MB)", fontsize=7.5, color="purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("global quant levels (pinned)")
    ax.set_ylabel("landed PSNR (dB)")
    ax.set_title(f"{scene_name} landed PSNR vs global levels")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([r["levels"] for r in s3], [r["actual_mb"] for r in s3],
            "o-", color="#1f77b4", label="total bytes")
    ax.plot([r["levels"] for r in s3], [r["byte_coded"] / 1e6 for r in s3],
            "^--", color="#2ca02c", label="coded (precision-dependent)")
    ax.plot([r["levels"] for r in s3], [r["byte_fixed"] / 1e6 for r in s3],
            "x:", color="#7f7f7f", label="fixed overhead")
    ax.plot([r["levels"] for r in s45], [r["actual_mb"] for r in s45],
            "s--", color="#d62728", label="shn 45ch variant")
    ax.axhline(10.0, color="black", ls=":", label="10 MB budget")
    for r in mixes:
        ax.axhline(r["actual_mb"], color="purple", ls="--", alpha=0.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("global quant levels (pinned)")
    ax.set_ylabel("container bytes (MB, zlib)")
    ax.set_title(f"{scene_name} bytes vs levels (fixed vs coded)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"directionB-{scene_name}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print("PNG written:", png_path, flush=True)

    # ---- render comparison grids (Q4: must look at images) ------------------
    def pos_to_idx(pos):
        if pos == "mid":
            return len(val_idcs) // 2
        if pos == "last":
            return len(val_idcs) - 1
        return pos

    def make_grid(tag_list, columns, fname, title_prefix):
        pos_list = [pos_to_idx(p) for p in VIEW_POSITIONS]
        fig, axes = plt.subplots(len(pos_list), len(columns), figsize=(4.2 * len(columns), 3.4 * len(pos_list)))
        if len(pos_list) == 1:
            axes = axes[None, :]
        for ri, vp in enumerate(pos_list):
            i = val_idcs[vp]
            col = 0
            for kind, tag in columns:
                ax = axes[ri, col]
                if kind.upper() == "GT":
                    img = images[i].detach().cpu().numpy()[0]
                    psnr_txt = ""
                else:
                    views = stored_views[tag]
                    img = views[vp].numpy()
                    row = next(r for r in rows if r["tag"] == tag)
                    pv = next(p for p in row["per_view"] if p["view"] == i)
                    psnr_txt = f" PSNR {pv['psnr']:.2f}"
                ax.imshow(np.clip(img, 0, 1))
                ax.set_xticks([])
                ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"{kind}{psnr_txt}", fontsize=9)
                else:
                    ax.set_ylabel(f"view {i}\n{psnr_txt}", fontsize=8)
                col += 1
        fig.suptitle(f"{scene_name} {title_prefix} — rows: representative val views; col: GT / landed", fontsize=11)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("PNG written:", out, flush=True)

    make_grid(
        None,
        [("GT", None), ("L23", "shn3ch-L23"), ("L256", "shn3ch-L256"), ("L4096", "shn3ch-L4096")],
        f"directionB-{scene_name}-renders.png",
        "GT vs landed at global levels 23/256/4096",
    )
    if "mix-plan" in stored_views:
        make_grid(
            None,
            [("GT", None), ("L23", "shn3ch-L23"), ("mix-plan", "mix-plan"), ("L4096", "shn3ch-L4096")],
            f"directionB-{scene_name}-mix-renders.png",
            "GT vs uniform-23 / per-attribute plan / uniform-4096",
        )

    print("DIRECTION B DONE", flush=True)


if __name__ == "__main__":
    main()
