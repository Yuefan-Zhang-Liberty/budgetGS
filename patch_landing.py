#!/usr/bin/env python
"""Patch train_joint.py: stream the real_landing per-view evaluation.

The old code rendered ALL validation views, cat'ed them into one tensor
(landed ~3GB + ref ~3GB for factor=1 full-res) and scored them after the fact.
With --mem=9G this OOM-killed the factor=1 landing (job 1981905). New code
renders one view at a time, scores it and frees it immediately.
"""
P = "/home/stud/zyuef/budgetgs-m0b/train_joint.py"
s = open(P, encoding="utf-8").read()

old = """    landed_splats = _splats_from_recon(recon, device, sh_degree)
    landed_views = _render_views(landed_splats, scene, val_idcs, device, sh_degree, antialiased)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    per_view = []
    for j, i in enumerate(val_idcs):
        m = _metrics(landed_views[j:j + 1], images[i], lpips_model, device)
        per_view.append({"view": int(i), **m})
        for k in totals:
            totals[k] += m[k]
    n = len(val_idcs)
    landed = {k: round(v / n, 4) for k, v in totals.items()}

    # also score the unquantized joint model for reference
    ref_splats = _splats_from_live(splats, device, sh_degree)
    ref_views = _render_views(ref_splats, scene, val_idcs, device, sh_degree, antialiased)
    rt = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for j, i in enumerate(val_idcs):
        m = _metrics(ref_views[j:j + 1], images[i], lpips_model, device)
        for k in rt:
            rt[k] += m[k]
    ref = {k: round(v / n, 4) for k, v in rt.items()}"""

new = """    landed_splats = _splats_from_recon(recon, device, sh_degree)

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
    ref = {k: round(v / n, 4) for k, v in rt.items()}"""

assert old in s, "old landing block not found"
assert s.count(old) == 1, f"expected exactly 1 occurrence, got {s.count(old)}"
s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("patched OK")
