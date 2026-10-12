"""K1: is the serialized size affine in the Gaussian count N?

Runs the byte ledger over every checkpoint of a run, regresses predicted
bytes on N (closed-form line fit, pure python), reports R^2 and the worst
residual, and re-encodes a sample of checkpoints with the real serializer to
verify the ledger. Criterion (docs/02-research-program.md, L672-679):

    a linear fit of bytes on N with R^2 > 0.98 and a worst residual under
    5 % of the file -> bytes are affine enough that a parameter controller
    ports with one exchange rate.

Usage::

    python -m budgetgs.evaluation.k1 --ckpt-dir runs/bicycle --ckpt-dir runs/garden \\
        --sample 6 [--levels 4096] [--out k1-report.json]

The script needs no torch unless the checkpoints are ``.pt`` files; ``.npz``
and ``.json`` checkpoints run on the system python3 (used for bring-up tests).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile

from ..modules.ledger import LedgerConfig, compute_ledger
from ..modules.serialize import write_container
from ..utils.ckpt import load_attributes


def linfit(xs: list[float], ys: list[float]) -> dict:
    """Closed-form least squares line ``y = slope * x + intercept``.

    Returns slope, intercept, R^2, the worst absolute residual in bytes, and
    the worst residual as a fraction of the largest file.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("linfit: need at least two points")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        raise ValueError("linfit: degenerate x (constant N)")
    slope = cov / var
    intercept = my - slope * mx
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    worst = max(abs(y - yh) for y, yh in zip(ys, yhat))
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "worst_abs_bytes": worst,
        "worst_pct": worst / max(ys) if ys else float("nan"),
    }


def _ckpt_files(ckpt_dir: str) -> list[str]:
    """All checkpoint files in ``ckpt_dir`` sorted by step number."""
    names = [
        f for f in os.listdir(ckpt_dir)
        if f.lower().endswith((".pt", ".npz", ".json")) and "ckpt" in f.lower()
    ]
    names.sort(key=lambda f: int("".join(ch for ch in f if ch.isdigit()) or 0))
    return [os.path.join(ckpt_dir, f) for f in names]


def _sample_indices(n: int, k: int) -> list[int]:
    """``k`` evenly spaced indices over ``[0, n)``, always including the last."""
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def analyze_ckpt_dir(
    ckpt_dir: str, levels: int = 4096, sample: int = 6
) -> dict:
    """Ledger every checkpoint, fit bytes-on-N, and real-encode a sample."""
    files = _ckpt_files(ckpt_dir)
    if not files:
        raise FileNotFoundError(f"analyze_ckpt_dir: no checkpoints in {ckpt_dir!r}")
    config = LedgerConfig(levels=levels)

    rows: list[dict] = []
    for i, path in enumerate(files):
        print(f"  ledger {i + 1}/{len(files)} {os.path.basename(path)}", flush=True)
        attributes = load_attributes(path)
        result = compute_ledger(attributes, config=config)
        n = len(attributes["means"])
        rows.append(
            {
                "ckpt": os.path.basename(path),
                "n_gaussians": n,
                "predicted_bytes": result.total_predicted_bytes,
            }
        )

    fit = linfit([r["n_gaussians"] for r in rows], [r["predicted_bytes"] for r in rows])

    samples: list[dict] = []
    for i in _sample_indices(len(rows), sample):
        print(f"  real-encode {os.path.basename(files[i])}", flush=True)
        path = files[i]
        attributes = load_attributes(path)
        predicted = compute_ledger(attributes, config=config).total_predicted_bytes
        with tempfile.NamedTemporaryFile(suffix=".bgs", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            actual = write_container(tmp_path, attributes, levels=levels)
        finally:
            os.unlink(tmp_path)
        err = abs(predicted - actual) / actual if actual else 0.0
        samples.append(
            {
                "ckpt": rows[i]["ckpt"],
                "n_gaussians": rows[i]["n_gaussians"],
                "predicted": predicted,
                "actual": actual,
                "err_pct": round(err * 100.0, 4),
            }
        )

    return {
        "ckpt_dir": ckpt_dir,
        "n_checkpoints": len(rows),
        "fit": fit,
        "passes": fit["r2"] > 0.98 and fit["worst_pct"] < 0.05,
        "samples": samples,
    }


def _fmt(report: dict) -> str:
    f = report["fit"]
    lines = [
        f"{report['ckpt_dir']}: {report['n_checkpoints']} checkpoints",
        f"  fit: bytes = {f['slope']:.4f} * N + {f['intercept']:.1f}",
        f"  R^2 = {f['r2']:.5f} | worst residual = {f['worst_abs_bytes']:.1f} B "
        f"({f['worst_pct'] * 100:.3f} % of file)",
        f"  criterion (R^2 > 0.98 and residual < 5 %): {'PASS' if report['passes'] else 'FAIL'}",
    ]
    if report["samples"]:
        lines.append("  ledger-vs-serializer sample check:")
        for s in report["samples"]:
            lines.append(
                f"    {s['ckpt']}: N={s['n_gaussians']} predicted={s['predicted']} "
                f"actual={s['actual']} err={s['err_pct']:.3f} %"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="budgetgs k1", description=__doc__)
    parser.add_argument("--ckpt-dir", action="append", required=True, metavar="DIR",
                        help="checkpoint directory of one run (repeatable)")
    parser.add_argument("--levels", type=int, default=4096)
    parser.add_argument("--sample", type=int, default=6, help="checkpoints to real-encode")
    parser.add_argument("--out", default=None, help="write a JSON report here")
    args = parser.parse_args(argv)

    reports = [analyze_ckpt_dir(d, levels=args.levels, sample=args.sample) for d in args.ckpt_dir]
    for r in reports:
        print(_fmt(r))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
