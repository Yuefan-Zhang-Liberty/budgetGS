"""Compare shn block layouts 3-channel vs 45-channel on a real 30k checkpoint.

Both layouts serialize the same 45 SH coefficients per gaussian; the only
difference is the block structure (scale table + one categorical model per
block).  Reports ledger (range-priced) bytes, real range bytes and real zlib
bytes for each layout, with per-group breakdowns.
"""

from __future__ import annotations

import argparse
import json
import tempfile

import numpy as np

from ..modules.ledger import LedgerConfig, compute_ledger
from ..modules.serialize import CODEC_RANGE, CODEC_ZLIB, read_container, write_container
from ..utils.ckpt import load_attributes

GROUPS3 = (
    ("means", 3), ("scales", 3), ("quats", 4), ("opacity", 1),
    ("sh0", 3), ("shn", 3),
)
GROUPS45 = (
    ("means", 3), ("scales", 3), ("quats", 4), ("opacity", 1),
    ("sh0", 3), ("shn", 45),
)


def _build(values3: dict, values45: dict, attributes: dict) -> tuple[dict, dict]:
    shn = attributes["shn"]
    if isinstance(shn, np.ndarray):
        n = shn.shape[0]
        k3 = shn.shape[1] * shn.shape[2] if shn.ndim == 3 else shn.shape[1] // 3
        v3 = shn.reshape(-1, 3)
        v45 = shn.reshape(n, -1)
    else:
        n = len(shn)
        v45 = np.asarray(shn, dtype=np.float64).reshape(n, -1)
        v3 = v45.reshape(-1, 3)
    for name, val in attributes.items():
        if name == "shn":
            continue
        if isinstance(val, np.ndarray):
            values3[name] = val.reshape(-1, 1) if val.ndim == 1 else val
            values45[name] = val.reshape(-1, 1) if val.ndim == 1 else val
        else:
            arr = np.asarray(val, dtype=np.float64)
            values3[name] = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            values45[name] = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    values3["shn"] = v3
    values45["shn"] = v45
    return values3, values45


def _run(attributes3: dict, attributes45: dict, levels: int, n_gaussians: int,
         zlib: bool = True) -> dict:
    config = LedgerConfig(levels=levels)
    out = {"levels": levels}
    # Real range coding is O(symbols) in pure python; on >150k gaussians it
    # takes minutes, while the ledger is exact to <1% and zlib is fast. Run
    # range only on small checkpoints, mark it "ledger-approx" otherwise.
    range_ok = n_gaussians <= 150_000
    for tag, groups, attrs in (
        ("shn3", GROUPS3, attributes3),
        ("shn45", GROUPS45, attributes45),
    ):
        result = compute_ledger(attrs, groups=groups, config=config)
        per = result.per_attribute_bytes()
        n_range = None
        n_zlib = None
        if zlib or range_ok:
            with tempfile.NamedTemporaryFile(suffix=".bgs", delete=False) as tmp:
                p = tmp.name
            try:
                if zlib:
                    n_zlib = write_container(p, attrs, groups=groups, levels=levels, codec=CODEC_ZLIB)
                if range_ok:
                    n_range = write_container(p, attrs, groups=groups, levels=levels, codec=CODEC_RANGE)
                    _ = read_container(p, groups=groups)  # round-trip sanity
            finally:
                import os
                os.unlink(p)
        out[tag] = {
            "ledger_bytes": result.total_predicted_bytes,
            "range_bytes": n_range,
            "zlib_bytes": n_zlib,
            "shn_ledger_bytes": per["shn"],
            "range_minus_ledger_pct": (
                round((n_range - result.total_predicted_bytes) / result.total_predicted_bytes * 100.0, 4)
                if n_range is not None else None
            ),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--levels", type=int, default=4096)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-zlib", action="store_true", help="ledger only (fast)")
    args = parser.parse_args(argv)

    attributes = load_attributes(args.ckpt)
    attributes3: dict = {}
    attributes45: dict = {}
    attributes3, attributes45 = _build(attributes3, attributes45, attributes)

    report = _run(attributes3, attributes45, args.levels, len(attributes["means"]),
                  zlib=not args.skip_zlib)
    report["n_gaussians"] = len(attributes["means"])
    report["shn_symbols_per_block_3"] = 15  # N*15 rows x 3 channels
    report["shn_symbols_per_block_45"] = 1  # N rows x 45 channels
    for tag in ("shn3", "shn45"):
        r = report[tag]
        rng = f"range={r['range_bytes']:,}B" if r["range_bytes"] is not None else "range=ledger-approx"
        zlb = f"zlib={r['zlib_bytes']:,}B" if r["zlib_bytes"] is not None else "zlib=skip"
        print(
            f"{tag}: ledger={r['ledger_bytes']:,}B {rng} {zlb} | "
            f"shn block={r['shn_ledger_bytes']:,}B | "
            f"range-ledger gap {r['range_minus_ledger_pct']}"
        )
    d45 = report['shn45']['ledger_bytes'] - report['shn3']['ledger_bytes']
    print(
        f"ledger: shn45 vs shn3 = {d45:,}B "
        f"({d45 / report['shn3']['ledger_bytes'] * 100:.3f}%)"
    )
    if report["shn45"]["range_bytes"] is not None:
        d45r = report['shn45']['range_bytes'] - report['shn3']['range_bytes']
        print(
            f"range: shn45 vs shn3 = {d45r:,}B "
            f"({d45r / report['shn3']['range_bytes'] * 100:.3f}%)"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
