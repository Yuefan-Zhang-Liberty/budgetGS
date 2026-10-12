"""Unit tests for the K1 bytes-vs-N analysis (plan Day 5).

Generates synthetic checkpoints (JSON, pure stdlib) with growing Gaussian
counts, runs the full K1 pipeline -- ledger every checkpoint, line fit,
real-encode sampling -- and checks the analysis machinery and the ≤ 1 %
ledger-vs-serializer agreement on them.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budgetgs.evaluation.k1 import _sample_indices, analyze_ckpt_dir, linfit


def _attributes(n: int, seed: int) -> dict[str, list]:
    """n synthetic gaussians with spreads similar to real 3DGS blocks."""
    rng = random.Random(seed)
    spreads = {
        "means": 0.8,
        "scales": 0.02,
        "quats": 0.4,
        "opacity": 0.3,
        "sh0": 0.5,
        "shn": 0.2,
    }
    channels = {"means": 3, "scales": 3, "quats": 4, "opacity": 1, "sh0": 3, "shn": 3}
    return {
        name: [[rng.gauss(0.0, spread) for _ in range(channels[name])] for _ in range(n)]
        for name, spread in spreads.items()
    }


def _write_json_ckpt(path: str, n: int, seed: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_attributes(n, seed), f)


class TestLinFit(unittest.TestCase):
    def test_exact_affine_recovers_slope(self):
        xs = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        ys = [10.0 + 2.5 * x for x in xs]
        fit = linfit(xs, ys)
        self.assertAlmostEqual(fit["slope"], 2.5, places=9)
        self.assertAlmostEqual(fit["intercept"], 10.0, places=6)
        self.assertAlmostEqual(fit["r2"], 1.0, places=9)
        self.assertAlmostEqual(fit["worst_abs_bytes"], 0.0, places=9)

    def test_sample_indices_include_last(self):
        idx = _sample_indices(30, 6)
        self.assertEqual(idx[-1], 29)
        self.assertEqual(len(idx), 6)
        self.assertEqual(len(_sample_indices(5, 10)), 5)


class TestK1Pipeline(unittest.TestCase):
    def test_affine_fit_and_encode_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            sizes = [4000, 8000, 12000, 16000]
            for i, n in enumerate(sizes):
                _write_json_ckpt(os.path.join(tmp, f"ckpt-{n:07d}.json"), n, seed=i)
            report = analyze_ckpt_dir(tmp, levels=1024, sample=2)

            self.assertEqual(report["n_checkpoints"], 4)
            self.assertTrue(report["passes"], f"fit={report['fit']}")
            fit = report["fit"]
            self.assertGreater(fit["r2"], 0.98)
            self.assertLess(fit["worst_pct"], 0.05)
            # synthetic per-symbol entropy is near-constant -> bytes ~= a*N + b
            self.assertGreater(fit["slope"], 0.0)

            self.assertEqual(len(report["samples"]), 2)
            for s in report["samples"]:
                self.assertLess(s["err_pct"], 1.0, f"sample={s}")


if __name__ == "__main__":
    unittest.main()
