"""Tests for `budgetgs fit` (block-precision bisection with real readback)."""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budgetgs.evaluation.fit import fit_levels  # noqa: E402
from budgetgs.modules.serialize import write_container  # noqa: E402


def synthetic_attributes(n: int, seed: int = 7) -> dict[str, list]:
    """Six attribute groups shaped like a gsplat checkpoint, as nested lists."""
    rng = random.Random(seed)
    groups = {
        "means": [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)],
        "scales": [(10 ** rng.uniform(-3, -1),) * 3 for _ in range(n)],
        "quats": [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)],
        "opacity": [(rng.uniform(-5, 2),) for _ in range(n)],
        "sh0": [(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)) for _ in range(n)],
        "shn": [tuple(rng.uniform(-0.2, 0.2) for _ in range(15)) for _ in range(n)],
    }
    return groups


def real_bytes(attributes: dict[str, list], levels: int, tmpdir: str) -> int:
    fd, path = tempfile.mkstemp(suffix=".bin", dir=tmpdir)
    os.close(fd)
    try:
        return write_container(path, attributes, levels=levels)
    finally:
        os.remove(path)


class TestFitLevels(unittest.TestCase):
    def setUp(self) -> None:
        self.attrs = synthetic_attributes(n=120, seed=3)
        self.tmp = tempfile.mkdtemp(prefix="fit-test-")

    def test_bytes_monotone_in_levels(self) -> None:
        sizes = [real_bytes(self.attrs, levels, self.tmp) for levels in (64, 256, 1024, 4096, 8192)]
        for a, b in zip(sizes, sizes[1:]):
            self.assertGreaterEqual(b, a, f"bytes must be non-decreasing, got {sizes}")

    def test_lands_on_reachable_budget(self) -> None:
        target = real_bytes(self.attrs, 1024, self.tmp)
        result = fit_levels(self.attrs, target, workdir=self.tmp)
        self.assertLessEqual(result.bytes, target, "landing must not exceed budget")
        self.assertLessEqual(result.error_bytes, max(0.01 * target, 256 * 1024))
        self.assertTrue(result.met_tolerance)

    def test_over_budget_at_min_precision(self) -> None:
        result = fit_levels(self.attrs, budget=16, workdir=self.tmp)
        self.assertEqual(result.levels, 2)
        self.assertGreater(result.bytes, result.budget)
        self.assertIn("over-budget", result.status)

    def test_fits_at_max_precision(self) -> None:
        result = fit_levels(self.attrs, budget=100 * 1024 * 1024, workdir=self.tmp)
        self.assertEqual(result.levels, 8192)
        self.assertLessEqual(result.bytes, result.budget)

    def test_reproducible(self) -> None:
        budget = real_bytes(self.attrs, 1024, self.tmp)
        first = fit_levels(self.attrs, budget, workdir=self.tmp)
        second = fit_levels(self.attrs, budget, workdir=self.tmp)
        self.assertEqual(first.levels, second.levels)
        self.assertEqual(first.bytes, second.bytes)
        self.assertEqual(first.evals, second.evals)

    def test_invalid_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_levels(self.attrs, budget=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
