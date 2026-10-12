"""Unit tests for the byte ledger (Week-1 core deliverable).

Covers, in order: (1) hand-computed entropy against the categorical model,
(2) quantize round-trip error bounds, (3) range-coder round-trip, (4) the
acceptance test |predicted - actual| / actual <= 1 % between the ledger and the
real serialized file on 10 000 synthetic gaussians, and (5) byte-exact read-back
of the quantized indices. Pure stdlib, so it runs on any python3 without the
conda environment.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budgetgs.modules.ledger import LedgerConfig, compute_ledger
from budgetgs.modules.probability import CategoricalModel
from budgetgs.modules.quantize import ATTRIBUTE_GROUPS, dequantize_channel, quantize_group
from budgetgs.modules.serialize import (
    RangeDecoder,
    RangeEncoder,
    read_container,
    write_container,
)


def _synthetic_attributes(n: int, seed: int) -> dict[str, list]:
    """n synthetic gaussians with per-group spread, mirroring real 3DGS blocks."""
    rng = random.Random(seed)
    spreads = {
        "means": 1.0,
        "scales": 0.02,
        "quats": 0.5,
        "opacity": 0.3,
        "sh0": 0.5,
        "shn": 0.2,
    }
    attributes: dict[str, list] = {}
    for name, channels in ATTRIBUTE_GROUPS:
        attributes[name] = [
            [rng.gauss(0.0, spreads[name]) for _ in range(channels)] for _ in range(n)
        ]
    return attributes


class TestCategoricalModel(unittest.TestCase):
    def test_hand_entropy(self) -> None:
        # symbols [0, 0, 1]: smoothed counts 0->3, 1->2, total 5.
        model = CategoricalModel.from_symbols([0, 0, 1], levels=64)
        expected = 2.0 * (-math.log2(0.6)) + 1.0 * (-math.log2(0.4))
        self.assertAlmostEqual(model.code_bits(), expected, places=9)
        self.assertEqual(model.n_distinct, 2)
        self.assertGreaterEqual(model.code_bytes(), 1)

    def test_code_bytes_never_zero(self) -> None:
        model = CategoricalModel.from_symbols([5, 5, 5], levels=64)
        self.assertEqual(model.code_bytes(), 1)


class TestQuantize(unittest.TestCase):
    def test_roundtrip_within_half_bin(self) -> None:
        rng = random.Random(1)
        n, levels = 500, 256
        values = [[rng.uniform(-2.0, 2.0) for _ in range(3)] for _ in range(n)]
        table, indices = quantize_group(values, levels)
        for c, (lo, hi) in enumerate(zip(table.mins, table.maxs)):
            col = [row[c] for row in values]
            recon = dequantize_channel(indices[c * n:(c + 1) * n], lo, hi, levels)
            half_bin = (hi - lo) / (levels - 1) / 2.0
            for orig, r in zip(col, recon):
                self.assertLessEqual(abs(orig - r), half_bin + 1e-9)


class TestRangeCoder(unittest.TestCase):
    def test_roundtrip_and_entropy_bounded(self) -> None:
        rng = random.Random(7)
        seq = [rng.randrange(6) for _ in range(5000)]
        # the model must be built from the actual symbol stream, otherwise the
        # entropy prediction and the coded size cannot agree
        model = CategoricalModel.from_symbols(seq, levels=200)
        symbols, cum, total = model.cumulative_freq()

        enc = RangeEncoder()
        for sym in seq:
            i = model.index_of(sym)
            enc.encode(cum[i], cum[i + 1], total)
        data = enc.finish()

        dec = RangeDecoder(data)
        out: list[int] = []
        for _ in seq:
            t, r = dec.scaled(total)
            i = 0
            while i + 1 < len(cum) and cum[i + 1] <= t:
                i += 1
            out.append(symbols[i])
            dec.decode(cum[i], cum[i + 1], r)

        self.assertEqual(out, seq)
        self.assertLessEqual(abs(len(data) - model.code_bits() / 8.0), 8.0)


class TestLedgerVsSerialize(unittest.TestCase):
    def test_ledger_predicts_within_one_percent_10k(self) -> None:
        attributes = _synthetic_attributes(10_000, seed=42)
        config = LedgerConfig(levels=4096)
        result = compute_ledger(attributes, config=config)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.bgs")
            actual = write_container(path, attributes, levels=config.levels)
        predicted = result.total_predicted_bytes
        err = abs(predicted - actual) / actual
        self.assertLessEqual(err, 0.01, f"predicted={predicted} actual={actual} err={err:.6f}")
        # the residual must be only the range coder's constant overhead; a
        # larger gap means the ledger and the serializer disagree on layout
        self.assertLessEqual(abs(predicted - actual), 2048)

    def test_read_back_recovers_indices_exactly(self) -> None:
        attributes = _synthetic_attributes(2_000, seed=11)
        config = LedgerConfig(levels=1024)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.bgs")
            write_container(path, attributes, levels=config.levels)
            read = read_container(path)
        for name, _channels in ATTRIBUTE_GROUPS:
            _table, indices = quantize_group(attributes[name], config.levels)
            self.assertEqual(read[name][0], indices)
            self.assertEqual(len(read[name][1].mins), _channels)


if __name__ == "__main__":
    unittest.main()
