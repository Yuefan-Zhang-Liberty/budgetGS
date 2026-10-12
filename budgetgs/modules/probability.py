"""Block-adaptive categorical probability model for the byte ledger.

One Laplace-smoothed categorical distribution per attribute group. ``code_bits``
returns -sum count[s] * log2(p[s]) over the quantized indices, the exact quantity
the serializer pays for with a range coder over the same smoothed cumulative
frequencies, so prediction and reality cannot diverge by more than the coder's
constant overhead.
"""

from __future__ import annotations

import bisect
from collections import Counter
import math


class CategoricalModel:
    """A categorical distribution over quantized symbols, Laplace-smoothed."""

    def __init__(self, counts: dict[int, int], total_symbols: int, levels: int) -> None:
        if total_symbols <= 0:
            raise ValueError("CategoricalModel: no symbols")
        if levels <= 1:
            raise ValueError("CategoricalModel: levels must be >= 2")
        self.counts = dict(counts)
        self.total_symbols = total_symbols
        self.levels = levels
        self._symbols = sorted(self.counts)
        self._total_smoothed = total_symbols + len(self._symbols)
        self._smoothed = {s: self.counts[s] + 1 for s in self._symbols}

    @classmethod
    def from_symbols(cls, symbols: list[int], levels: int) -> "CategoricalModel":
        return cls(dict(Counter(symbols)), len(symbols), levels)

    @property
    def n_distinct(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> list[int]:
        return list(self._symbols)

    def prob(self, symbol: int) -> float:
        return (self.counts.get(symbol, 0) + 1) / self._total_smoothed

    def code_bits(self) -> float:
        """-sum count[s] * log2(p[s]) over every symbol actually seen."""
        return sum(self.counts[s] * (-math.log2(self.prob(s))) for s in self._symbols)

    def code_bytes(self) -> int:
        """Predicted coded size; never zero because a real block costs bytes."""
        return max(1, math.ceil(self.code_bits() / 8.0))

    def raw_counts(self) -> list[tuple[int, int]]:
        """Sorted (symbol, raw count) pairs, as transmitted in the attribute table."""
        return [(s, self.counts[s]) for s in self._symbols]

    def cumulative_freq(self) -> tuple[list[int], list[int], int]:
        """Return (sorted symbols, cumulative smoothed frequencies, total)."""
        cum: list[int] = []
        acc = 0
        for s in self._symbols:
            cum.append(acc)
            acc += self._smoothed[s]
        cum.append(acc)
        return self._symbols, cum, self._total_smoothed

    def index_of(self, symbol: int) -> int:
        """Position of ``symbol`` in the sorted distinct-symbol list."""
        return bisect.bisect_left(self._symbols, symbol)
