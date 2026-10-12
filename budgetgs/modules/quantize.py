"""Attribute-group quantization for BudgetGS.

Per the Week-1 plan (docs/plans/2026-10-08-week1-budgetgs-ramp-up.md, Day 4),
attributes are grouped (means, scales, quats, opacity, sh0, shn); each group is
quantized per channel against a per-block min/max scale table, producing a flat
stream of quantized indices that one block-adaptive categorical distribution per
group will price.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed attribute group layout: (name, channels). The serializer writes groups in
# this order and the ledger prices them in the same order.
ATTRIBUTE_GROUPS: tuple[tuple[str, int], ...] = (
    ("means", 3),
    ("scales", 3),
    ("quats", 4),
    ("opacity", 1),
    ("sh0", 3),
    ("shn", 3),
)


@dataclass(frozen=True)
class ScaleTable:
    """Per-channel min/max quantization bounds (the 'min/max scale table')."""

    mins: tuple[float, ...]
    maxs: tuple[float, ...]


def _channel_bounds(values: list[float], levels: int) -> tuple[float, float]:
    """Min/max for one channel, widened when degenerate so quantization stays valid."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    return lo, hi


def quantize_channel(values: list[float], lo: float, hi: float, levels: int) -> list[int]:
    """Uniformly quantize one channel into ``levels`` integer bins over [lo, hi]."""
    scale = (levels - 1) / (hi - lo)
    return [max(0, min(levels - 1, int(round((v - lo) * scale)))) for v in values]


def dequantize_channel(indices: list[int], lo: float, hi: float, levels: int) -> list[float]:
    """Recover the bin centers a channel's quantized indices stand for."""
    inv = (hi - lo) / (levels - 1)
    return [lo + i * inv for i in indices]


def quantize_group(values, levels: int) -> tuple[ScaleTable, list[int]]:
    """Quantize one attribute group.

    ``values`` is a list of n rows of ``channels`` floats (``values[r][c]``)
    or a numpy array. Returns ``(ScaleTable, flat_indices)`` where
    ``flat_indices`` lists the quantized indices channel by channel (channel 0
    first) — exactly the symbol stream the categorical model prices.

    Numpy-vectorized: a 1.1 M-gaussian checkpoint (50 M symbols) takes seconds
    instead of minutes; ``np.round`` keeps Python's round-half-to-even so the
    indices are bit-identical to the pure-Python path.
    """
    if values is None or (hasattr(values, "__len__") and len(values) == 0):
        raise ValueError("quantize_group: empty attribute block")
    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        channels = arr.shape[1]
        if channels == 0:
            raise ValueError("quantize_group: attribute rows are empty")
        mins: list[float] = []
        maxs: list[float] = []
        flat: list[int] = []
        for c in range(channels):
            col = arr[:, c]
            lo = float(col.min())
            hi = float(col.max())
            if hi - lo < 1e-12:
                hi = lo + 1e-6
            mins.append(lo)
            maxs.append(hi)
            idx = np.rint((col - lo) * ((levels - 1) / (hi - lo))).astype(np.int64)
            np.clip(idx, 0, levels - 1, out=idx)
            flat.extend(idx.tolist())
        return ScaleTable(tuple(mins), tuple(maxs)), flat
    except ImportError:
        pass  # pragma: no cover - numpy is a project dependency

    # pure-python fallback for list/npz/json inputs
    if not isinstance(values[0], (list, tuple)):
        values = [[v] for v in values]
    channels = len(values[0])
    if channels == 0:
        raise ValueError("quantize_group: attribute rows are empty")
    if any(len(row) != channels for row in values):
        raise ValueError("quantize_group: inconsistent row lengths")
    per_channel = [[row[c] for row in values] for c in range(channels)]
    mins: list[float] = []
    maxs: list[float] = []
    flat: list[int] = []
    for c in range(channels):
        lo, hi = _channel_bounds(per_channel[c], levels)
        mins.append(lo)
        maxs.append(hi)
        flat.extend(quantize_channel(per_channel[c], lo, hi, levels))
    return ScaleTable(tuple(mins), tuple(maxs)), flat
