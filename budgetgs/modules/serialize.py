"""Real serializer for BudgetGS checkpoints.

Writes exactly the container the ledger predicts: header, per-group
name/channels/scale table/probability table, then range-coded quantized indices
using the same smoothed cumulative frequencies the ledger priced. Returns the
number of bytes actually written; the reader recovers the quantized indices
byte-exactly.

The range coder is a classic carryless 32-bit coder (LZMA-style) with a static,
transmitted frequency table per group, so its output tracks the ledger's entropy
prediction to within a small constant overhead.
"""

from __future__ import annotations

import array
import bisect
import struct
import zlib

from . import format as fmt
from .probability import CategoricalModel
from .quantize import ATTRIBUTE_GROUPS, quantize_group, ScaleTable

# Per-group codec id in the container: 0 = range coder (entropy-tight, the
# ledger prices it), 1 = zlib (fast feedback path for fit bisection).
CODEC_RANGE = 0
CODEC_ZLIB = 1


class RangeEncoder:
    """Carryless 32-bit range encoder (LZMA-style)."""

    TOP = 1 << 24
    MASK = 0xFFFFFFFF

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()
        self.cache = 0
        self.cache_size = 1

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > self.MASK:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & self.MASK

    def encode(self, cum_low: int, cum_high: int, total: int) -> None:
        self.range //= total
        self.low += cum_low * self.range
        self.range *= cum_high - cum_low
        while self.range < self.TOP:
            self._shift_low()
            self.range = (self.range << 8) & self.MASK

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    """Matching range decoder for :class:`RangeEncoder`."""

    TOP = 1 << 24
    MASK = 0xFFFFFFFF

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        # The encoder's stream begins with its initial cache byte (0x00), so the
        # decoder pre-reads five bytes to align ``code`` with the encoder's low.
        for _ in range(5):
            self.code = ((self.code << 8) | self._read()) & self.MASK

    def _read(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def scaled(self, total: int) -> tuple[int, int]:
        """Return (t, r) with r = range // total and t = code // r, clamped."""
        r = self.range // total
        t = self.code // r
        if t >= total:
            t = total - 1
        return t, r

    def decode(self, cum_low: int, cum_high: int, r: int) -> None:
        self.code -= cum_low * r
        self.range = r * (cum_high - cum_low)
        while self.range < self.TOP:
            self.code = ((self.code << 8) | self._read()) & self.MASK
            self.range = (self.range << 8) & self.MASK


def _encode_symbols(
    symbols: list[int], cum: list[int], total: int, indices, enc: RangeEncoder
) -> None:
    try:
        import numpy as np

        i_arr = np.searchsorted(np.asarray(symbols, dtype=np.int64), indices)
        for i in i_arr:
            enc.encode(cum[i], cum[i + 1], total)
        return
    except ImportError:
        pass  # pure-python fallback below
    for sym in indices:
        i = bisect.bisect_left(symbols, sym)
        enc.encode(cum[i], cum[i + 1], total)


def _decode_symbols(
    data: bytes, symbols: list[int], cum: list[int], total: int, n_symbols: int
) -> list[int]:
    dec = RangeDecoder(data)
    out: list[int] = []
    for _ in range(n_symbols):
        t, r = dec.scaled(total)
        i = bisect.bisect_right(cum, t) - 1
        if i < 0:
            i = 0
        out.append(symbols[i])
        dec.decode(cum[i], cum[i + 1], r)
    return out


def _write_group(
    out: bytearray,
    name: str,
    channels: int,
    scale_table: ScaleTable,
    indices: list[int],
    model: CategoricalModel,
    codec: int = CODEC_RANGE,
) -> None:
    nb = name.encode("utf-8")
    out += struct.pack("<I", len(nb)) + nb
    out += struct.pack("<I", channels)
    for lo, hi in zip(scale_table.mins, scale_table.maxs):
        out += struct.pack("<dd", lo, hi)
    out += struct.pack("<Q", len(indices))
    out += struct.pack("<I", codec)
    if codec == CODEC_ZLIB:
        packed = array.array("I", indices).tobytes()
        coded = zlib.compress(packed, level=6)
        out += struct.pack("<I", len(coded))
        out += coded
    else:
        out += struct.pack("<I", model.n_distinct)
        symbols, cum, total = model.cumulative_freq()
        enc = RangeEncoder()
        _encode_symbols(symbols, cum, total, indices, enc)
        coded = enc.finish()
        out += struct.pack("<I", len(coded))
        for sym, cnt in model.raw_counts():
            out += struct.pack("<II", sym, cnt)
        out += coded
    padding = (fmt.ALIGNMENT - (len(out) % fmt.ALIGNMENT)) % fmt.ALIGNMENT
    out += b"\x00" * padding


def write_container(
    path: str,
    attributes: dict[str, list],
    groups: tuple[tuple[str, int], ...] = ATTRIBUTE_GROUPS,
    levels: int = 4096,
    codec: int = CODEC_RANGE,
) -> int:
    """Serialize ``attributes`` to ``path``; return the actual byte count.

    ``codec`` selects the per-group entropy coding: ``CODEC_RANGE`` is the
    ledger-priced range coder (default); ``CODEC_ZLIB`` is a fast feedback
    path for fit bisection that still writes and reads back real containers.
    """
    n_gaussians = len(attributes[groups[0][0]])
    out = bytearray()
    out += fmt.MAGIC
    out += struct.pack("<IIIII", fmt.VERSION, n_gaussians, len(groups), levels, codec)
    for name, channels in groups:
        if name not in attributes:
            raise KeyError(
                f"serialize: missing attribute group {name!r}; found {sorted(attributes)}"
            )
        values = attributes[name]
        scale_table, indices = quantize_group(values, levels)
        model = CategoricalModel.from_symbols(indices, levels)
        _write_group(out, name, channels, scale_table, indices, model, codec=codec)
    with open(path, "wb") as f:
        f.write(out)
    return len(out)


def read_container(
    path: str, groups: tuple[tuple[str, int], ...] = ATTRIBUTE_GROUPS
) -> dict[str, tuple[list[int], ScaleTable]]:
    """Read a container back; return per-group (quantized indices, scale table)."""
    with open(path, "rb") as f:
        data = f.read()

    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    magic = take(8)
    if magic != fmt.MAGIC:
        raise ValueError(f"read_container: bad magic {magic!r}")
    version, _n_gaussians, _n_groups, levels, _reserved = struct.unpack("<IIIII", take(20))
    if version != fmt.VERSION:
        raise ValueError(f"read_container: unsupported version {version}")

    result: dict[str, tuple[list[int], ScaleTable]] = {}
    for _ in range(len(groups)):
        (name_len,) = struct.unpack("<I", take(4))
        name = take(name_len).decode("utf-8")
        (channels,) = struct.unpack("<I", take(4))
        mins: list[float] = []
        maxs: list[float] = []
        for _ in range(channels):
            lo, hi = struct.unpack("<dd", take(16))
            mins.append(lo)
            maxs.append(hi)
        (n_symbols,) = struct.unpack("<Q", take(8))
        (codec,) = struct.unpack("<I", take(4))
        if codec == CODEC_ZLIB:
            (coded_len,) = struct.unpack("<I", take(4))
            packed = zlib.decompress(take(coded_len))
            counts = [0, 1]  # placeholder; zlib groups carry no counts table
            indices = list(array.array("I", packed))
            result[name] = (indices, ScaleTable(tuple(mins), tuple(maxs)))
            # Skip padding to the alignment boundary exactly as the writer.
            padding = (fmt.ALIGNMENT - (pos % fmt.ALIGNMENT)) % fmt.ALIGNMENT
            take(padding)
            continue
        (n_distinct,) = struct.unpack("<I", take(4))
        (coded_len,) = struct.unpack("<I", take(4))
        entries = [struct.unpack("<II", take(8)) for _ in range(n_distinct)]
        coded = take(coded_len)
        # Skip padding to the alignment boundary, measured from the absolute
        # stream offset exactly as the writer computes it (cumulative, not
        # per-section, because the 28-byte header is not 8-aligned).
        padding = (fmt.ALIGNMENT - (pos % fmt.ALIGNMENT)) % fmt.ALIGNMENT
        take(padding)
        counts = {sym: cnt for sym, cnt in entries}
        model = CategoricalModel(counts, n_symbols, levels)
        symbols, cum, total = model.cumulative_freq()
        indices = _decode_symbols(coded, symbols, cum, total, n_symbols)
        result[name] = (indices, ScaleTable(tuple(mins), tuple(maxs)))
    return result
