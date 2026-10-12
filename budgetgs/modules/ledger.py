"""Code-length ledger: predicts the serialized size of a checkpoint.

Week-1 core deliverable (plan Day 4). The ledger prices, attribute by attribute,
the -sum log2 p of the quantized indices under one block-adaptive categorical
model per group, then adds every fixed overhead the serializer really writes:
the container header, per-block min/max scale tables, transmitted probability
tables (attribute table) and alignment padding.

Acceptance (plan): |predicted - actual| / actual <= 1 % on serialized files of
>= 1 MB. Because the serializer uses the very same quantization and probability
model, the only residual difference is the range coder's constant overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import format as fmt
from .probability import CategoricalModel
from .quantize import ATTRIBUTE_GROUPS, quantize_group


@dataclass(frozen=True)
class LedgerConfig:
    levels: int = 4096
    alignment: int = fmt.ALIGNMENT
    header_bytes: int = fmt.HEADER_BYTES


@dataclass
class AttributeLedger:
    name: str
    channels: int
    n_symbols: int
    n_distinct: int
    code_bits: float
    code_bytes: int
    scale_bytes: int
    table_bytes: int
    name_bytes: int
    section_bytes: int  # fixed overhead + coded data, excluding padding
    padding: int

    @property
    def total_bytes(self) -> int:
        return self.section_bytes + self.padding


@dataclass
class LedgerResult:
    config: LedgerConfig
    attributes: list[AttributeLedger]
    total_predicted_bytes: int

    def per_attribute_bytes(self) -> dict[str, int]:
        return {a.name: a.total_bytes for a in self.attributes}


def _group_ledger(
    name: str,
    channels: int,
    n_symbols: int,
    n_distinct: int,
    code_bits: float,
    config: LedgerConfig,
    offset: int,
) -> AttributeLedger:
    code_bytes = max(1, math.ceil(code_bits / 8.0))
    name_bytes = fmt.NAME_LEN_BYTES + len(name.encode("utf-8"))
    scale_bytes = channels * fmt.SCALE_ENTRY_BYTES
    table_bytes = (
        fmt.N_DISTINCT_BYTES
        + fmt.CODED_LEN_BYTES
        + n_distinct * fmt.SYMBOL_COUNT_ENTRY_BYTES
    )
    fixed = name_bytes + fmt.CHANNELS_BYTES + scale_bytes + fmt.N_SYMBOLS_BYTES + table_bytes
    section = fixed + code_bytes
    # Padding is computed from the cumulative offset exactly as the serializer
    # does (header 28 is not 8-aligned, so per-group padding alone would drift).
    padding = (config.alignment - ((offset + section) % config.alignment)) % config.alignment
    return AttributeLedger(
        name, channels, n_symbols, n_distinct, code_bits, code_bytes,
        scale_bytes, table_bytes, name_bytes, section, padding,
    )


def compute_ledger(
    attributes: dict[str, list],
    groups: tuple[tuple[str, int], ...] = ATTRIBUTE_GROUPS,
    config: LedgerConfig = LedgerConfig(),
) -> LedgerResult:
    """Predict the serialized file size of ``attributes``, attribute by attribute."""
    running = config.header_bytes
    entries: list[AttributeLedger] = []
    for name, channels in groups:
        if name not in attributes:
            raise KeyError(
                f"ledger: missing attribute group {name!r}; found {sorted(attributes)}"
            )
        values = attributes[name]
        _scale_table, indices = quantize_group(values, config.levels)
        model = CategoricalModel.from_symbols(indices, config.levels)
        entry = _group_ledger(
            name, channels, len(indices), model.n_distinct, model.code_bits(),
            config, running,
        )
        running += entry.total_bytes
        entries.append(entry)
    return LedgerResult(config, entries, running)
