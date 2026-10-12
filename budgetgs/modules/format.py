"""On-disk byte format constants for the BudgetGS container.

The ledger (code-length predictor) and the serializer must agree on every byte
of fixed overhead, otherwise the <= 1 % acceptance test can never pass. Keeping
the layout constants here guarantees the two sides cannot drift apart.

Container layout
================
header::

    magic          8 bytes   b"BUDGETGS"
    version        4 bytes   uint32 = 1
    n_gaussians    4 bytes   uint32
    n_groups       4 bytes   uint32
    levels         4 bytes   uint32
    reserved       4 bytes   uint32

per group (fixed order of ATTRIBUTE_GROUPS)::

    name_len       4 bytes   uint32
    name           name_len bytes (utf-8)
    n_channels     4 bytes   uint32
    scale table    16 bytes per channel (min float64, max float64)
    n_symbols      8 bytes   uint64
    n_distinct     4 bytes   uint32
    coded_len      4 bytes   uint32
    table          8 bytes per distinct symbol (symbol uint32, count uint32)
    coded data     coded_len bytes (range-coded quantized indices)
    padding        to ALIGNMENT
"""

from __future__ import annotations

MAGIC = b"BUDGETGS"
VERSION = 1

HEADER_BYTES = 28  # magic(8) + version(4) + n_gaussians(4) + n_groups(4) + levels(4) + reserved(4)
NAME_LEN_BYTES = 4
CHANNELS_BYTES = 4
N_SYMBOLS_BYTES = 8
N_DISTINCT_BYTES = 4
CODED_LEN_BYTES = 4
SCALE_ENTRY_BYTES = 16  # min(float64) + max(float64)
SYMBOL_COUNT_ENTRY_BYTES = 8  # symbol(uint32) + count(uint32)
ALIGNMENT = 8
