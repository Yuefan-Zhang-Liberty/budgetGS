"""Checkpoint attribute loading for the BudgetGS CLI.

Checkpoints arrive as torch ``.pt`` files from gsplat training runs, or as
``.npz``/``.json`` dumps during bring-up. Loading is best-effort: attribute keys
are mapped through a small alias table and missing groups raise a clear error
instead of silently producing a wrong ledger.
"""

from __future__ import annotations

import json
import os

from ..modules.quantize import ATTRIBUTE_GROUPS

_ALIASES = {
    "means": ("means", "xyz", "positions"),
    "scales": ("scales", "scales3d"),
    "quats": ("quats", "quaternions", "rotations", "rot"),
    "opacity": ("opacities", "opacity", "alphas"),
    "sh0": ("sh0", "sh_0", "features_sh0"),
    "shn": ("shn", "sh_rest", "shN", "features_rest"),
}


def _extract(container: dict, group: str):
    for alias in _ALIASES[group]:
        if alias in container:
            return container[alias]
    return None


def load_attributes(path: str) -> dict[str, list]:
    """Load the six attribute groups from a ``.pt``/``.pth``/``.ckpt``/``.npz``/``.json`` file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth", ".ckpt"):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]
            container = {
                k: (v.detach().numpy() if hasattr(v, "detach") else v) for k, v in ckpt.items()
            }
        else:
            container = {"means": getattr(ckpt, "means", None)}
    elif ext == ".npz":
        import numpy as np

        with np.load(path) as z:
            container = {k: z[k].tolist() for k in z.files}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            container = json.load(f)
    else:
        raise ValueError(
            f"unsupported checkpoint extension {ext!r}; use .pt/.pth/.ckpt/.npz/.json"
        )

    attributes: dict[str, list] = {}
    for name, channels in ATTRIBUTE_GROUPS:
        value = _extract(container, name)
        if value is None:
            raise KeyError(
                f"checkpoint has no attribute for group {name!r}; found keys: {sorted(container)}"
            )
        attributes[name] = _to_rows(value, channels)
    return attributes


def _to_rows(value, channels: int = 0) -> "object":
    """Normalize one attribute group to an N x C row layout.

    Real gsplat checkpoints arrive in three shapes: flat scalars (opacity is
    ``(N,)``), nested single blocks (sh0 is ``(N, 1, 3)``), and per-gaussian
    matrices (shn is ``(N, 15, 3)``).  numpy views are reshaped in one
    vectorized call; plain python lists are converted through numpy.  The
    declared group ``channels`` decides the flattening: shn with channels=3
    becomes ``(N*15, 3)`` rows (per-block scale table + model, the winner of
    the shn-3-vs-45 comparison); channels=45 keeps ``(N, 45)``.
    """
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return value
    try:
        import numpy as np

        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            if channels and channels < arr.shape[1] * arr.shape[2]:
                arr = arr.reshape(-1, channels)
            else:
                arr = arr.reshape(arr.shape[0], -1)
        return arr
    except ImportError:  # pragma: no cover - numpy is a project dependency
        pass
    if not hasattr(value[0], "__len__"):
        return [[v] for v in value]
    rows = []
    for row in value:
        flat = []
        stack = list(row)
        while stack:
            item = stack.pop(0)
            if hasattr(item, "__len__"):
                stack[0:0] = list(item)
            else:
                flat.append(item)
        rows.append(flat)
    return rows
