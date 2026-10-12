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
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ckpt.items()
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
    for name, _channels in ATTRIBUTE_GROUPS:
        value = _extract(container, name)
        if value is None:
            raise KeyError(
                f"checkpoint has no attribute for group {name!r}; found keys: {sorted(container)}"
            )
        attributes[name] = value
    return attributes
