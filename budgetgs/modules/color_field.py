"""Shared hashgrid color field for BudgetGS (Compact-3DGS route).

Replaces per-Gaussian SH color (sh0 + shn) with a shared, compact neural color
field: a multi-resolution hash grid over position, optionally augmented with a
SphericalHarmonics encoding of the view direction, decoded by a tiny MLP into
RGB.  This turns the per-Gaussian 48 float SH coefficients (of which the 45-dim
shn block alone costs ~59% of the byte budget) into a fixed-size shared field
whose bytes do NOT grow with the number of Gaussians, so the saved budget can be
spent on geometry (more Gaussians).

view_dependent=True implements the full Compact-3DGS recipe; view_dependent=False
drops the direction encoding for a pure position-only field (simplest smoke).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ColorField(nn.Module):
    def __init__(
        self,
        n_levels: int = 10,
        log2_hashmap_size: int = 17,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        per_level_scale: float = 1.447,
        mlp_neurons: int = 64,
        mlp_hidden: int = 2,
        sh_degree: int = 3,
        view_dependent: bool = True,
    ):
        super().__init__()
        import tinycudann as tcnn

        self.view_dependent = view_dependent
        self.recolor = tcnn.Encoding(
            3,
            {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.n_dir_dims = 0
        if view_dependent:
            self.direction_encoding = tcnn.Encoding(
                3, {"otype": "SphericalHarmonics", "degree": sh_degree}
            )
            self.n_dir_dims = self.direction_encoding.n_output_dims

        in_dims = self.recolor.n_output_dims + self.n_dir_dims
        self.mlp_head = tcnn.Network(
            in_dims,
            3,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": mlp_neurons,
                "n_hidden_layers": mlp_hidden,
            },
        )

    def forward(self, means, cam_center=None):
        x = self.recolor(means)
        if self.view_dependent:
            if cam_center is None:
                raise ValueError("view_dependent ColorField requires cam_center")
            d = means - cam_center
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
            x = torch.cat([x, self.direction_encoding(d)], dim=-1)
        return self.mlp_head(x)  # (N, 3)

    def num_params(self) -> int:
        n = int(self.recolor.params.numel()) + int(self.mlp_head.params.numel())
        if self.view_dependent:
            n += int(self.direction_encoding.params.numel())
        return n

    def hashgrid_params(self) -> int:
        return int(self.recolor.params.numel())

    def mlp_params(self) -> int:
        n = int(self.mlp_head.params.numel())
        if self.view_dependent:
            n += int(self.direction_encoding.params.numel())
        return n

    def storage_bytes(self, bits_per_hash: int = 8, bits_per_mlp: int = 16) -> int:
        """Estimated landed bytes: 8-bit hash grid + 16-bit MLP (no compression)."""
        return self.hashgrid_params() * bits_per_hash // 8 + self.mlp_params() * bits_per_mlp // 8
