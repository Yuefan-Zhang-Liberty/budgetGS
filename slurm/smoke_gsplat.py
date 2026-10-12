"""Cluster smoke test for BudgetGS.

Verifies that the compute environment can import torch and gsplat, sees a GPU,
and runs one forward + backward pass through the gsplat rasterizer.
"""

from __future__ import annotations

import torch
import gsplat


def main() -> None:
    """Run the smoke checks and print one line per check."""
    print('torch_version', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print('gpu_name', torch.cuda.get_device_name(0))

    n_gaussians = 1024
    means = torch.randn(n_gaussians, 3, device=device) * 0.05
    scales = torch.rand(n_gaussians, 3, device=device) * 0.01 + 1e-3
    quats = torch.randn(n_gaussians, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacities = torch.sigmoid(torch.randn(n_gaussians, device=device))
    colors = torch.rand(n_gaussians, 3, device=device)

    width, height = 128, 128
    viewmat = torch.eye(4, device=device)
    intrinsics = torch.tensor([[100.0, 0.0, width / 2.0],
                               [0.0, 100.0, height / 2.0],
                               [0.0, 0.0, 1.0]], device=device)

    renders, alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
    )
    loss = renders.sum() + alphas.sum()
    loss.backward()
    assert means.grad is not None
    print('forward_backward_ok', tuple(renders.shape))
    print('SMOKE_DONE')


if __name__ == '__main__':
    main()
