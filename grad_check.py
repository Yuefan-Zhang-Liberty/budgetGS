"""Grad check: does mu_raw actually receive gradients in quantize_attr?"""
import torch
import sys

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import quantize_attr

torch.manual_seed(0)


def check(name, x, levels, init=-3.0):
    mu_raw = torch.nn.Parameter(torch.full((), float(init), dtype=torch.float32))
    mu = torch.nn.functional.softplus(mu_raw) + 1e-6
    lvl = torch.tensor(float(levels), dtype=torch.float32)
    xq = quantize_attr(x, lvl, noise=True, mu=mu)
    loss = (xq - x).abs().mean()
    loss.backward()
    g = None if mu_raw.grad is None else float(mu_raw.grad)
    print(
        f"[{name}] init={init} mu={float(mu):.6f} "
        f"mu_raw.grad={g} loss={float(loss):.6f}"
    )
    return g


# opacity-like: beta left-skewed (center-dense at low values)
x1 = (
    torch.distributions.Beta(0.3, 3.0).sample((5000, 1)).to(torch.float32)
)
x1.requires_grad_()
check("beta(0.3,3) opacity-like", x1, 32)

# gaussian center-dense
x2 = (torch.randn(5000, 1) * 0.1 + 0.5).clip(0, 1).to(torch.float32)
x2.requires_grad_()
check("gauss center-dense", x2, 1024)

# uniform
x3 = torch.rand(5000, 1).to(torch.float32)
x3.requires_grad_()
check("uniform", x3, 1024)

# init sensitivity on the beta distribution
for init in (-1.0, 1.0, 3.0):
    check("beta init-sweep", x1, 32, init=init)
