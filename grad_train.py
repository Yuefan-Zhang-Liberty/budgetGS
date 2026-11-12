"""Simulate a few training steps: does AdamW lr=1e-2 grow mu for
center-dense data? (quantize_attr noise=True + L1, no rendering needed --
mu's gradient path in the real loop is exactly this)."""
import torch
import sys

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import quantize_attr

torch.manual_seed(0)
N = 20000
x = torch.distributions.Beta(0.3, 3.0).sample((N, 1)).to(torch.float32)
lvl = torch.tensor(32.0, dtype=torch.float32)

mu_raw = torch.nn.Parameter(torch.full((1,), -1.0, dtype=torch.float32))
opt = torch.optim.AdamW([{"params": mu_raw, "lr": 1e-2}])

for step in range(1, 2001):
    opt.zero_grad()
    mu = torch.nn.functional.softplus(mu_raw.clamp(-5.0, 6.0)) + 1e-6
    xq = quantize_attr(x, lvl, noise=True, mu=mu[0])
    loss = (xq - x).abs().mean()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        muv = float(torch.nn.functional.softplus(mu_raw) + 1e-6)
        print(
            f"step {step}: mu_raw={float(mu_raw):.4f} "
            f"mu={muv:.4f} loss={float(loss):.6f}"
        )

print("FINAL mu_raw", float(mu_raw), "mu", float(torch.nn.functional.softplus(mu_raw) + 1e-6))
