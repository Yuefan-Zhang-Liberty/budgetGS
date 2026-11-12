"""Simulate mu training on REAL room-1974260 attribute distributions.
Answers: for the actual data, does growing mu reduce the L1+noise loss?"""
import torch
import sys

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import quantize_attr

torch.manual_seed(0)
ckpt = torch.load(
    "/home/stud/zyuef/budgetgs-m0b/runs/joint/joint-room.pt",
    map_location="cpu",
    weights_only=False,
)
levels = {"means": 1052.0, "scales": 132.0, "quats": 132.0,
          "opacity": 38.0, "sh0": 66.0, "shn": 10.0}
data = {
    "means": ckpt["means"],
    "scales": ckpt["scales"],
    "quats": ckpt["quats"],
    "opacity": torch.sigmoid(ckpt["opacity"]),  # logit -> [0,1]
    "sh0": ckpt["sh0"],
    "shn": ckpt["shn"],
}

for name in levels:
    x = data[name].to(torch.float32)
    mu_raw = torch.nn.Parameter(torch.full((), -1.0, dtype=torch.float32))
    opt = torch.optim.AdamW([{"params": mu_raw, "lr": 1e-2}])
    lvl = torch.tensor(levels[name], dtype=torch.float32)
    l0 = l500 = None
    for step in range(1, 501):
        opt.zero_grad()
        mu = torch.nn.functional.softplus(mu_raw.clamp(-5.0, 6.0)) + 1e-6
        xq = quantize_attr(x, lvl, noise=True, mu=mu)
        loss = (xq - x).abs().mean()
        loss.backward()
        opt.step()
        if step == 1:
            l0 = float(loss)
        if step == 500:
            l500 = float(loss)
    muv = float(torch.nn.functional.softplus(mu_raw) + 1e-6)
    print(
        f"{name:8s}: loss {l0:.6f} -> {l500:.6f} "
        f"mu 0.3133 -> {muv:.4f}  (mu_raw={float(mu_raw):.4f})"
    )
