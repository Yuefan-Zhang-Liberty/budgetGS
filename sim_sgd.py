"""Simulate mu training with SGD lr=1e0 on REAL bonsai-1975189 attr dists
(subsampled rows, shn trimmed to 3 channels) to verify mu_raw settles in the
optimum range instead of slamming into the clamp edges (1975189 AdamW
pathology) over a long run."""
import torch
import sys

sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
from train_joint import quantize_attr

torch.manual_seed(0)
ckpt = torch.load(
    "/home/stud/zyuef/budgetgs-m0b/runs/joint/joint-bonsai.pt",
    map_location="cpu",
    weights_only=False,
)
n = int(ckpt.get("n_gaussians", len(ckpt["means"])))
perm = torch.randperm(n)[:30000]
levels = {"means": 1047.0, "scales": 136.0, "quats": 136.0,
          "opacity": 36.0, "sh0": 64.0, "shn": 10.0}
data = {
    "means": ckpt["means"][perm].reshape(-1, 3),
    "scales": ckpt["scales"][perm].reshape(-1, 3),
    "quats": ckpt["quats"][perm].reshape(-1, 4),
    "opacity": torch.sigmoid(ckpt["opacity"][perm].reshape(-1, 1)),
    "sh0": ckpt["sh0"][perm].reshape(-1, 3),
    "shn": ckpt["shn"][perm].reshape(30000, -1)[:, :3],
}

STEPS = 3000
for name in levels:
    x = data[name].to(torch.float32)
    mu_raw = torch.nn.Parameter(torch.full((), -1.0, dtype=torch.float32))
    opt = torch.optim.SGD([{"params": mu_raw, "lr": 1e0}], momentum=0.0)
    lvl = torch.tensor(levels[name], dtype=torch.float32)
    l0 = lN = None
    traj = []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        mu = torch.nn.functional.softplus(mu_raw.clamp(-5.0, 6.0)) + 1e-6
        xq = quantize_attr(x, lvl, noise=True, mu=mu)
        loss = (xq - x).abs().mean()
        loss.backward()
        opt.step()
        if step == 1:
            l0 = float(loss)
        if step % 300 == 0:
            traj.append(float(mu_raw))
        if step == STEPS:
            lN = float(loss)
    muv = float(torch.nn.functional.softplus(mu_raw) + 1e-6)
    print(
        f"{name:8s}: loss {l0:.6f} -> {lN:.6f} | "
        f"mu_raw traj: " + " ".join(f"{v:.3f}" for v in traj) +
        f" | end mu={muv:.4f} (raw={float(mu_raw):.4f})"
    )
