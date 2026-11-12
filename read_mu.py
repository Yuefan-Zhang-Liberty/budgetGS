"""Read joint-bonsai.pt (1975189) mulaw_mu + attr stats."""
import torch

ckpt = torch.load(
    "/home/stud/zyuef/budgetgs-m0b/runs/joint/joint-bonsai.pt",
    map_location="cpu",
    weights_only=False,
)
print("n_gaussians:", ckpt.get("n_gaussians"))
print("step:", ckpt.get("step"))
mu = ckpt["mulaw_mu"]
names = ["means", "scales", "quats", "opacity", "sh0", "shn"]
print("mulaw_mu:", ["%.5f" % v for v in mu.tolist()])
for i, n in enumerate(names):
    print(f"  {n:8s} mu={mu[i].item():.5f}")
# attr ranges
for n in ["means", "scales", "quats", "opacity", "sh0", "shn"]:
    t = ckpt[n]
    print(f"  {n:8s} shape={tuple(t.shape)} min={t.min().item():.4f} max={t.max().item():.4f}")
