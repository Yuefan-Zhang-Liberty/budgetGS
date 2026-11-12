"""Inspect joint-room.pt (1974260) checkpoint structure."""
import torch

ckpt = torch.load(
    "/home/stud/zyuef/budgetgs-m0b/runs/joint/joint-room.pt",
    map_location="cpu",
    weights_only=False,
)
print("top keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
        else:
            print(f"  {k}: {type(v)}", getattr(v, "shape", ""))
