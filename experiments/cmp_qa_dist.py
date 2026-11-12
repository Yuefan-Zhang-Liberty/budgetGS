import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
ROOT = "/home/stud/zyuef/budgetgs-m0b"
RUNS = [("P02D", "runs/joint-dual-room375k-p02d"),
        ("V2FT", "runs/purel1-ft-room375k"),
        ("QA", "runs/qa-ft-room375k")]

for tag, d in RUNS:
    p = os.path.join(ROOT, d, "joint-room.pt")
    if not os.path.exists(p):
        print("==== %s MISSING" % tag); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("==== %s  N=%d" % (tag, ck["means"].shape[0]))
    for k in ["means", "scales", "quats", "opacity", "sh0", "shn"]:
        x = ck[k].numpy().reshape(-1)
        print("  %-8s min=%+.4f max=%+.4f std=%.4f mean=%+.4f"
              % (k, x.min(), x.max(), x.std(), x.mean()))
