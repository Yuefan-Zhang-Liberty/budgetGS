import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/home/stud/zyuef/budgetgs-m0b")
ROOT = "/home/stud/zyuef/budgetgs-m0b"

for tag, d in [("P02D", "runs/joint-dual-room375k-p02d"),
               ("V2FT", "runs/purel1-ft-room375k")]:
    ck = torch.load(os.path.join(ROOT, d, "joint-room.pt"), map_location="cpu", weights_only=False)
    m = ck["means"].numpy()
    print("==== %s" % tag)
    print("  means range per-axis:", np.round(m.min(0), 4), np.round(m.max(0), 4))
    print("  means std:", np.round(m.std(0), 4), " overall std:", round(float(m.std()), 5))
    # nearest-neighbor spacing estimate
    idx = np.random.default_rng(0).choice(len(m), min(20000, len(m)), replace=False)
    sub = m[idx]
    from scipy.spatial.distance import cdist
    dd = cdist(sub, sub)
    dd[dd == 0] = 1e9
    nn = dd.min(1)
    print("  NN spacing median/mean:", round(float(np.median(nn)), 5), round(float(nn.mean()), 5))
    s = ck["scales"].numpy()
    print("  scales mean/median:", round(float(s.mean()), 4), round(float(np.median(s)), 4))
