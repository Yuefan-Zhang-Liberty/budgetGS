#!/usr/bin/env python3
import json
base="/home/stud/zyuef/budgetgs-m0b/runs/analysis"

def fmt(v):
    if isinstance(v,float): return f"{v:.4f}"
    return str(v)

for scene in ["room","bonsai","garden"]:
    p=f"{base}/directionB-{scene}.json"
    d=json.load(open(p))
    print("="*25, scene, "="*25)
    print("n_gauss:", d.get("n_gaussians"), "| ref:", d.get("ref"))
    skip={"ref","ref_per_view","val_idcs","loaded_attribute_shapes","raw_shn_shape","val_every","val_views"}
    for k,v in d.items():
        if k in skip: continue
        if isinstance(v,(int,float,str,bool)):
            print(f"  [{k}] = {v}")
        elif isinstance(v,list):
            if v and isinstance(v[0],dict):
                print(f"  [{k}] LIST len={len(v)} keys={list(v[0].keys())}")
            else:
                print(f"  [{k}] list len={len(v)}: {v[:12]}")
        elif isinstance(v,dict):
            print(f"  [{k}] DICT keys={list(v.keys())[:25]}")
    print()
