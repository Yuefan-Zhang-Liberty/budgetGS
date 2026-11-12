#!/usr/bin/env python3
import json
base="/home/stud/zyuef/budgetgs-m0b/runs/analysis"
for scene in ["room","bonsai","garden"]:
    d=json.load(open(f"{base}/directionB-{scene}.json"))
    ref=d["ref"]
    print("="*22, scene, f"ref {ref['psnr']:.2f}/{ref['ssim']:.4f}/{ref['lpips']:.4f}", "="*22)
    for r in d["rows"]:
        landed=r.get("landed") or {}
        lv=r.get("levels")
        lvstr = lv if isinstance(lv,int) else ",".join(f"{k[:2]}:{v}" for k,v in lv.items())
        p=landed.get("psnr"); s=landed.get("ssim"); l=landed.get("lpips")
        ps = f"{p:.2f}" if isinstance(p,(int,float)) else str(p)
        ss = f"{s:.3f}" if isinstance(s,(int,float)) else str(s)
        ls = f"{l:.3f}" if isinstance(l,(int,float)) else str(l)
        print(f"  {r['tag']:14s} {r['actual_mb']:6.2f}MB  PSNR {ps:>6} SSIM {ss:>6} LPIPS {ls:>6}  lv={lvstr}")
    print()
