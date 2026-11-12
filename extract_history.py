import json, os

def load(p):
    with open(p) as f:
        return json.load(f)

def as_list(h):
    if isinstance(h, list):
        return h
    if isinstance(h, dict):
        keys = sorted(h.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        return [h[k] for k in keys]
    return []

def show(path, tag):
    print("===", tag)
    d = load(path)
    print("keys:", sorted(d.keys()))
    h = as_list(d.get("history") or d.get("history_tail") or [])
    print("n_history:", len(h))
    if h:
        print("first_sample:", h[0])
        print("last_sample:", h[-1])
        sel = h if len(h) <= 24 else (h[:6] + h[len(h)//2-3:len(h)//2+3] + h[-12:])
        for it in sel:
            print(it)
    print("landed:", {k: d.get(k) for k in ("landed_mb", "landed_psnr", "landed_ssim", "landed_lpips", "final_mb", "final_psnr", "target_met", "mulaw_mu") if k in d})
    print("ref:", d.get("ref_metrics"))

base = "/home/stud/zyuef/budgetgs-m0b"
show(os.path.join(base, "runs/joint-dual-room375k/joint-room-result.json"), "f4-375k")
show(os.path.join(base, "runs/joint/joint-room-result.json"), "f4-246k")
show(os.path.join(base, "runs/joint-dual-room375k-f1/joint-room-result.json"), "f1-375k")
show(os.path.join(base, "runs/joint-dual1e4/joint-bonsai-result.json"), "bonsai-375k")
