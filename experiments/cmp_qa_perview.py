import json, os
ROOT = "/home/stud/zyuef/budgetgs-m0b"
RUNS = [("P02D", "runs/joint-dual-room375k-p02d"),
        ("V2FT", "runs/purel1-ft-room375k"),
        ("QA", "runs/qa-ft-room375k")]
BAD = {168, 288, 272, 40, 232, 280, 224, 296}

for tag, d in RUNS:
    p = os.path.join(ROOT, d, "joint-room-result.json")
    if not os.path.exists(p):
        print("==== %s MISSING" % tag); continue
    r = json.load(open(p))
    pv = r["per_view"]
    bad = [x["psnr"] for x in pv if x["view"] in BAD]
    good = [x["psnr"] for x in pv if x["view"] not in BAD]
    allv = [x["psnr"] for x in pv]
    landed = r["landed"]["psnr"]; ref = r["ref"]["psnr"]
    print("==== %s" % tag)
    print("  landed=%.4f ref=%.4f gap=%.3f  MB=%.3f"
          % (landed, ref, ref - landed, r["actual_mb"]))
    print("  bad-8 mean=%.3f  good-31 mean=%.3f  all mean=%.3f"
          % (sum(bad)/len(bad), sum(good)/len(good), sum(allv)/len(allv)))
    print("  bad views:", {x["view"]: round(x["psnr"], 2) for x in pv if x["view"] in BAD})
