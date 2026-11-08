"""Independent check of a run's promos against the source scripts (no AI).

For every promo segment: start/end must coincide with a real beat or shot boundary in the screenplay,
no dialogue line may be cut mid-way, total length 20-60 s, and something must happen in the first 3 s.
usage: python model/verify_outputs.py <promos.json> <scripts folder>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promofinder.parser import parse_folder  # noqa: E402

res = json.load(open(sys.argv[1], encoding="utf-8"))
eps = {e.ep: e for e in parse_folder(sys.argv[2])}
fails, checks = [], 0
for i, p in enumerate(res["promos"], 1):
    total = 0.0
    for s in p["segments"]:
        e = eps[s["episode"]]
        bounds = {round(b.start, 2) for b in e.beats} | {round(b.end, 2) for b in e.beats}
        checks += 4
        if round(s["start_s"], 2) not in bounds:
            fails.append(f"#{i} E{s['episode']} start {s['start']} is not a beat boundary")
        if round(s["end_s"], 2) not in bounds:
            fails.append(f"#{i} E{s['episode']} end {s['end']} is not a beat boundary")
        cut = [b for b in e.beats if b.kind == "DIALOGUE" and b.start < s["end_s"] - 0.05 and b.end > s["end_s"] + 0.05]
        if cut:
            fails.append(f"#{i} E{s['episode']} cuts dialogue mid-line at {s['end']}: {cut[0].text[:40]}")
        if not (0 <= s["start_s"] < s["end_s"] <= e.duration + 0.01):
            fails.append(f"#{i} E{s['episode']} outside episode")
        total += s["end_s"] - s["start_s"]
    checks += 2
    if not 20 <= total <= 60.01:
        fails.append(f"#{i} total {total:.1f}s outside 20-60")
    first = [t for t in p["transcript"] if t["kind"] in ("DIALOGUE", "ACTION", "SFX")]
    if not first or first[0]["t"] > "00:03.00":
        if "late_hook" not in p["flags"]:
            fails.append(f"#{i} nothing in first 3 s and not flagged")
print(f"{len(res['promos'])} promos, {checks} checks, {len(fails)} failures")
for f in fails:
    print("  FAIL", f)
