"""Print a compact summary of a run's promos.json (for reviewing results)."""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
u = r["llm_usage"]
print(f"mode {r['mode']} | runtime {r['runtime_s']}s | candidates {r['all_candidates']} | calls {u['calls']} "
      f"| tokens in/out {u['tokens_in']}/{u['tokens_out']} | served {u.get('served_by')}")
print("failover events:", (u.get("failover_events") or [])[:5])
print(f"problems: {len(r['rejected_or_problems'])}")
for p in r["rejected_or_problems"][:12]:
    print("  -", p)
print("premise:", r["bible"].get("premise"))
print("continuity:", r["bible"].get("continuity_issues"))
for i, p in enumerate(r["promos"], 1):
    s = " + ".join(f"E{x['episode']} {x['start']}-{x['end']}" for x in p["segments"])
    j, w = p["judge"] or {}, p["scores"]
    print(f"{i:2d}. [{w['final']:.3f}] {s} ({p['duration_s']:.0f}s) {p['title'][:55]} | "
          f"h{j.get('hook')} c{j.get('cliffhanger')} s{j.get('standalone')} | p={w.get('model_prob')} | "
          f"{p['source']} | {','.join(p['flags'])}")
