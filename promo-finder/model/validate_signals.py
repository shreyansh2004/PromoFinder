"""Validate every ranking signal on REAL ads, the same way the ML model was validated.

For each unique English short-drama ad (Whisper transcript of the first 15 s and, for ads up to 3 min,
the last 20 s) we compute:
  - the stranger test: the exact REVIEW prompt the tool uses, via the same LLM router (Claude first)
  - script signals: the same rule-based hook/cliffhanger scoring, applied to the transcript text
and measure how well each ranks high-reach above low-reach ads WITHIN the same app (AUC), plus
per-source AUC and the link with active days on held-out mid-reach ads.
Results -> model/artifacts/signal_validation.json and stranger_on_ads.csv

usage: python model/validate_signals.py <ads_dataset_tagged.csv> <transcripts_dir> [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)
from promofinder import prompts as P  # noqa: E402
from promofinder.features import CLIFF_WORDS, INTENSE_TONES, VIOLENT_REVEAL, words  # noqa: E402
from promofinder.heuristics import RESOLVED  # noqa: E402
from promofinder.llm import LLM  # noqa: E402

ART = ROOT / "model" / "artifacts"
src, tdir = Path(sys.argv[1]), Path(sys.argv[2])
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

raw = pd.read_csv(src, dtype={"ad_id": str})
u = raw[(raw.language == "en") & (raw.is_short_drama_ad == 1) & (raw.is_first_of_creative == 1)].copy()
u["source"] = np.where(u.app == "unspecified", "search", "named")


def fmt(t):
    m, s = divmod(max(float(t), 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def transcript(ad_id, dur):
    f = tdir / f"{ad_id}.json"
    t = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    head, tail = t.get("head_segments") or [], t.get("tail_segments") or []
    lines = [f"[{fmt(s['start'])}] VOICE: \"{s['text']}\"" for s in head]
    if not lines:
        lines.append("[00:00.0] (no speech in the first 15 seconds)")
    if tail:
        lines.append("(...)")
        lines += [f"[{fmt(s['start'])}] VOICE: \"{s['text']}\"" for s in tail]
    elif dur and dur > 17:
        lines.append("(... the rest of the ad is not available)")
    return "\n".join(lines), head, tail


# ------------------------------------------------------------------ script signals on transcript text
def rule_hook(text):
    s = 0.0
    for line in [x for x in text.replace("!", "!|").replace("?", "?|").replace(".", ".|").split("|") if x.strip()][:3]:
        n = len(words(line))
        s += 1.0 + 0.8 * ("!" in line) + 0.6 * ("?" in line) + 0.6 * (0 < n <= 7) + 0.8 * bool(VIOLENT_REVEAL.search(line))
        s += 1.2 * bool(INTENSE_TONES.search(line))
    return s


def rule_cliff(text):
    if not text.strip():
        return 0.0
    last = text.strip()[-160:]
    return (1.0 * ("?" in last) + 0.6 * ("!" in last) + 1.0 * bool(CLIFF_WORDS.search(last))
            + 0.6 * bool(VIOLENT_REVEAL.search(last)) - 1.8 * bool(RESOLVED.search(last)))


# ------------------------------------------------------------------ stranger test via the tool's own router
llm = LLM()
print("LLM chain:", llm.describe(), flush=True)
cache = ART / "stranger_on_ads.csv"
done = pd.read_csv(cache, dtype={"ad_id": str}).set_index("ad_id").to_dict("index") if cache.exists() else {}


def one(r):
    text, head, tail = transcript(r.ad_id, r.duration_s)
    try:
        j = llm.json(P.REVIEW_SYSTEM, P.REVIEW_USER.format(dur=min(float(r.duration_s or 15), 999), transcript=text),
                     validate=lambda o: [int(o[k]) for k in ("hook", "cliffhanger", "standalone")])
        return {"ad_id": r.ad_id, "s_hook": int(j["hook"]), "s_cliff": int(j["cliffhanger"]),
                "s_standalone": int(j["standalone"]), "s_understood": j.get("what_is_happening", ""), "error": ""}
    except Exception as e:
        return {"ad_id": r.ad_id, "error": str(e)[:150]}


todo = [r for _, r in u.iterrows() if r.ad_id not in done or str(done[r.ad_id].get("error", "")) not in ("", "nan")]
print(f"ads: {len(u)} | already scored: {len(u) - len(todo)} | to score: {len(todo)}", flush=True)
rows = [dict(ad_id=k, **v) for k, v in done.items()]
with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = [ex.submit(one, r) for r in todo]
    for i, f in enumerate(as_completed(futs), 1):
        rows = [x for x in rows if x["ad_id"] != f.result()["ad_id"]] + [f.result()]
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8")
            print(f"  scored {i}/{len(todo)}", flush=True)
s = pd.DataFrame(rows)
s["ad_id"] = s.ad_id.astype(str)
d = u.merge(s, on="ad_id", how="left")
d["stranger"] = (d.s_hook + d.s_cliff + d.s_standalone) / 15
d["rules"] = d.first_3s_text.fillna("").map(rule_hook) + d.ending_20s_text.fillna("").map(rule_cliff)
d["rules_hook_only"] = d.first_3s_text.fillna("").map(rule_hook)
print(f"\nscored OK: {d.stranger.notna().sum()} / {len(d)} | LLM calls {llm.calls}, served by {llm.served_by}")


# ------------------------------------------------------------------ metrics (no training -> no leakage)
def within_app(df, col):
    v, w = [], []
    for a, g in df.groupby("app"):
        g = g.dropna(subset=[col])
        if g.y.nunique() == 2:
            v.append(roc_auc_score(g.y, g[col]))
            w.append(len(g))
    return float(np.average(v, weights=w)) if v else float("nan")


def boot_ci(df, col, n=500):
    rng = np.random.default_rng(0)
    vals = []
    parts = [g for _, g in df.groupby("app")]
    for _ in range(n):
        b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in parts])
        vals.append(within_app(b, col))
    return [round(float(np.nanpercentile(vals, 2.5)), 3), round(float(np.nanpercentile(vals, 97.5)), 3)]


hl = d[d.creative_best_tier.isin(["high", "low"])].copy()
hl["y"] = (hl.creative_best_tier == "high").astype(int)
both = hl.groupby("app").y.nunique().loc[lambda x: x == 2].index
hl = hl[hl.app.isin(both)]
mid = d[d.creative_best_tier == "mid"]
res = {}
print(f"\nWithin-app AUC on {len(hl)} ads from {len(both)} apps (0.50 = coin flip)\n")
print(f"{'signal':28s} within-app  95% CI          search  named   days(mid) rho")
for col, name in (("stranger", "stranger test (total)"), ("s_hook", "  stranger: hook"), ("s_cliff", "  stranger: cliffhanger"),
                  ("s_standalone", "  stranger: standalone"), ("rules", "script signals (total)"),
                  ("rules_hook_only", "  script: hook only")):
    wa = within_app(hl, col)
    ci = boot_ci(hl, col)
    per = {}
    for sname in ("search", "named"):
        g = hl[(hl.source == sname)].dropna(subset=[col])
        per[sname] = round(float(roc_auc_score(g.y, g[col])), 3) if g.y.nunique() == 2 else None
    mm = mid.dropna(subset=[col])
    rho = float(spearmanr(mm[col], mm.days_shown)[0]) if len(mm) > 10 else float("nan")
    res[col] = {"within_app_auc": round(wa, 3), "ci95": ci, "per_source": per, "mid_days_rho": round(rho, 3)}
    print(f"{name:28s} {wa:.3f}      {ci}   {per['search']}   {per['named']}   {rho:+.2f}")

meta = json.loads((ART / "model_meta.json").read_text(encoding="utf-8"))
res["ml_model"] = {"within_app_auc": meta["full"]["within_app_auc"]}
skill = lambda a: round(float(np.clip((a - 0.5) * 2, 0, 0.5)), 3)  # noqa: E731
res["earned_weights"] = {"model": skill(meta["full"]["within_app_auc"]), "editor": skill(res["stranger"]["within_app_auc"]),
                         "rules": skill(res["rules"]["within_app_auc"])}
res["n_ads"], res["apps"] = int(len(hl)), list(both)
res["llm"] = llm.describe()
(ART / "signal_validation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\nearned weights:", res["earned_weights"])
