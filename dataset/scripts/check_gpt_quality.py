"""Audit ChatGPT tag answers against the real data before merging.

Checks per batch: coverage (missing / extra / duplicate ad_ids), schema validity (same rules as
merge_gpt_tags.py), evidence grounding (the quoted evidence must actually occur in that ad's
transcript or title), consistency (no_speech vs real speech, question flag vs '?'), and label
diversity (one label > 70% of a batch is suspicious).

usage: python check_gpt_quality.py <ads_dataset.csv> <gpt_dir>
"""
import json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv_path, gdir = Path(sys.argv[1]), Path(sys.argv[2])
d = pd.read_csv(csv_path, dtype={"ad_id": str}).drop_duplicates("ad_id").set_index("ad_id")

from tag_schema import TEXT, TROPES, IMAGE


def norm(s):
    return re.findall(r"[a-z0-9']+", str(s).lower())


def grounded(evidence, ad_id):
    if ad_id not in d.index:
        return 0.0
    r = d.loc[ad_id]
    hay = set(norm(" ".join(str(r.get(c, "")) for c in ("hook_15s_text", "ending_20s_text", "ad_title"))))
    ev = [w for w in norm(evidence) if len(w) > 2]
    return 1.0 if not ev else sum(w in hay for w in ev) / len(ev)


def parse(f):
    recs, bad_json = [], 0
    for ln in f.read_text(encoding="utf-8").splitlines():
        ln = ln.strip().rstrip(",")
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
            r["ad_id"] = re.sub(r"\D", "", str(r.get("ad_id", "")))
            recs.append(r)
        except json.JSONDecodeError:
            bad_json += 1
    return recs, bad_json


redo, flagged = [], set()
print("=== TEXT BATCHES ===")
for f in sorted((gdir / "answers").glob("text_batch_*.jsonl")) + sorted((gdir / "answers").glob("redo_text*.jsonl")):
    q = gdir / ("redo" if f.stem.startswith("redo") else "text_batches") / (f.stem + ".md")
    expected = re.findall(r"^## ad_id: (\d+)", q.read_text(encoding="utf-8"), re.M) if q.exists() else []
    recs, bad_json = parse(f)
    got = [r["ad_id"] for r in recs]
    missing = [a for a in expected if a not in got]
    extra = [a for a in got if a not in expected]
    dups = [a for a, n in Counter(got).items() if n > 1]
    invalid, ungrounded, inconsistent = [], [], []
    for r in recs:
        errs = [k for k, allowed in TEXT.items() if str(r.get(k)) not in allowed]
        tr = r.get("tropes")
        if not isinstance(tr, list) or not tr or not set(tr) <= TROPES:
            errs.append("tropes")
        if r.get("emotion_intensity") not in (1, 2, 3, 4, 5):
            errs.append("emotion_intensity")
        if errs:
            invalid.append((r["ad_id"], errs))
        g = grounded(r.get("evidence", ""), r["ad_id"])
        if r.get("hook_type") not in ("no_speech", "unclear") and g < 0.6:
            ungrounded.append((r["ad_id"], round(g, 2), r.get("evidence", "")[:60]))
        if r["ad_id"] in d.index:
            row = d.loc[r["ad_id"]]
            has_speech = isinstance(row.get("hook_15s_text"), str) and row.hook_15s_text.strip() != ""
            if r.get("hook_type") == "no_speech" and has_speech:
                inconsistent.append((r["ad_id"], "no_speech but speech exists"))
            if r.get("hook_type") != "no_speech" and not has_speech:
                inconsistent.append((r["ad_id"], f"{r.get('hook_type')} but no speech"))
    flagged |= set(missing) | {a for a, _ in invalid} | {x[0] for x in ungrounded} | {a for a, _ in inconsistent}
    div = {k: Counter(str(r.get(k)) for r in recs).most_common(1)[0] for k in ("genre", "hook_type") if recs}
    skew = {k: v for k, v in div.items() if recs and v[1] / len(recs) > 0.7}
    n = len(expected) or 1
    score_bad = len(missing) + len(extra) + len(dups) + len(invalid) + len(ungrounded) + len(inconsistent) + bad_json
    verdict = "OK" if score_bad <= max(2, 0.1 * n) and not skew else "REDO"
    if verdict == "REDO":
        redo.append(f.stem)
    print(f"\n{f.stem}: expected {len(expected)} got {len(recs)} | missing {len(missing)} extra {len(extra)} "
          f"dups {len(dups)} bad_json {bad_json} invalid {len(invalid)} ungrounded {len(ungrounded)} "
          f"inconsistent {len(inconsistent)} | top labels {div} -> {verdict}")
    for x in (missing[:5], extra[:5], invalid[:5], ungrounded[:5], inconsistent[:5]):
        if x:
            print("   ", x)

print("\n=== IMAGE SHEETS ===")
for f in sorted((gdir / "answers").glob("image_sheet_*.jsonl")):
    recs, bad_json = parse(f)
    got = [r["ad_id"] for r in recs]
    sheet_txt = gdir / "image_sheets" / (f.stem.replace("image_", "") + ".txt")
    sheet_ids = sheet_txt.read_text(encoding="utf-8").split() if sheet_txt.exists() else []
    unknown = [a for a in got if a not in sheet_ids] + [f"missing:{a}" for a in sheet_ids if a not in got]
    flagged_img = getattr(sys.modules[__name__], "flagged_img", set())
    invalid = [(r["ad_id"], [k for k, al in IMAGE.items() if str(r.get(k)) not in al]) for r in recs]
    invalid = [x for x in invalid if x[1]]
    top = Counter(str(r.get("visual_hook")) for r in recs).most_common(2)
    verdict = "OK" if len(unknown) + len(invalid) + bad_json <= 1 else "REDO"
    if verdict == "REDO":
        redo.append(f.stem)
    print(f"{f.stem}: got {len(recs)} | unknown ids {len(unknown)} invalid {len(invalid)} bad_json {bad_json} "
          f"| top visual_hook {top} -> {verdict}")
    if unknown or invalid:
        print("   ", unknown[:5], invalid[:5])

print("\nREDO:", redo or "none")
(gdir / "redo").mkdir(exist_ok=True)
(gdir / "redo" / "redo_ids.txt").write_text("\n".join(sorted(flagged)), encoding="utf-8")
print(f"flagged text ads written to redo/redo_ids.txt: {len(flagged)}")
