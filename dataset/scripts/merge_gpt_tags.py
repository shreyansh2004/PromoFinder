"""Validate ChatGPT answers and merge them into the dataset as ai_* columns.

Reads gpt/answers/text_batch_*.jsonl and image_sheet_*.jsonl (pasted from ChatGPT).
Every line is checked: ad_id must exist, enums must be allowed values. Bad lines are
reported and skipped, never "fixed" by guessing.

usage: python merge_gpt_tags.py <ads_dataset.csv> <answers_dir> <out_csv> "<tagger name>"
"""
import json, re, sys
from pathlib import Path
import pandas as pd

csv_path, ans_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tagger = sys.argv[4] if len(sys.argv) > 4 else "ChatGPT"
d = pd.read_csv(csv_path, dtype={"ad_id": str})
ids = set(d.ad_id)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tag_schema import TEXT, TROPES, IMAGE


def lines(f):
    for ln in f.read_text(encoding="utf-8").splitlines():
        ln = ln.strip().rstrip(",")
        if ln.startswith("{"):
            yield ln


def check(rec, enums, kind):
    errs = []
    if str(rec.get("ad_id")) not in ids:
        errs.append("unknown ad_id")
    for k, allowed in enums.items():
        if str(rec.get(k)) not in allowed:
            errs.append(f"{k}={rec.get(k)!r}")
    if kind == "text":
        tr = rec.get("tropes")
        if not isinstance(tr, list) or not tr or not set(tr) <= TROPES:
            errs.append(f"tropes={tr!r}")
        if rec.get("emotion_intensity") not in (1, 2, 3, 4, 5):
            errs.append("emotion_intensity")
        for b in ("hook_is_question", "open_mid_conflict", "narrator_voiceover"):
            if not isinstance(rec.get(b), bool):
                errs.append(b)
    else:
        if not isinstance(rec.get("on_screen_text"), bool):
            errs.append("on_screen_text")
    return errs


tags, bad = {}, []
SOURCES = (("text", "text_batch_*.jsonl", TEXT), ("text", "redo_text*.jsonl", TEXT),  # redo overrides
           ("image", "image_sheet_*.jsonl", IMAGE), ("image", "redo_image*.jsonl", IMAGE))
for kind, pattern, enums in SOURCES:
    for f in sorted(ans_dir.glob(pattern)):
        for ln in lines(f):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                bad.append((f.name, "?", "not valid JSON"))
                continue
            rec["ad_id"] = re.sub(r"\D", "", str(rec.get("ad_id", "")))
            errs = check(rec, enums, kind)
            if errs:
                bad.append((f.name, rec.get("ad_id"), "; ".join(errs)))
                continue
            row = tags.setdefault(rec["ad_id"], {})
            for k, v in rec.items():
                if k == "ad_id":
                    continue
                col = f"ai_{'img_' if kind == 'image' else ''}{k}"
                row[col] = "|".join(v) if isinstance(v, list) else v

t = pd.DataFrame.from_dict(tags, orient="index").rename_axis("ad_id").reset_index()
if len(t) and "ai_hook_type" in t:
    base = d.drop_duplicates("ad_id").set_index("ad_id")
    has_speech = t.ad_id.map(base.hook_15s_text.fillna("").str.strip().ne(""))
    contradict = (t.ai_hook_type == "no_speech") & has_speech
    t.loc[contradict, "ai_hook_type"] = None  # contradicts the real transcript: drop, don't guess
    long_no_end = t.ad_id.map((base.duration_s > 180) & base.ending_20s_text.fillna("").str.strip().eq(""))
    t.loc[long_no_end.fillna(False), "ai_ending_type"] = None  # ending was never transcribed
    print(f"dropped hook_type contradicting transcript: {int(contradict.sum())} | "
          f"ending_type blanked (not transcribed): {int(long_no_end.fillna(False).sum())}")
if len(t):
    t["ai_tagger"] = tagger
    # copy tags to every ad that shares the same creative (same video)
    grp = d.set_index("ad_id").creative_group
    t["creative_group"] = t.ad_id.map(grp)
    t = t.drop(columns="ad_id").drop_duplicates("creative_group")
    out = d.merge(t, on="creative_group", how="left")
else:
    out = d
out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"tagged ads: {len(tags)} | rejected lines: {len(bad)} -> {out_csv}")
for f, a, e in bad[:50]:
    print(f"  REJECTED {f} ad_id={a}: {e}")
