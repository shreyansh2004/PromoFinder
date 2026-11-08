"""Why were some AI-proposed clips rejected? Prints span of start..end and hook..cliffhanger beats."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promofinder.clips import Library, norm_beat_id  # noqa: E402
from promofinder.parser import parse_folder  # noqa: E402

lib = Library(parse_folder(sys.argv[1]))
for f in (ROOT / ".cache" / "llm").glob("*.json"):
    text = json.loads(f.read_text(encoding="utf-8"))["text"]
    if '"promos"' not in text or '"hook_beat"' not in text:
        continue
    try:
        start = text.find("{")
        obj = json.loads(text[start:text.rfind("}") + 1])
    except Exception:
        continue
    for p in obj.get("promos", []):
        ids = {k: norm_beat_id(p.get(k)) for k in ("start_beat", "end_beat", "hook_beat", "cliffhanger_beat")}
        b = {k: lib.beats.get(v) for k, v in ids.items()}
        if not b["start_beat"] or not b["end_beat"]:
            continue
        span = b["end_beat"].end - b["start_beat"].start
        if span > 60 or span < 20:
            hc = (b["cliffhanger_beat"].end - b["hook_beat"].start) if b["hook_beat"] and b["cliffhanger_beat"] else None
            print(f"{ids['start_beat']}..{ids['end_beat']} span {span:.0f}s | hook..cliff "
                  f"{ids['hook_beat']}..{ids['cliffhanger_beat']} = {hc if hc is None else round(hc)}s")
