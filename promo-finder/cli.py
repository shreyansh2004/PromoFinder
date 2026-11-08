"""Command line: python cli.py --input <scripts folder> [--out outputs/run] [--top 15] [--no-llm]

LLM is configured by env vars (or a .env file): LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
"""
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from promofinder.llm import LLM  # noqa: E402
from promofinder.pipeline import run  # noqa: E402

ap = argparse.ArgumentParser(description="Find and rank promo clips in microdrama screenplays")
ap.add_argument("--input", required=True, help="folder with episode .txt screenplays")
ap.add_argument("--out", default=None, help="output folder (default outputs/<timestamp>)")
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--no-llm", action="store_true", help="rule-based mode, no API needed")
ap.add_argument("--model", default=None, help="override LLM_MODEL")
ap.add_argument("--max-judge", type=int, default=30, help="clips sent to LLM tagging/judging")
ap.add_argument("--episodes", default=None, help="subset, e.g. 1-5 or 1,3,10-12 (default: all)")
a = ap.parse_args()

out = a.out or str(Path("outputs") / time.strftime("%Y%m%d-%H%M%S"))
res = run(a.input, out, llm=LLM(model=a.model), top_n=a.top, use_llm=not a.no_llm, max_judge=a.max_judge,
          episodes=a.episodes)
print(f"\n{len(res['promos'])} promos -> {out}  (mode={res['mode']}, {res['runtime_s']}s, LLM calls {res['llm_usage']['calls']})")
for i, p in enumerate(res["promos"][:10], 1):
    segs = " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
    print(f"{i:2d}. [{p['scores']['final']:.3f}] {segs} ({p['duration_s']:.0f}s) {p['title'][:60]}")
