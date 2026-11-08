"""End-to-end run: scripts folder -> ranked promos (+ report files)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import prompts as P
from .candidates import build_bible, dedupe, heuristic_candidates, llm_candidates, llm_montages
from .clips import Library, describe
from .llm import LLM
from .parser import fmt, parse_folder
from .ranker import SuccessModel, diversify, final_scores, review, rule_score

TAG_KEYS = ("genre", "tropes", "hook_type", "hook_is_question", "open_mid_conflict", "narrator_voiceover",
            "ending_type", "setting", "emotion_intensity", "visual_hook")


def tag_clip(clip, llm) -> dict:
    f = clip.features
    try:
        t = llm.json(P.TAG_SYSTEM, P.TAG_USER.format(title=clip.title, first3=f["_first3_text"] or "[no speech]",
                                                     hook15=f["_hook15_text"] or "[no speech]",
                                                     ending=f["_ending_text"] or "[no speech]",
                                                     visual3=f["_visual3_text"] or "[none]"))
        return {k: t.get(k) for k in TAG_KEYS}
    except Exception as e:
        return {"error": str(e)[:200]}


def parse_episode_range(spec: str | None) -> set[int] | None:
    """'1-5' or '1,3,7-9' -> {1,2,3,4,5}; None/'' -> all."""
    if not spec:
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def run(folder: str | Path, out_dir: str | Path | None = None, llm: LLM | None = None, top_n: int = 15,
        use_llm: bool | None = None, max_judge: int = 30, progress=None, episodes: str | None = None) -> dict:
    t0 = time.time()
    say = progress or (lambda m: print(m, flush=True))
    llm = llm or LLM()
    use_llm = llm.available if use_llm is None else (use_llm and llm.available)
    model = SuccessModel()

    say("Parsing screenplays")
    wanted = parse_episode_range(episodes)
    episodes = [e for e in parse_folder(folder) if wanted is None or e.ep in wanted]
    if not episodes:
        raise ValueError("No .txt screenplay files found")
    lib = Library(episodes)
    warnings = [f"Ep {e.ep}: {w}" for e in episodes for w in e.warnings]

    say("Building series bible")
    try:
        bible = build_bible(episodes, llm if use_llm else None)
    except Exception as e:  # all providers down: keep going rule-based
        say(f"AI unavailable for the series bible ({str(e)[:80]}); using loglines")
        bible = build_bible(episodes, None)

    problems = []
    say("Finding candidate moments")
    clips = heuristic_candidates(lib, per_ep=2 if use_llm else 3)
    if use_llm:
        mined, pr = llm_candidates(lib, llm, bible, progress=say)
        clips += mined
        problems += pr
    for c in clips:
        c.scores["rules"] = rule_score(lib, c)
        c.scores["heuristic"] = c.scores["rules"]
    clips = dedupe(clips)
    if use_llm:
        say("Building cross-episode montages")
        seeds = sorted([c for c in clips if c.source == "llm"], key=lambda c: -c.scores["rules"])
        mont, pr = llm_montages(lib, llm, bible, seeds)
        problems += pr
        for c in mont:
            c.scores["rules"] = rule_score(lib, c)
        clips += mont

    say(f"Describing {len(clips)} clips")
    for c in clips:
        describe(lib, c)

    # shortlist before the (slower) LLM tagging / judging
    clips.sort(key=lambda c: -(c.scores["rules"] + (1.0 if c.source.startswith("llm") else 0)))
    short = clips[: max_judge] if use_llm else clips
    if use_llm:
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def work(c):
            c.judge, c.tags = review(lib, c, llm)

        with ThreadPoolExecutor(max_workers=int(os.environ.get("LLM_CONCURRENCY", "2"))) as ex:
            futs = [ex.submit(work, c) for c in short]
            for i, _ in enumerate(as_completed(futs), 1):
                say(f"Stranger test {i}/{len(short)} clips")
    for c in short:
        p, name = model.predict(c, use_tags=use_llm and "error" not in c.tags)
        c.scores["model_prob"] = None if p is None else round(p, 4)
        c.scores["model"] = name

    final_scores(short, model, use_llm)
    ranked = diversify(short, top_n)
    for i, c in enumerate(ranked, 1):
        c.id = f"P{i:02d}"

    result = {
        "show_folder": str(folder), "episodes": len(episodes), "beats": sum(len(e.beats) for e in episodes),
        "mode": "llm" if use_llm else "heuristic", "llm": llm.describe(),
        "success_model": model.meta.get("summary", "not trained") if model.available else "not trained",
        "bible": bible, "promos": [clip_to_dict(c) for c in ranked[:top_n]],
        "all_candidates": len(clips), "rejected_or_problems": problems, "parse_warnings": warnings,
        "llm_usage": {"calls": llm.calls, "cache_hits": llm.cache_hits, "tokens_in": llm.tokens_in,
                      "tokens_out": llm.tokens_out, "served_by": llm.served_by,
                      "failover_events": llm.events[-30:]},
        "runtime_s": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if out_dir:
        write_outputs(result, Path(out_dir))
    say("Done")
    return result


def clip_to_dict(c) -> dict:
    return {"id": c.id, "title": c.title, "source": c.source, "duration_s": c.duration,
            "segments": [s.to_dict() for s in c.segments], "hook": c.hook, "cliffhanger": c.cliffhanger,
            "why": c.why, "landing_episode": c.landing_episode, "flags": c.flags, "scores": c.scores,
            "tags": c.tags, "judge": c.judge,
            "features": {k: v for k, v in c.features.items() if not k.startswith("_")},
            "first_3s_text": c.features.get("_first3_text", ""), "transcript": c.transcript,
            "cut_list": [f"ep{s.ep:02d}: ffmpeg -ss {s.start:.2f} -to {s.end:.2f}" for s in c.segments]}


def write_outputs(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "promos.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "title", "segments (episode start-end)", "duration_s", "hook", "cliffhanger",
                    "why", "final_score", "model_prob", "editor_score", "landing_episode", "flags", "source"])
        for i, p in enumerate(res["promos"], 1):
            segs = " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
            sc = p["scores"]
            w.writerow([i, p["id"], p["title"], segs, p["duration_s"], p["hook"], p["cliffhanger"], p["why"],
                        sc.get("final"), sc.get("model_prob"), sc.get("editor"), p["landing_episode"],
                        ";".join(p["flags"]), p["source"]])
    lines = [f"# Promo report: {Path(res['show_folder']).name}", "",
             f"- Episodes: {res['episodes']} | beats: {res['beats']} | mode: {res['mode']} | LLM: {res['llm']}",
             f"- Success model: {res['success_model']}",
             f"- Candidates considered: {res['all_candidates']} | runtime: {res['runtime_s']} s", "",
             f"**Premise:** {res['bible'].get('premise', '')}", ""]
    for i, p in enumerate(res["promos"], 1):
        segs = " + ".join(f"Ep {s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
        sc = p["scores"]
        lines += [f"## {i}. {p['title']}  ({p['duration_s']:.0f}s)", f"- **Clip:** {segs}",
                  f"- **Hook (0-3s):** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}",
                  f"- **Why it works:** {p['why']}",
                  f"- **Score:** {sc.get('final')} (model {sc.get('model_prob')}, editor {sc.get('editor')}, "
                  f"rules {sc.get('rules_norm')}, penalty {sc.get('penalty')})",
                  f"- **Landing episode:** {p['landing_episode']} | flags: {', '.join(p['flags']) or 'none'}", "",
                  "<details><summary>Transcript</summary>", ""]
        lines += [f"    [{t['t']}] {t['kind']} {t['speaker']}: {t['text']}" for t in p["transcript"]
                  if t["kind"] != "CAMERA"]
        lines += ["", "</details>", ""]
    if res["rejected_or_problems"]:
        lines += ["## Rejected candidates / problems", ""] + [f"- {x}" for x in res["rejected_or_problems"]]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
