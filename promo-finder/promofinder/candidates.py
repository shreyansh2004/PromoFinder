"""Find candidate promos: LLM proposals (beat IDs only) or a transparent heuristic fallback."""
from __future__ import annotations

import json
from collections import Counter

from . import prompts as P
from .clips import Clip, Library, MAX_S, MIN_S, fix_and_validate, make_segment, norm_beat_id, trim_montage
from .heuristics import hook_score, score_episode
from .parser import Episode


# ---------------------------------------------------------------- series bible
def build_bible(episodes: list[Episode], llm) -> dict:
    loglines = "\n".join(f"Ep {e.ep}: {e.logline}" for e in episodes)
    if not llm or not llm.available:
        return {"premise": episodes[0].logline if episodes else "", "characters": [], "turning_points": [],
                "continuity_issues": [], "source": "heuristic"}
    spk = {}
    for e in episodes:
        for b in e.beats:
            if b.kind == "DIALOGUE":
                spk.setdefault(b.speaker, set()).add(e.ep)
    speakers = "\n".join(f"{k}: eps {','.join(map(str, sorted(v)))}" for k, v in sorted(spk.items(), key=lambda x: -len(x[1])))
    bible = llm.json(P.BIBLE_SYSTEM, P.BIBLE_USER.format(loglines=loglines, speakers=speakers[:6000]))
    bible["source"] = "llm"
    return bible


def _char_line(bible: dict) -> str:
    return "; ".join(f"{c.get('name')} ({c.get('role')}; also called {', '.join(c.get('aliases', [])[:4])})"
                     for c in bible.get("characters", [])[:12]) or "unknown"


# ---------------------------------------------------------------- heuristic fallback
def heuristic_candidates(lib: Library, per_ep: int = 3) -> list[Clip]:
    out = []
    for ep in lib.eps.values():
        sc = score_episode(ep)
        anchors = sorted([b for b in ep.beats if b.kind in ("DIALOGUE", "ACTION")],
                         key=lambda b: -sc[b.beat_id]["hook"])[:8]
        made = 0
        for a in anchors:
            ends = [b for b in ep.beats if b.kind in ("DIALOGUE", "ACTION", "SFX")
                    and MIN_S + 2 <= b.end - a.start <= MAX_S - 4]
            if not ends:
                continue
            best = max(ends, key=lambda b: sc[b.beat_id]["cliff"] + 0.02 * (b.end - a.start))
            seg = make_segment(lib, a.beat_id, best.beat_id)
            clip = Clip([seg], "heuristic", title=f"Ep {ep.ep}: {a.text[:50]}",
                        hook=f"{a.speaker + ': ' if a.speaker else ''}{a.text[:120]}",
                        cliffhanger=f"{best.speaker + ': ' if best.speaker else ''}{best.text[:120]}",
                        why="High rule-based hook score at the open and cliffhanger score at the cut.")
            clip.scores["heuristic"] = round(sc[a.beat_id]["hook"] + sc[best.beat_id]["cliff"], 2)
            clip = fix_and_validate(lib, clip)
            if clip:
                out.append(clip)
                made += 1
            if made >= per_ep:
                break
    return out


# ---------------------------------------------------------------- LLM proposals
def _beats_block(ep: Episode, max_cam_chars: int = 110) -> str:
    rows = []
    for b in ep.beats:
        line = b.line()
        if b.kind == "CAMERA" and len(line) > max_cam_chars + 30:
            line = line[: max_cam_chars + 30] + "..."
        rows.append(line)
    return f"--- EPISODE {ep.ep} (duration {ep.duration:.0f}s) ---\n" + "\n".join(rows)


def llm_candidates(lib: Library, llm, bible: dict, window: int = 4, per_ep: str = "2-3",
                   progress=None) -> tuple[list[Clip], list[str]]:
    eps = [lib.eps[k] for k in sorted(lib.eps)]
    clips, problems = [], []
    for i in range(0, len(eps), window):
        chunk = eps[i:i + window]
        so_far = " ".join(f"Ep {e.ep}: {e.logline}" for e in eps[max(0, i - 6):i]) or "This is the start of the series."
        user = P.MINE_USER.format(premise=bible.get("premise", ""), characters=_char_line(bible),
                                  story_so_far=so_far, ep_range=f"{chunk[0].ep}-{chunk[-1].ep}", per_ep=per_ep,
                                  beats="\n\n".join(_beats_block(e) for e in chunk))
        if progress:
            progress(f"Mining episodes {chunk[0].ep}-{chunk[-1].ep}")
        try:
            res = llm.json(P.MINE_SYSTEM, user, validate=lambda o: o["promos"])
        except Exception as e:
            problems.append(f"episodes {chunk[0].ep}-{chunk[-1].ep}: {e}")
            continue
        for p in res.get("promos", []):
            seg = make_segment(lib, str(p.get("start_beat")), str(p.get("end_beat")))
            if seg is None:
                problems.append(f"invalid beat ids {p.get('start_beat')}..{p.get('end_beat')} (rejected)")
                continue
            clip = Clip([seg], "llm", title=p.get("title", ""), hook=p.get("hook", ""),
                        cliffhanger=p.get("cliffhanger", ""), why=p.get("why", ""))
            clip.scores["llm_hook_beat"] = p.get("hook_beat")
            fixed = fix_and_validate(lib, clip)
            if fixed is None and p.get("hook_beat") and p.get("cliffhanger_beat"):
                # too long or short: retry with the model's own hook beat -> cliffhanger beat (a tighter cut)
                seg2 = make_segment(lib, str(p["hook_beat"]), str(p["cliffhanger_beat"]))
                if seg2 is not None:
                    clip.segments = [seg2]
                    fixed = fix_and_validate(lib, clip)
                    if fixed:
                        fixed.flags.append("tightened_to_hook_and_cliffhanger")
            if fixed is None:
                # still too long: keep the model's cliffhanger ending, re-open on the strongest hook that fits
                end_b = lib.beats.get(norm_beat_id(p.get("cliffhanger_beat") or p.get("end_beat")))
                if end_b:
                    ep_obj = lib.eps[end_b.ep]
                    opts = [b for b in ep_obj.beats if b.kind in ("DIALOGUE", "ACTION")
                            and MIN_S + 2 <= end_b.end - b.start <= MAX_S - 3]
                    if opts:
                        best = max(opts, key=lambda b: hook_score(b))
                        seg3 = make_segment(lib, best.beat_id, end_b.beat_id)
                        clip.segments = [seg3] if seg3 else []
                        fixed = fix_and_validate(lib, clip) if seg3 else None
                        if fixed:
                            fixed.flags.append("reopened_on_stronger_hook")
            if fixed is None:
                problems.append(f"Ep {seg.ep} {p.get('start_beat')}..{p.get('end_beat')}: outside 20-60 s after snapping (rejected)")
                continue
            clips.append(fixed)
    return clips, problems


def llm_montages(lib: Library, llm, bible: dict, seeds: list[Clip], n: int = 3) -> tuple[list[Clip], list[str]]:
    if not seeds:
        return [], []
    moments = []
    for c in seeds[:20]:
        s = c.segments[0]
        ids = [f"{b.beat_id}@{b.start - s.start:.0f}s" for b in lib.beats_in(s.ep, s.start, s.end)
               if b.kind in ("DIALOGUE", "ACTION")]
        moments.append(f"- Ep {s.ep} ({s.dur:.0f}s) | {c.title} | hook: {c.hook} | cliff: {c.cliffhanger} | "
                       f"beats (offset from moment start): {', '.join(ids)}")
    try:
        res = llm.json(P.MONTAGE_SYSTEM, P.MONTAGE_USER.format(premise=bible.get("premise", ""), n=n,
                                                               moments="\n".join(moments)),
                       validate=lambda o: o["promos"])
    except Exception as e:
        return [], [f"montage: {e}"]
    out, problems = [], []
    for p in res.get("promos", []):
        segs = [make_segment(lib, str(s.get("start_beat")), str(s.get("end_beat"))) for s in p.get("segments", [])]
        if not segs or any(s is None for s in segs):
            problems.append("montage with invalid beat ids (rejected)")
            continue
        raw = Clip(segs, "llm-montage", title=p.get("title", ""), hook=p.get("hook", ""),
                   cliffhanger=p.get("cliffhanger", ""), why=p.get("why", ""))
        if raw.duration > MAX_S:
            trim_montage(lib, raw)
            raw.flags.append("montage_trimmed")
        clip = fix_and_validate(lib, raw)
        if clip is None:
            problems.append(f"montage '{p.get('title')}' outside 20-60 s (rejected)")
            continue
        out.append(clip)
    return out, problems


def dedupe(clips: list[Clip], max_share: float = 0.4) -> list[Clip]:
    """Drop a clip if more than 40% of the shorter clip overlaps one already kept (same scene twice).
    AI-proposed clips are preferred over rule-based ones, then higher rule score."""
    kept = []
    for c in sorted(clips, key=lambda c: -(c.scores.get("heuristic", 0) + (2 if c.source.startswith("llm") else 0))):
        dup = False
        for k in kept:
            if len(c.segments) == 1 and len(k.segments) == 1 and c.segments[0].ep == k.segments[0].ep:
                a, b = c.segments[0], k.segments[0]
                inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
                if inter / max(min(a.dur, b.dur), 1e-6) > max_share:
                    dup = True
                    break
        if not dup:
            kept.append(c)
    return kept
