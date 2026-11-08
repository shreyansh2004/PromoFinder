"""Turn beat-ID ranges into exact, validated clips (pure code, no AI).

The LLM only ever names beat IDs; every timestamp below comes from the screenplay.
Rules: cut on CAMERA-shot boundaries, never mid-line, 20-60 s total, hook inside the first 3 s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import CLOSEUP, text_features, timing_features, visual_features
from .heuristics import hook_score
from .parser import Episode, fmt

MIN_S, MAX_S, HOOK_S, SNAP_S = 20.0, 60.0, 3.0, 1.5


@dataclass
class Segment:
    ep: int
    start: float
    end: float
    start_beat: str
    end_beat: str

    @property
    def dur(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        return {"episode": self.ep, "start": fmt(self.start), "end": fmt(self.end), "start_s": self.start,
                "end_s": self.end, "duration_s": self.dur, "start_beat": self.start_beat, "end_beat": self.end_beat}


@dataclass
class Clip:
    segments: list
    source: str                       # "llm" | "heuristic" | "llm-montage"
    title: str = ""
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    trope: str = ""
    flags: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    landing_episode: int | None = None
    id: str = ""

    @property
    def duration(self) -> float:
        return round(sum(s.dur for s in self.segments), 2)

    def key(self):
        return tuple((s.ep, round(s.start, 1), round(s.end, 1)) for s in self.segments)


class Library:
    """Index over all episodes for fast beat lookups."""

    def __init__(self, episodes: list[Episode]):
        self.eps = {e.ep: e for e in episodes}
        self.beats = {b.beat_id: b for e in episodes for b in e.beats}

    def shots(self, ep: int):
        return [b for b in self.eps[ep].beats if b.kind == "CAMERA"]

    def beats_in(self, ep: int, start: float, end: float):
        return [b for b in self.eps[ep].beats if b.end > start + 1e-6 and b.start < end - 1e-6]


def _snap_start(lib: Library, ep: int, t: float) -> float:
    cands = [s.start for s in lib.shots(ep) if s.start <= t + 1e-6 and t - s.start <= SNAP_S]
    return max(cands) if cands else t


def _finish_lines(e, t: float) -> float:
    """Extend t until no line of dialogue is still running at t (lines can overlap)."""
    changed = True
    while changed:
        changed = False
        for b in e.beats:
            if b.kind == "DIALOGUE" and b.start < t - 0.05 < b.end and b.end > t + 1e-6:
                t, changed = b.end, True
    return t


def _snap_end(lib: Library, ep: int, t: float) -> float:
    e = lib.eps[ep]
    t = _finish_lines(e, t)  # never cut a line of dialogue
    cands = [s.end for s in lib.shots(ep) if s.end >= t - 1e-6 and s.end - t <= SNAP_S]
    snapped = min(cands) if cands else t
    if _finish_lines(e, snapped) != snapped:  # the shot end would land inside the next line: don't snap
        snapped = t
    return min(snapped, e.duration)


def trim_montage(lib: Library, clip: "Clip", target: float = 55.0) -> None:
    """Shrink an over-long montage: keep the opening of the first segment (hook) and the ending of the
    last (cliffhanger); every segment gets an equal share of the time budget, cut on beat/shot edges."""
    n = len(clip.segments)
    cap = target / n
    for i, s in enumerate(clip.segments):
        if s.dur <= cap:
            continue
        e = lib.eps[s.ep]
        if i == n - 1:  # keep the end
            starts = [b.start for b in e.beats if s.end - cap <= b.start < s.end]
            if starts:
                s.start = round(_snap_start(lib, s.ep, min(starts)), 2)
        else:  # keep the start
            ends = [b.end for b in e.beats if s.start < b.end <= s.start + cap and b.kind in ("DIALOGUE", "ACTION")]
            if ends:
                s.end = round(_snap_end(lib, s.ep, max(ends)), 2)


def norm_beat_id(x) -> str:
    """'E01-B030@12s' or ' e1-b30 ' -> 'E01-B030' (models sometimes echo annotations)."""
    import re
    m = re.search(r"E0*(\d+)\s*-\s*B0*(\d+)", str(x or ""), re.I)
    return f"E{int(m.group(1)):02d}-B{int(m.group(2)):03d}" if m else str(x)


def make_segment(lib: Library, start_beat: str, end_beat: str) -> Segment | None:
    start_beat, end_beat = norm_beat_id(start_beat), norm_beat_id(end_beat)
    a, b = lib.beats.get(start_beat), lib.beats.get(end_beat)
    if not a or not b or a.ep != b.ep:
        return None
    if b.end < a.start:
        a, b = b, a
    s = _snap_start(lib, a.ep, a.start)
    e = _snap_end(lib, a.ep, max(b.end, a.end))
    return Segment(a.ep, round(s, 2), round(e, 2), a.beat_id, b.beat_id)


def timeline(lib: Library, clip: Clip):
    """Beats inside the clip with clip-relative times."""
    out, off = [], 0.0
    for seg in clip.segments:
        for b in lib.beats_in(seg.ep, seg.start, seg.end):
            rs = max(b.start, seg.start) - seg.start + off
            re_ = min(b.end, seg.end) - seg.start + off
            out.append((rs, re_, b))
        off += seg.dur
    return out


def first_hook_offset(lib: Library, clip: Clip) -> float | None:
    for rs, _, b in timeline(lib, clip):
        if b.kind == "DIALOGUE" or (b.kind in ("ACTION", "SFX") and hook_score(b) >= 1.5):
            return rs
    return None


def fix_and_validate(lib: Library, clip: Clip) -> Clip | None:
    """Enforce duration and hook rules. Returns None if the clip cannot be made valid."""
    segs = [s for s in clip.segments if s and s.dur > 0.5]
    if not segs:
        return None
    clip.segments = segs
    # hook must land in the first 3 s: trim dead lead-in on the first segment if possible
    off = first_hook_offset(lib, clip)
    if off is not None and off > HOOK_S:
        first = clip.segments[0]
        tl = [x for x in timeline(lib, clip) if x[0] == off]
        if tl:
            new_start = _snap_start(lib, first.ep, tl[0][2].start)
            if clip.duration - (new_start - first.start) >= MIN_S:
                first.start = round(new_start, 2)
                clip.flags.append("trimmed_lead_in")
    # too short: extend the first segment backwards shot by shot (forward could resolve the cliffhanger)
    guard = 0
    while clip.duration < MIN_S and guard < 40:
        first = clip.segments[0]
        prev = [s.start for s in lib.shots(first.ep) if s.start < first.start - 1e-6]
        if not prev:
            break
        first.start = round(max(prev), 2)
        clip.flags.append("extended_back") if "extended_back" not in clip.flags else None
        guard += 1
    if clip.duration < MIN_S:
        return None
    if clip.duration > MAX_S:
        return None
    off = first_hook_offset(lib, clip)
    if off is None or off > HOOK_S:
        clip.flags.append("late_hook")
    return clip


def describe(lib: Library, clip: Clip) -> None:
    """Fill transcript, script-derived features (same definitions as the ad dataset), landing ep."""
    tl = timeline(lib, clip)
    clip.transcript = [{"t": fmt(rs), "ep": b.ep, "beat": b.beat_id, "kind": b.kind,
                        "speaker": b.speaker, "tone": b.tone, "text": b.text} for rs, _, b in tl]
    dia = [(rs, b) for rs, _, b in tl if b.kind == "DIALOGUE"]
    first3 = " ".join(b.text for rs, b in dia if rs < 3.0)
    hook15 = " ".join(b.text for rs, b in dia if rs < 15.0)
    ending = " ".join(b.text for rs, b in dia if rs >= clip.duration - 20.0)
    speech_start = dia[0][0] if dia else None
    cam3 = [(rs, b) for rs, _, b in tl if b.kind == "CAMERA" and rs < 3.0]
    n_cuts = sum(1 for rs, b in cam3 if rs > 0.05) + sum(1 for i, s in enumerate(clip.segments[1:], 1)
                                                         if sum(x.dur for x in clip.segments[:i]) < 3.0)
    closeup = any(CLOSEUP.search(b.text) for _, b in cam3)
    clip.features = {**text_features(first3, hook15, ending), **timing_features(speech_start),
                     **visual_features(n_cuts, closeup)}
    clip.features["_first3_text"], clip.features["_hook15_text"], clip.features["_ending_text"] = first3, hook15, ending
    clip.features["_visual3_text"] = " ".join(b.text for rs, _, b in tl if rs < 3.0 and b.kind in ("CAMERA", "ACTION"))
    last = clip.segments[-1]
    ep = lib.eps[last.ep]
    clip.landing_episode = last.ep if ep.duration - last.end > 5.0 else min(last.ep + 1, max(lib.eps))
