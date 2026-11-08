"""Parse microdrama screenplay .txt files into structured, timestamped beats.

Pure code, no AI. Tolerant to the format variations seen in the input pack
(upper/lower-case headers, missing scene end, O.S. tags, multi-line text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?"
RE_HEADER = re.compile(r"^\*\*(?P<label>[^*]+?)\s*:?\s*\*\*\s*:?\s*$")
RE_TIMES = re.compile(rf"Start:\s*\(?\s*(?P<s>{TS})\s*\)?\s*\|\s*End:\s*\(?\s*(?P<e>{TS})\s*\)?", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_SCENE_HEADING = re.compile(r"^scene heading:\s*(.+)$", re.I)
RE_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.+)$", re.I)
RE_CONT = re.compile(r"^CONTINUITY NOTES:\s*(.+)$", re.I)
NON_DIALOGUE = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX",
                "SOUND": "SFX", "MUSIC": "MUSIC", "VFX": "VFX", "TEXT": "TEXT", "SUPER": "TEXT"}


def to_sec(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return round(sec, 3)


def fmt(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    beat_id: str
    ep: int
    scene: int
    idx: int
    kind: str            # DIALOGUE | CAMERA | ACTION | SFX | MUSIC | VFX | TEXT
    speaker: str         # raw speaker label for dialogue, "" otherwise
    tone: str            # delivery tag, e.g. "Cold, Commanding"
    off_screen: bool
    start: float
    end: float
    text: str

    def line(self) -> str:
        """Compact one-line form used in LLM prompts."""
        who = f"{self.speaker} ({self.tone})" if self.kind == "DIALOGUE" and self.tone else self.speaker
        body = f'{who}: "{self.text}"' if self.kind == "DIALOGUE" else self.text
        return f"[{self.beat_id} {fmt(self.start)}-{fmt(self.end)}] {self.kind}: {body}"


@dataclass
class Episode:
    ep: int
    file: str
    logline: str = ""
    continuity: str = ""
    scenes: dict = field(default_factory=dict)   # scene number -> heading
    beats: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _split_label(label: str):
    """'LADY JULIANA (Confused, Breathless)' -> ('LADY JULIANA', 'Confused, Breathless', False)."""
    label = label.strip().rstrip(":").strip()
    parens = re.findall(r"\(([^)]*)\)", label)
    name = re.sub(r"\([^)]*\)", "", label).strip()
    off = any(p.strip().upper() in {"O.S.", "OS", "V.O.", "VO", "O.C."} for p in parens)
    tone = ", ".join(p.strip() for p in parens if p.strip().upper() not in {"O.S.", "OS", "V.O.", "VO", "O.C."})
    return name, tone, off


def parse_episode(path: str | Path, fallback_ep: int | None = None) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ep_num = fallback_ep
    m = re.search(r"episode[-_ ]?0*(\d+)", path.stem, re.I)
    if m:
        ep_num = int(m.group(1))
    ep = Episode(ep=ep_num or 0, file=path.name)

    scene, i, idx = 0, 0, 0
    while i < len(lines):
        raw = lines[i].strip()
        if (mm := RE_EP_NUM.match(raw)):
            ep.ep = int(mm.group(1))
        elif (mm := RE_LOGLINE.match(raw)):
            ep.logline = mm.group(1).strip()
        elif (mm := RE_CONT.match(raw)):
            ep.continuity = mm.group(1).strip()
        elif (mm := RE_SCENE.match(raw)):
            scene = int(mm.group(1))
            ep.scenes.setdefault(scene, "")
        elif (mm := RE_SCENE_HEADING.match(raw)):
            ep.scenes[scene] = mm.group(1).strip()
        elif (mm := RE_HEADER.match(raw)):
            label = mm.group("label")
            # timestamps on the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = RE_TIMES.search(lines[j]) if j < len(lines) else None
            if not tm:
                ep.warnings.append(f"line {i+1}: header without timestamps: {raw[:60]}")
                i += 1
                continue
            # body text until blank line or next header
            k, body = j + 1, []
            while k < len(lines) and lines[k].strip() and not RE_HEADER.match(lines[k].strip()) \
                    and not lines[k].strip().startswith("---"):
                body.append(lines[k].strip())
                k += 1
            name, tone, off = _split_label(label)
            kind = NON_DIALOGUE.get(name.upper(), "DIALOGUE")
            text = " ".join(body).strip().strip('"').strip("“”").strip()
            s, e = to_sec(tm.group("s")), to_sec(tm.group("e"))
            if e < s:
                ep.warnings.append(f"line {j+1}: end before start ({tm.group(0)})")
                s, e = e, s
            idx += 1
            ep.beats.append(Beat(beat_id=f"E{ep.ep:02d}-B{idx:03d}", ep=ep.ep, scene=scene, idx=idx, kind=kind,
                                 speaker=name if kind == "DIALOGUE" else "", tone=tone if kind == "DIALOGUE" else "",
                                 off_screen=off, start=s, end=e, text=text))
            i = k
            continue
        i += 1

    # beat ids use the final episode number (header may come after filename guess)
    for b in ep.beats:
        b.ep = ep.ep
        b.beat_id = f"E{ep.ep:02d}-B{b.idx:03d}"
    if not ep.beats:
        ep.warnings.append("no beats parsed")
    return ep


def parse_folder(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"))
    eps = [parse_episode(f, fallback_ep=n + 1) for n, f in enumerate(files)]
    return sorted(eps, key=lambda e: e.ep)
