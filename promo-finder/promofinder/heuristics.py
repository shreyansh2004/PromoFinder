"""Rule-based beat scores (no AI). Used to seed candidates, as a ranking signal and as the
no-API fallback. Every rule is visible here, so an editor can see why a beat scored high."""
from __future__ import annotations

import re

from .features import CLIFF_WORDS, CLOSEUP, INTENSE_TONES, VIOLENT_REVEAL, words
from .parser import Episode

RESOLVED = re.compile(r"fades? (?:out|to black|away)|fade-out|into the sunlight|hand[- ]in[- ]hand|together|peace(?:ful)?|"
                      r"relie[fv]|smil|embrace|triumphant|gently|tender|warmth|sleep|calm|settles", re.I)


def hook_score(b) -> float:
    s = 0.0
    if b.kind == "DIALOGUE":
        n = len(words(b.text))
        s += 1.0
        s += 1.2 * bool(INTENSE_TONES.search(b.tone))
        s += 0.8 * ("!" in b.text) + 0.6 * ("?" in b.text)
        s += 0.6 * (0 < n <= 7)
        s += 0.8 * bool(VIOLENT_REVEAL.search(b.text))
    elif b.kind in ("ACTION", "CAMERA"):
        s += 1.5 * bool(VIOLENT_REVEAL.search(b.text))
        s += 0.5 * bool(CLOSEUP.search(b.text)) * bool(INTENSE_TONES.search(b.text))
        s += 0.4 * bool(INTENSE_TONES.search(b.text))
    elif b.kind == "SFX":
        s += 0.7 * bool(VIOLENT_REVEAL.search(b.text) or CLIFF_WORDS.search(b.text))
    return round(s, 2)


def cliff_score(b, ep: Episode) -> float:
    s = 0.0
    if b.kind == "DIALOGUE":
        s += 1.0 * ("?" in b.text) + 0.6 * ("!" in b.text)
        s += 1.0 * bool(CLIFF_WORDS.search(b.text))
        s += 0.6 * bool(INTENSE_TONES.search(b.tone))
    elif b.kind in ("ACTION", "CAMERA", "SFX"):
        s += 1.2 * bool(CLIFF_WORDS.search(b.text)) + 0.6 * bool(VIOLENT_REVEAL.search(b.text))
    # a calm or happy fade-out is a resolution, not a cliffhanger (found in the first AI test run)
    s -= 1.8 * bool(RESOLVED.search(b.text))
    return round(s, 2)


def score_episode(ep: Episode) -> dict:
    return {b.beat_id: {"hook": hook_score(b), "cliff": cliff_score(b, ep)} for b in ep.beats}
