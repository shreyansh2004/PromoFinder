"""Score and rank clips.

final = w_model * P(high reach | features)       <- model trained on real short-drama ads
      + w_editor * cold-viewer judge (LLM)        <- standalone / hook / cliffhanger, 1-5
      + w_rules * rule-based hook+cliff score     <- transparent heuristics
      - penalties (late hook, repaired duration)
then a diversity pass (MMR) so the top list is not five clips of one scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import prompts as P
from .features import tag_features
from .heuristics import cliff_score, hook_score

ART = Path(__file__).resolve().parent.parent / "model" / "artifacts"


class SuccessModel:
    """Loads the trained model(s). 'full' uses AI tags + text/visual features, 'basic' only measurable ones."""

    def __init__(self, art_dir: Path = ART):
        self.models, self.meta = {}, {}
        mp = art_dir / "model_meta.json"
        if mp.exists():
            self.meta = json.loads(mp.read_text(encoding="utf-8"))
            for name in ("full", "basic"):
                f = art_dir / f"model_{name}.json"  # plain-text model: scaler + logistic regression
                if f.exists():
                    self.models[name] = json.loads(f.read_text(encoding="utf-8"))

    @property
    def available(self) -> bool:
        return bool(self.models)

    def weight(self, name: str | None) -> float:
        """Ranking weight earned by out-of-sample skill: min(0.5, 2 x (within-app AUC - 0.5))."""
        return float(self.meta.get(name, {}).get("weight", 0.0)) if name else 0.0

    def predict(self, clip, use_tags: bool) -> tuple[float | None, str | None]:
        name = "full" if use_tags and "full" in self.models else ("basic" if "basic" in self.models else None)
        if not name:
            return None, None
        m = self.models[name]
        feats = {**clip.features, **(tag_features(clip.tags) if name == "full" else {})}
        x = np.array([float(feats.get(c, 0) or 0) for c in m["features"]])
        z = (x - np.array(m["mean"])) / np.where(np.array(m["scale"]) == 0, 1, np.array(m["scale"]))
        logit = float(z @ np.array(m["coef"]) + m["intercept"])
        return 1 / (1 + np.exp(-logit)), name


def rule_score(lib, clip) -> float:
    tl_first = [b for s in clip.segments for b in lib.beats_in(s.ep, s.start, s.start + 3.0)]
    last = clip.segments[-1]
    tl_last = lib.beats_in(last.ep, last.end - 6.0, last.end)
    h = max((hook_score(b) for b in tl_first), default=0)
    c = max((cliff_score(b, lib.eps[last.ep]) for b in tl_last), default=0)
    return round(h + c, 2)


REVIEW_JUDGE_KEYS = ("what_is_happening", "open_question", "hook", "cliffhanger", "standalone", "confusing")


def review(lib, clip, llm) -> tuple[dict, dict]:
    """One cold call: stranger test (judge) + tags, from the clip transcript only. Returns (judge, tags)."""
    lines = []
    for t in clip.transcript:
        if t["kind"] == "DIALOGUE":
            lines.append(f"[{t['t']}] {t['speaker']}{' (' + t['tone'] + ')' if t['tone'] else ''}: \"{t['text']}\"")
        elif t["kind"] in ("ACTION", "SFX", "CAMERA") and (t["kind"] != "CAMERA" or t["t"] < "00:03"):
            lines.append(f"[{t['t']}] ({t['text']})")
    try:
        r = llm.json(P.REVIEW_SYSTEM, P.REVIEW_USER.format(dur=clip.duration, transcript="\n".join(lines)),
                     validate=lambda o: [int(o[k]) for k in ("hook", "cliffhanger", "standalone")])
    except Exception as e:
        return {"error": str(e)[:200]}, {"error": str(e)[:200]}
    j = {k: r.get(k) for k in REVIEW_JUDGE_KEYS}
    for k in ("hook", "cliffhanger", "standalone"):
        j[k] = max(1, min(5, int(j[k])))
    tags = {k: v for k, v in r.items() if k not in REVIEW_JUDGE_KEYS}
    return j, tags


def judge(lib, clip, llm) -> dict:
    lines = []
    for t in clip.transcript:
        if t["kind"] == "DIALOGUE":
            lines.append(f"[{t['t']}] {t['speaker']}{' (' + t['tone'] + ')' if t['tone'] else ''}: \"{t['text']}\"")
        elif t["kind"] in ("ACTION", "SFX"):
            lines.append(f"[{t['t']}] ({t['text']})")
    try:
        j = llm.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(dur=clip.duration, transcript="\n".join(lines)),
                     validate=lambda o: [int(o[k]) for k in ("hook", "cliffhanger", "standalone")])
        for k in ("hook", "cliffhanger", "standalone"):
            j[k] = max(1, min(5, int(j[k])))
        return j
    except Exception as e:
        return {"error": str(e)[:200]}


def earned_weights() -> dict | None:
    """Skill each signal earned on real ads (model/validate_signals.py): min(0.5, 2 x (within-app AUC - 0.5))."""
    f = ART / "signal_validation.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get("earned_weights")


FLOOR = 0.02  # every signal keeps a small say, so a clip is never ranked by one signal alone


def final_scores(clips, model: SuccessModel, use_llm: bool) -> None:
    rules = np.array([c.scores.get("rules", 0) for c in clips], dtype=float)
    r_norm = (rules - rules.min()) / (np.ptp(rules) or 1)
    ew = earned_weights()
    for c, rn in zip(clips, r_norm):
        p = c.scores.get("model_prob")
        j = c.judge if c.judge and "error" not in c.judge else None
        editor = (j["hook"] + j["cliffhanger"] + j["standalone"]) / 15 if j else None
        if use_llm and editor is None:
            editor = 0.5  # review failed: neutral, so rules alone cannot push the clip up
            if "review_failed" not in c.flags:
                c.flags.append("review_failed")
        if ew:  # weights earned on real ads, normalised over the signals this run has
            raw = {"model": max(model.weight(c.scores.get("model")), FLOOR) if p is not None else 0.0,
                   "editor": max(ew["editor"], FLOOR) if editor is not None else 0.0,
                   "rules": max(ew["rules"], FLOOR)}
        else:  # no validation file: original priors
            raw = {"model": model.weight(c.scores.get("model")) if p is not None else 0.0,
                   "editor": 0.5 if editor is not None else 0.0, "rules": 0.0}
            raw["rules"] = 1 - raw["model"] - raw["editor"]
        tot = sum(raw.values()) or 1.0
        wts = {k: v / tot for k, v in raw.items()}
        parts = [(wts["model"], p or 0.0), (wts["editor"], editor or 0.0), (wts["rules"], rn)]
        c.scores["weights"] = {k: round(v, 3) for k, v in wts.items()}
        s = sum(w * v for w, v in parts)
        pen = 0.12 * ("late_hook" in c.flags) + 0.04 * ("extended_back" in c.flags)
        if j and j["standalone"] <= 2:
            pen += 0.15
            if "not_standalone" not in c.flags:
                c.flags.append("not_standalone")
        if j and j["cliffhanger"] <= 2:  # a resolved ending fails the brief, whatever else is good
            pen += 0.12
            if "weak_cliffhanger" not in c.flags:
                c.flags.append("weak_cliffhanger")
        c.scores.update({"rules_norm": round(float(rn), 3), "editor": None if editor is None else round(editor, 3),
                         "penalty": round(pen, 3), "final": round(float(s - pen), 4)})


def diversify(clips, top_n: int, lam: float = 0.85):
    """Maximal-marginal-relevance re-ranking over episode, hook type and trope."""
    pool, chosen = sorted(clips, key=lambda c: -c.scores["final"]), []

    def sim(a, b):
        same_ep = len({s.ep for s in a.segments} & {s.ep for s in b.segments}) > 0
        same_hook = a.tags.get("hook_type") and a.tags.get("hook_type") == b.tags.get("hook_type")
        return 0.6 * same_ep + 0.4 * bool(same_hook)

    while pool and len(chosen) < top_n:
        best = max(pool, key=lambda c: lam * c.scores["final"] - (1 - lam) * max((sim(c, k) for k in chosen), default=0))
        chosen.append(best)
        pool.remove(best)
    return chosen + pool
