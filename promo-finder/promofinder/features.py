"""Features shared by training (real ads) and prediction (script promos).

The SAME functions compute features for both, so the model sees identically defined inputs:
  ads     -> from Whisper transcript (first 3 s / 15 s / last 20 s) + measured video signals
  promos  -> from the screenplay beats inside the clip (dialogue timing, CAMERA/ACTION lines)
"""
from __future__ import annotations

import re

import numpy as np

# ---------- vocabularies for the transparent, rule-based signals
INTENSE_TONES = re.compile(r"gasp|scream|shriek|shock|furious|outrag|sob|hysteric|cold|mock|venom|desperat|plead|"
                           r"trembl|enrag|livid|horrif|cruel|gloat|sneer|menac|threat|snarl|roar|crying|tears|"
                           r"panic|terrif|seeth|bitter|vicious|disgust|unhinged|breathless", re.I)
VIOLENT_REVEAL = re.compile(r"\bslap|strike|struck|punch|kick|shove|grab|drag|blood|knife|blade|sword|gun|pistol|"
                            r"chain|shackle|kneel|explo|fire|flame|burn|kiss|ring\b|duke|heir|pregnan|divorc|arrest|"
                            r"dead|death|kill|poison|scar|crash|scream|slam|collapse|faint|tear[s ]|wedding|altar|"
                            r"betray|whip|crop|choke|hang|gallows|dungeon|cell\b|guards?\b|bursts?\b|shatter", re.I)
CLIFF_WORDS = re.compile(r"\bwill\b|never|until|unless|or else|pay for|you'?ll|who are you|what\?|why\?|how\?|"
                         r"\bwait\b|\bstop\b|truth|secret|know who|cut to black|freeze|bursts? in|doors? (?:fly|burst|swing)|"
                         r"cliffhanger|ominous|swells?|reveal", re.I)
CLOSEUP = re.compile(r"close[- ]?up|extreme close|\bECU\b|\bCU\b|tight (?:shot|insert|close)", re.I)


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", s or "")


def text_features(first_3s: str, hook_15s: str, ending: str) -> dict:
    """Pure-text features; identical definition for ad transcripts and script dialogue."""
    first_3s, hook_15s, ending = first_3s or "", hook_15s or "", ending or ""
    first_sentence = re.split(r"(?<=[.!?])\s+", hook_15s.strip())[0] if hook_15s.strip() else ""
    w15 = words(hook_15s)
    return {
        "f_first3_words": len(words(first_3s)),
        "f_hook15_words": len(w15),
        "f_first_line_question": int("?" in first_sentence),
        "f_hook15_questions": hook_15s.count("?"),
        "f_hook15_exclaims": hook_15s.count("!"),
        "f_hook15_you_rate": round(sum(w.lower() in {"you", "your", "you're", "yours"} for w in w15) / max(len(w15), 1), 3),
        "f_hook15_short_lines": sum(1 for x in re.split(r"[.!?]+", hook_15s) if 0 < len(words(x)) <= 4),
        "f_ending_question": int("?" in ending[-120:]),
    }


def timing_features(speech_start: float | None) -> dict:
    s = 15.0 if speech_start is None or np.isnan(speech_start) else min(float(speech_start), 15.0)
    return {"f_speech_start_s": round(s, 2), "f_silent_open_3s": int(s > 3.0)}


def visual_features(n_cuts_3s, face_closeup_3s) -> dict:
    return {"f_cuts_3s": int(min(n_cuts_3s or 0, 4)), "f_face_closeup_3s": int(bool(face_closeup_3s))}


# ---------- AI tag features (same enums for ChatGPT-tagged ads and LLM-tagged promos)
HOOK_TYPES = ["violence_or_slap", "public_humiliation", "betrayal_reveal", "identity_or_status_reveal",
              "threat_or_ultimatum", "romantic_or_sexual_shock", "insult_line", "rescue_entrance",
              "narrator_setup", "question_to_viewer", "calm_dialogue", "no_speech", "unclear"]
GENRES = ["billionaire_ceo_romance", "mafia_romance", "werewolf_paranormal", "fantasy_cultivation",
          "historical_royal", "revenge_betrayal", "family_drama", "marriage_divorce",
          "hidden_identity_status", "thriller_crime", "comedy", "unclear"]
TROPES = ["jilted_or_cheated", "contract_or_fake_marriage", "secret_baby_or_pregnancy", "hidden_rich_or_powerful",
          "revenge_comeback", "humiliation_underdog", "rescue_protector", "forbidden_or_age_gap",
          "rebirth_second_chance", "love_triangle", "evil_family_or_inlaws", "amnesia", "enemies_to_lovers", "none"]
ENDING_TYPES = ["unanswered_question", "pending_threat", "interrupted_reveal", "arrival_or_entrance",
                "decision_pending", "resolved", "call_to_action_download", "no_speech", "unclear"]
VISUAL_HOOKS = ["violence_slap_fight", "intimacy_kiss", "crying_or_shock_closeup", "wedding_or_formal_event",
                "wealth_luxury_display", "public_humiliation_crowd", "injury_blood_hospital", "pregnancy_or_baby",
                "supernatural_or_fantasy", "text_overlay_only", "calm_talking", "other", "unclear"]
SETTINGS = ["modern", "historical", "fantasy", "unclear"]


def tag_features(tags: dict) -> dict:
    """Map AI tags to model inputs. Missing tags -> all zeros / neutral."""
    out = {}
    for name, vocab, key in (("hook", HOOK_TYPES, "hook_type"), ("genre", GENRES, "genre"),
                             ("vis", VISUAL_HOOKS, "visual_hook"), ("setting", SETTINGS, "setting")):
        v = tags.get(key)
        for c in vocab:
            out[f"t_{name}_{c}"] = int(v == c)
    tr = tags.get("tropes") or []
    if isinstance(tr, str):
        tr = [t for t in tr.split("|") if t]
    for c in TROPES:
        out[f"t_trope_{c}"] = int(c in tr)
    out["t_emotion_intensity"] = float(tags.get("emotion_intensity") or 3)
    for b in ("open_mid_conflict", "hook_is_question", "narrator_voiceover"):
        v = tags.get(b)
        out[f"t_{b}"] = int(v is True or str(v).lower() == "true")
    return out
