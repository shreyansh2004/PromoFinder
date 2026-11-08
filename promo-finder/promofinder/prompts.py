"""All LLM prompts in one place. The LLM only names beat IDs and tags; code does the rest."""

BIBLE_SYSTEM = "You analyse microdrama series for a marketing team. Use only names and facts present in the input. Reply with JSON only."
BIBLE_USER = """Below are the episode loglines, continuity notes and the speaker labels used in the scripts
(with the episodes they appear in). Build a short series bible.

Return JSON:
{{"premise": "<one sentence a stranger would understand>",
  "characters": [{{"name": "<canonical name>", "aliases": ["<speaker labels or descriptions that refer to them>"],
                   "role": "<protagonist|love interest|villain|family|ally|other>", "one_line": "<who they are>"}}],
  "turning_points": [{{"episode": <int>, "what": "<event>"}}],
  "continuity_issues": ["<same thing named differently across episodes, e.g. place or prop spellings>"]}}

LOGLINES:
{loglines}

SPEAKER LABELS:
{speakers}
"""

MINE_SYSTEM = ("You are a performance-marketing editor at a microdrama app. You cut 20-60 second promo clips "
               "for TikTok, Reels and Shorts that make a scrolling stranger stop and need the next episode. "
               "You pick moments from the script only. Reply with JSON only.")
MINE_USER = """SERIES PREMISE: {premise}
CHARACTERS: {characters}
STORY SO FAR: {story_so_far}

Below are timestamped beats of episodes {ep_range}. Each line: [BEAT_ID start-end] KIND: text.

Propose {per_ep} promo clips PER EPISODE (fewer if an episode has no strong moment). Each promo:
- opens on a HOOK in its first 3 seconds: a slap, a reveal, a humiliation, a shocking or cruel line, mid-conflict.
  Do NOT open on calm setup.
- ends on a CLIFFHANGER: cut on an unresolved line or action (a threat, a question, someone arriving, a reveal
  about to happen). Never include the resolution.
- makes sense to someone who has never seen the show.
- lasts 20-60 seconds (check the timestamps).
- is one continuous stretch inside ONE episode (start_beat and end_beat from the same episode).
Consider the whole episode, not just its start or end.

Return JSON: {{"promos": [{{"episode": <int>, "start_beat": "<BEAT_ID>", "end_beat": "<BEAT_ID>",
  "hook_beat": "<BEAT_ID>", "cliffhanger_beat": "<BEAT_ID>", "title": "<5-8 word ad title>",
  "hook": "<what happens in the first 3 s>", "cliffhanger": "<the unresolved moment at the cut>",
  "why": "<one line on why a stranger would stop and want more>"}}]}}
Use ONLY beat IDs that appear below.

BEATS:
{beats}
"""

MONTAGE_SYSTEM = MINE_SYSTEM
MONTAGE_USER = """SERIES PREMISE: {premise}
Below are strong single moments already found (with their beat ranges). Combine 2-3 of them, possibly from
different episodes, into {n} MONTAGE promos that sell the series premise to a stranger in 20-60 seconds total:
open on the most arresting moment, add the minimum context, end on the biggest unresolved question.

Return JSON: {{"promos": [{{"segments": [{{"start_beat": "<id>", "end_beat": "<id>"}}],
  "title": "<5-8 words>", "hook": "<first 3 s>", "cliffhanger": "<final unresolved moment>",
  "why": "<one line>"}}]}}
TIME BUDGET: each segment 8-20 seconds, the whole montage 30-55 seconds. Use the beat offsets to check:
a segment from a beat at 12s to a beat at 25s lasts about 13-16 seconds. Montages over 60 seconds are rejected.
Use only start/end beat pairs taken from the moments below (shorten a moment by choosing a later start_beat
or earlier end_beat from the same moment's list of beats).

MOMENTS:
{moments}
"""

TAG_SYSTEM = "You label short-drama promo clips for a research dataset. Use only the given text. Reply with JSON only."
TAG_USER = """Label this promo clip. Inputs are what is said and shown in it.

TITLE: {title}
FIRST 3 SECONDS (spoken): {first3}
FIRST 15 SECONDS (spoken) -> judge hook_type from THIS: {hook15}
LAST 20 SECONDS (spoken): {ending}
WHAT IS ON SCREEN IN THE FIRST 3 SECONDS (camera/action notes): {visual3}

Return JSON exactly:
{{"genre": "<billionaire_ceo_romance|mafia_romance|werewolf_paranormal|fantasy_cultivation|historical_royal|revenge_betrayal|family_drama|marriage_divorce|hidden_identity_status|thriller_crime|comedy|unclear>",
 "tropes": ["<1-3 of: jilted_or_cheated|contract_or_fake_marriage|secret_baby_or_pregnancy|hidden_rich_or_powerful|revenge_comeback|humiliation_underdog|rescue_protector|forbidden_or_age_gap|rebirth_second_chance|love_triangle|evil_family_or_inlaws|amnesia|enemies_to_lovers|none>"],
 "hook_type": "<violence_or_slap|public_humiliation|betrayal_reveal|identity_or_status_reveal|threat_or_ultimatum|romantic_or_sexual_shock|insult_line|rescue_entrance|narrator_setup|question_to_viewer|calm_dialogue|no_speech|unclear>",
 "hook_is_question": true|false, "open_mid_conflict": true|false, "narrator_voiceover": true|false,
 "ending_type": "<unanswered_question|pending_threat|interrupted_reveal|arrival_or_entrance|decision_pending|resolved|call_to_action_download|no_speech|unclear>",
 "setting": "<modern|historical|fantasy|unclear>", "emotion_intensity": <1-5>,
 "visual_hook": "<violence_slap_fight|intimacy_kiss|crying_or_shock_closeup|wedding_or_formal_event|wealth_luxury_display|public_humiliation_crowd|injury_blood_hospital|pregnancy_or_baby|supernatural_or_fantasy|text_overlay_only|calm_talking|other|unclear>"}}
"""

REVIEW_SYSTEM = ("You are a stranger scrolling TikTok who has never heard of this show, and afterwards a careful "
                 "labeller. Be honest and critical. Use only what is in the clip. Reply with JSON only.")
REVIEW_USER = """Here is everything you would see and hear in a {dur:.0f}-second clip (no other context).
Lines marked (...) are on-screen action or sound.

{transcript}

Return JSON exactly:
{{"what_is_happening": "<one sentence>", "open_question": "<what you now need to know, or 'none'>",
 "hook": <1-5: would the first 3 seconds stop your scroll?>,
 "cliffhanger": <1-5: does it cut on an UNRESOLVED moment so you need the next part? A calm or happy ending = 1-2>,
 "standalone": <1-5: does it make sense with zero context?>,
 "confusing": "<what confused you, or 'nothing'>",
 "genre": "<billionaire_ceo_romance|mafia_romance|werewolf_paranormal|fantasy_cultivation|historical_royal|revenge_betrayal|family_drama|marriage_divorce|hidden_identity_status|thriller_crime|comedy|unclear>",
 "tropes": ["<1-3 of: jilted_or_cheated|contract_or_fake_marriage|secret_baby_or_pregnancy|hidden_rich_or_powerful|revenge_comeback|humiliation_underdog|rescue_protector|forbidden_or_age_gap|rebirth_second_chance|love_triangle|evil_family_or_inlaws|amnesia|enemies_to_lovers|none>"],
 "hook_type": "<judge from the first 15 s: violence_or_slap|public_humiliation|betrayal_reveal|identity_or_status_reveal|threat_or_ultimatum|romantic_or_sexual_shock|insult_line|rescue_entrance|narrator_setup|question_to_viewer|calm_dialogue|no_speech|unclear>",
 "hook_is_question": true|false, "open_mid_conflict": true|false, "narrator_voiceover": false,
 "ending_type": "<unanswered_question|pending_threat|interrupted_reveal|arrival_or_entrance|decision_pending|resolved|unclear>",
 "setting": "<modern|historical|fantasy|unclear>", "emotion_intensity": <1-5>,
 "visual_hook": "<what is on screen in the first 3 s: violence_slap_fight|intimacy_kiss|crying_or_shock_closeup|wedding_or_formal_event|wealth_luxury_display|public_humiliation_crowd|injury_blood_hospital|pregnancy_or_baby|supernatural_or_fantasy|text_overlay_only|calm_talking|other|unclear>"}}
"""

JUDGE_SYSTEM = ("You are a stranger scrolling TikTok who has never heard of this show. Be honest and critical. "
                "Reply with JSON only.")
JUDGE_USER = """Here is everything you would see and hear in a {dur:.0f}-second clip (no other context).

{transcript}

Return JSON:
{{"what_is_happening": "<one sentence>", "who_is_who": "<one sentence, or 'unclear'>",
 "open_question": "<what you now need to know, or 'none'>",
 "hook": <1-5: would the first 3 seconds stop your scroll?>,
 "cliffhanger": <1-5: how badly do you need the next part?>,
 "standalone": <1-5: does it make sense with zero context?>,
 "confusing": "<what confused you, or 'nothing'>"}}
"""
