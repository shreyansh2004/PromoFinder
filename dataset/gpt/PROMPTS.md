# AI tags with ChatGPT Plus (no API key)

ChatGPT adds **descriptive tags** (genre, trope, hook type, visual hook) to real ads. These tags are AI-generated and go in columns prefixed `ai_`. The success labels (reach, active days) are never touched.

## Steps
1. Open a **new chat** (GPT-4o or newer). Paste **Prompt A** and send it.
2. Upload `text_batches/text_batch_01.md` and send `batch 01`. If the answer is cut off, send `continue`.
3. Copy the JSON lines into `answers/text_batch_01.jsonl`. Repeat for each batch, in the same chat.
4. Open another **new chat**, paste **Prompt B**, then upload `image_sheets/sheet_01.jpg` … (up to 10 images per message). Copy the answers into `answers/image_sheet_01.jsonl`, and so on.
5. Run the merge (it checks every line and rejects anything invalid):
   ```
   .venv\Scripts\python -I scripts\merge_gpt_tags.py data\ads_dataset.csv gpt\answers data\ads_dataset_tagged.csv "ChatGPT GPT-4o"
   ```
   It prints rejected lines. Re-ask ChatGPT only for those ad_ids.
6. Spot-check 20 random rows against the transcript and keyframes, and write down how many were wrong. That number is your tag-accuracy figure for the doc.

---

## Prompt A: text tags (transcript + title)
```
You are labelling real short-drama (microdrama) app ads for a research dataset.
I will upload files. Each ad has: ad_id, app, the advertiser's ad title, hashtags,
and the speech-to-text of its first 3 s, first 15 s and last 20 s.

Rules:
- Read every ad yourself. Do NOT write code, do NOT use keyword matching, do NOT skip ads.
- Use ONLY the given text. If something cannot be told from the text, use "unclear".
- Speech-to-text can contain errors; read for meaning.
- Output ONLY JSON Lines inside one code block: one line per ad, in the same order, nothing else.
- If you run out of space, stop at a complete line; I will say "continue".

Each line must be exactly:
{"ad_id":"<copy exactly>",
 "genre":"<one of: billionaire_ceo_romance | mafia_romance | werewolf_paranormal | fantasy_cultivation | historical_royal | revenge_betrayal | family_drama | marriage_divorce | hidden_identity_status | thriller_crime | comedy | unclear>",
 "tropes":["<1-3 of: jilted_or_cheated | contract_or_fake_marriage | secret_baby_or_pregnancy | hidden_rich_or_powerful | revenge_comeback | humiliation_underdog | rescue_protector | forbidden_or_age_gap | rebirth_second_chance | love_triangle | evil_family_or_inlaws | amnesia | enemies_to_lovers | none>"],
 "hook_type":"<one of: violence_or_slap | public_humiliation | betrayal_reveal | identity_or_status_reveal | threat_or_ultimatum | romantic_or_sexual_shock | insult_line | rescue_entrance | narrator_setup | question_to_viewer | calm_dialogue | no_speech | unclear>",
 "hook_is_question":true|false,
 "open_mid_conflict":true|false,
 "narrator_voiceover":true|false,
 "ending_type":"<one of: unanswered_question | pending_threat | interrupted_reveal | arrival_or_entrance | decision_pending | resolved | call_to_action_download | no_speech | unclear>",
 "setting":"<modern | historical | fantasy | unclear>",
 "emotion_intensity":<1-5>,
 "evidence":"<max 12 words copied from the transcript that justify hook_type>",
 "confidence":"<low | medium | high>"}

Reply "ready" and wait for the first file.
```

## Prompt B: visual hook tags (keyframe sheets)
```
You are labelling real short-drama app ads for a research dataset.
Each image I upload is a sheet of up to 8 ads. Each ad is a white bar with its ad_id,
followed by 3 frames taken at 0.5 s, 1.5 s and 2.5 s of the ad (left to right).

Rules:
- Describe ONLY what is visible. Do not guess the story. If unsure, use "unclear".
- Copy each ad_id exactly as printed. One line per ad, in the order on the sheet (left-right, top-bottom).
- Output ONLY JSON Lines inside one code block, nothing else.

Each line must be exactly:
{"ad_id":"<copy exactly>",
 "visual_hook":"<one of: violence_slap_fight | intimacy_kiss | crying_or_shock_closeup | wedding_or_formal_event | wealth_luxury_display | public_humiliation_crowd | injury_blood_hospital | pregnancy_or_baby | supernatural_or_fantasy | text_overlay_only | calm_talking | other | unclear>",
 "on_screen_text":true|false,
 "on_screen_text_content":"<visible caption/overlay text, max 15 words, or empty>",
 "shot_type":"<closeup | medium | wide | mixed>",
 "people_count":"<0 | 1 | 2 | 3+>",
 "visual_setting":"<home | office | street_outdoor | party_event | hospital | historical | fantasy | vehicle | other | unclear>",
 "confidence":"<low | medium | high>"}

Reply "ready" and wait for the first image.
```
