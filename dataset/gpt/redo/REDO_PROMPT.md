# Redo: 34 ads still tagged "no_speech" even though they have dialogue

All other checks now pass: no invalid values, and every evidence quote is real. The 34 ads in `redo_text.md` are tagged `hook_type: "no_speech"`, but their first 15 seconds contain dialogue. ChatGPT looked at the empty first 3 seconds instead.

**Image sheets 1–28 are fine** (spot-checked 8/8 correct). Please finish sheets 29–39 the same way.

## What to do
1. In the text-tagging chat, paste the message below and upload `redo_text.md`.
2. Save the answer as `gpt/answers/redo_text.jsonl`. It overrides the old answers for these ads.

## Message to paste
```
In this file every ad HAS dialogue in its first 15 seconds, so hook_type must NOT be "no_speech".
Relabel each ad from scratch with the same JSON format and allowed values as before.
Judge hook_type from the "first 15 seconds" line (speech that starts after 3 seconds is still a spoken hook).
Output only JSON Lines in one code block, one line per ad, same order as the file.
```
