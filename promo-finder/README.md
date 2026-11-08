---
title: Promo Finder
emoji: 🎬
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Finds and ranks 20-60 s promo clips in microdrama scripts
---

# Promo Finder

Finds and ranks 20–60 second promo clips in microdrama screenplays. Each clip opens on a hook in the first 3 seconds and cuts on a cliffhanger, with exact episode timestamps an editor can cut from. Built for the Kuku AI Generalist assignment (Problem 1).

**Live app:** see the Hugging Face Space linked in the submission. Upload a show's episode `.txt` files or a `.zip` of them.

## Run it locally
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env          # add ANTHROPIC_API_KEY and/or GEMINI_API_KEY (optional)
.venv\Scripts\python -m uvicorn app:app --port 8000
```
Open http://localhost:8000. Command line:
```bash
.venv\Scripts\python cli.py --input "path\to\scripts" --out outputs\run1            # AI + rules
.venv\Scripts\python cli.py --input "path\to\scripts" --no-llm                      # rules only, no key
.venv\Scripts\python cli.py --input "path\to\scripts" --episodes 1-5 --model claude-opus-5-5
.venv\Scripts\python model\verify_outputs.py outputs\run1\promos.json "path\to\scripts"   # independent check
```

## Models and failover
- Providers are tried in `LLM_PROVIDERS` order (default `claude,gemini`).
- On a rate limit, timeout, retired model or bad key, the tool moves to the next model or provider. If none answers, it continues rule-based.
- The web app has a dropdown to pick the model; the chosen model goes first, with the rest of the chain behind it.
- Keys go in `.env` (see `.env.example`), or in the Space's secrets when hosted.

## How it works
| Step | Done by | What |
|---|---|---|
| Parse | code | Screenplay → 3,700+ timestamped beats (camera, action, dialogue, sound) |
| Series bible | AI, 1 call | Premise, characters and their aliases, continuity problems |
| Find moments | AI + rules | The AI names **beat IDs only**, never timestamps. Rule-based candidates are always added |
| Cut | code | Snap to shot boundaries, never cut mid-line, 20–60 s, hook within 3 s, de-duplicate, repair over-long picks |
| Stranger test | AI | Reads only the clip, like a scroller with zero context; rates hook, cliffhanger and standalone sense |
| Rank | code | Weighted by each signal's **measured skill on real ads**, then a diversity pass |
| Review | human | Editor approves or rejects in the UI, saved to `feedback/feedback.csv` |

## Evidence
- **Dataset:** `dataset/` holds 596 real short-drama ads from TikTok's public Ad Library, with Whisper transcripts, measured visual and audio hook signals, and ChatGPT tags. Each tag was checked against the transcript; none of the evidence quotes ChatGPT gave were invented.
- **Model report:** `model/artifacts/training_report.md`.
  - Audit: leakage, rare features, correlated pairs, Cramér's V, VIF.
  - A confound was found and fixed: the high- and low-reach ads came from different apps.
  - Every ranking signal was validated with within-app AUC on 221 real ads: stranger test 0.61, ML model 0.54, script rules 0.51.
  - The ranking weights are earned from those numbers.
- **Readable model:** `model/artifacts/model_full.json` is the trained model in plain text (scaler means, coefficients, intercept).
- **Output check:** `model/verify_outputs.py` re-checks every promo against the scripts. The 50-episode run passed 106/106 checks.

## Hosting (Hugging Face Space)
- The `Dockerfile` builds the app.
- `.github/workflows/sync-to-hf.yml` deploys every push to the Space, once these are set:
  - GitHub secret `HF_TOKEN`
  - GitHub variable `HF_SPACE`
- Space secrets:
  - `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY`
  - `APP_ACCESS_CODE`: only people with this code can use the AI mode; the rule-based mode stays open to everyone.
- Folder paths are disabled on the hosted app, so it accepts uploads only.

## Not in this repo
Kuku's input pack (the scripts) and its outputs are confidential and are git-ignored. Third-party ad videos and keyframes are also excluded; only metadata, transcripts and tags are included.
