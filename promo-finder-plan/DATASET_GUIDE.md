# Building the "ads that worked" dataset (≈3 hours, $0)

**Goal:** collect 150–300 real short-drama ads, each with
(a) its **hook**, meaning the first ~3 s (spoken line + what's on screen), and its **ending/cliffhanger**,
(b) at least one **performance signal**: reach, impressions, days running, or a CTR rank.
We then train a small model (features → P(winner)) and use it to rank the promos our tool finds.

In the doc, say: *"Prototype is trained on public ad-transparency data. In production, retrain on Kuku's own Meta/TikTok Ads Manager data (3s hold, CTR, CPI)."*

---

## 1. Sources (verified, in priority order)

| # | Source | What you get per ad | How | Why |
|---|---|---|---|---|
| **A** | **TikTok Ad Library (EU DSA library)**: library.tiktok.com | video URL, ad title, `firstShown`, `lastShown`, `daysShown`, **`uniqueUsersSeen` range** (e.g. "100K-1M"), country/age reach | No login. Either search by hand, or use the Apify actor "TikTok Ad Library Scraper" (kamerozkan; examples at github.com/kamerozkan/tiktok-ad-library-scraper) on the free Apify credit. Exports CSV/JSON | **Main dataset.** Official public repository, has video + reach + duration |
| **B** | **Meta Ad Library**, country = GB (the UK is covered by the transparency rules) | creative snapshot (video), start/end date, active flag, **collation count** (copies of the same creative = scaling signal), EU/UK reach with "scrape ad details" | Apify actor `curious_coder/facebook-ads-library-scraper` (input: an Ad Library search URL with `country=GB&q=ReelShort`) | Second platform, more volume |
| **C** | **TikTok Creative Center → Top Ads** (free TikTok Ads login) | **CTR percentile ("Top X%")**, likes; sort by **2s / 6s view rate** | **Manual only** (its terms ban scraping). Search app names or "drama", sort by CTR, then by 6s view rate. Copy 30–50 rows into a sheet | Only free *click/hold-rate* signal → hook vs cliffhanger labels |
| **D** | **admanage.ai** brand pages (e.g. `/brand/shorts_-_reel_short_drama`, `/ad-library/<slug>`) | top ads with reach, days running, MP4 download, headline | Manual | Small "known winners" check set |
| **E** | AdMapix / AppGrowing blog teardowns (DramaBox, PineDrama, StoryMatrix) | impression estimates + one-line hook for ~10–20 top ads | Manual | Sanity check + hook-type definitions |

**Skip these:** PiPiAds (its trial shows no ad details), WinnerAdSpy (winners are paywalled), MagicBrief (shut down), AdSpy and Sensor Tower (paid).
**Do not use as labels:** anything that is an AI *predicted* hook score (e.g. the Apify "Short Drama Hook Analyzer"). That would be circular.

**Search terms** (advertiser or keyword): ReelShort, DramaBox, ShortMax, GoodShort, FlexTV, NetShort, My Drama, DramaWave, MoboReels, Kuku TV.
**Countries:** **GB first** (English-language ads, closest to a US audience), then IE. Use DE/FR only if volume is low; Gemini can translate.

## 2. Step by step

**Step 1: Collect metadata (40 min)**
1. Create an Apify account (free monthly credit). Run the TikTok Ad Library actor:
   `{"searchTerms": ["ReelShort","DramaBox","ShortMax","GoodShort","FlexTV","NetShort","My Drama","DramaWave"], "country": "GB", "daysBack": 365, "maxItemsPerSearch": 60, "maxItems": 500}`
   Set "maximum cost per run" to about $2. Export CSV, then save it as `data/raw/tiktok_library.csv`.
2. (Optional) Run the Meta Ad Library actor for the same apps with `country=GB` and `scrapeAdDetails=true`. Save it as `data/raw/meta_library.csv`.
3. Manual: in TikTok Creative Center, log 30–50 drama ads into `data/raw/tiktok_cc_manual.csv` (columns: url, app, ctr_pct, likes, view_rate_rank_note).
4. **Fallback if Apify fails:** search library.tiktok.com by hand (country GB, sort by audience size). Copy about 60 ads into the sheet. That takes about 1 h.

**Step 2: Get the content (40 min)**
1. `download.py`: fetch each `videoUrl` into `data/videos/<ad_id>.mp4`. Skip failures, log them, and keep videos **local only** (never commit them).
2. ffmpeg: `ffmpeg -i in.mp4 -t 6 head.mp4` and `ffmpeg -sseof -8 -i in.mp4 tail.mp4`, plus the audio as a `.wav`.
3. Transcribe with **faster-whisper** (local, free, word timestamps) or Groq's free Whisper API. Store `first_3s_text`, `first_line`, `last_line` and the full transcript.

**Step 3: Tag features with Gemini (40 min, cached)**
- Gemini API (AI Studio key) → Files API upload of `head.mp4` + `tail.mp4` (or the full MP4).
- Use **the same `feature_tag` prompt that tags our own script promos** (section 4). Then a winner pattern learned from ads applies directly to our candidates.
- Free tier is rate-limited. Run it sequentially with a sleep and a disk cache. About 200 videos fits in a day.

**Step 4: Labels (20 min)**

| Label | Formula | Notes |
|---|---|---|
| `reach_log` | log10(midpoint of `uniqueUsersSeen` range) | main signal |
| `days` | `daysShown` (or lastShown − firstShown) | survival = kept running because it worked |
| `winner` | **within the same advertiser**: top third by `reach_log` **or** `days ≥ 30` → 1; bottom third **and** `days ≤ 7` → 0; middle → dropped | comparing within an advertiser controls for budget size |
| censoring | drop ads first shown < 30 days ago that are still active | they haven't had time to win or lose |
| `hook_label` / `cliff_label` (source C only) | 6s-view-rate rank → hook; CTR percentile → cliffhanger | small separate check |

**Step 5: Train and test (30 min)**
- One-hot features → **logistic regression** (n is small, so keep it simple and interpretable).
- Validate with **GroupKFold by advertiser** and report AUC honestly. Even ~0.6 is a useful finding if you state it with its limits.
- Output a **win-rate table per hook type** with Wilson confidence intervals. This is the readable artifact for marketers and for the doc.
- **Key test of the LLM judge:** build ~50 winner-vs-loser pairs from the same advertiser. Show only the transcripts to (1) the LLM pairwise judge, (2) the heuristic score, (3) the learned prior. Report the **% of pairs where each picks the real winner** (50% = coin flip). This answers "is the LLM's opinion actually good?" with real outcomes.

**Step 6: Plug into the prototype (20 min)**
`rank.py`: `score = 0.5·prior + 0.3·pairwise + 0.2·heuristic − penalties`. Use the prior only if `prior.pkl` exists. Show the weights in the report.

## 3. Dataset schema: `data/ads_dataset.csv`
```
ad_id, source(tiktok_lib|meta_lib|tiktok_cc|admanage|blog), app, country, ad_url,
first_shown, last_shown, days_shown, still_active, reach_low, reach_high, reach_log,
collation_count, ctr_pct, likes, duration_s,
first_3s_text, first_line, last_line, transcript,
hook_type, hook_modality, hook_is_question, hook_words, hook_has_violence,
hook_has_reveal, open_mid_conflict, narrator_or_text_overlay, cliff_type,
cliff_unresolved, trope, emotion_intensity, n_characters, visual_dependency,
winner, split
```

## 4. `feature_tag` prompt (used for scraped ads AND our promos)
```
You label short-drama promo clips. Use ONLY what is in the input. Return JSON:
{ "hook_type": one of [violence_slap, public_humiliation, betrayal_reveal,
   identity_reveal, threat_or_ultimatum, proposal_or_romance_shock, insult_line,
   rescue_entrance, narrator_setup, other],
  "hook_modality": [spoken_line, action, text_overlay, narrator],
  "open_mid_conflict": bool, "hook_is_question": bool, "hook_words": int,
  "cliff_type": [unanswered_question, pending_threat, interrupted_reveal,
   arrival_or_entrance, decision_pending, resolved_no_cliff],
  "cliff_unresolved": 1-5, "trope": [jilted_bride, hidden_identity_rich,
   revenge, contract_marriage, forbidden_romance, rescue_protector, family_betrayal, other],
  "emotion_intensity": 1-5, "n_characters": int, "visual_dependency": 1-5 }
Input: FIRST_3S: ... | FULL_TRANSCRIPT: ... | VISUAL_NOTES: ...
```
For our script promos: FIRST_3S comes from the beats in the first 3 s, and VISUAL_NOTES = the ACTION/CAMERA lines. For scraped ads: from Whisper + Gemini's visual description. **Same fields, so the model transfers.**

## 5. Honest limits (put in the doc)
- Reach and days running measure **exposure and survival, not conversion**. Spend, targeting and visuals confound them.
- Comparing within an advertiser helps with budget, but n ≈ 100–200 gives a **directional prior, not proof**.
- EU/UK ads ≠ the US audience; competitor shows ≠ Kuku's genre.
- Our promos are text (scripts) while the ads are video, so features are restricted to things that both can express.
- **Legal/ethics:** the data comes from public transparency libraries, and only metadata, transcripts and features go in the repo, never the videos. Scraping through Apify may conflict with platform terms, so disclose it. In production, use Kuku's own data plus the official APIs (TikTok Commercial Content API, Meta Ad Library API).

## 6. Production version (one paragraph in the doc)
Kuku's Ads Manager data provides exact 3s hold (→ hook model), CTR/CPI (→ cliffhanger model) and thousands of ads. For each new show, the tool proposes promos, the learned model ranks them, and the editor approves. The top-k are posted as an A/B test (or a bandit across hook types), results are fed back, and the model is retrained weekly.
