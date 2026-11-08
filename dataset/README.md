# Short-drama ads dataset (real data, no synthetic rows)

**What it is:** ads from short-drama apps (ReelShort, DramaBox, ShortMax, NetShort, My Drama, DramaWave, GoodShort, FlexTV, MoboReels, Kuku TV). Each ad comes with its real reach and a transcript of its hook and ending.

**Source:** the [TikTok Ad Library](https://library.tiktok.com), TikTok's public EU/UK ad-transparency repository under the Digital Services Act. It needs no login. Collected 2026-11-08, covering ads shown in the last 365 days.

## How each column is produced
| Column(s) | Origin |
|---|---|
| `unique_users_seen`, `reach_low/high`, `first_shown`, `last_shown`, `days_shown`, `ad_title`, `advertiser_legal_name` | Copied from the Ad Library API response (raw JSON kept in `data/raw/tiktok/`) |
| `reach_tier` (high / mid / low) | TikTok's own reach filter: 100K+, 10K–100K, under 10K unique users |
| `winner` | 1 = high tier, 0 = low tier, blank = mid. This is a rule applied to TikTok's numbers, not a judgment |
| `first_3s_text`, `hook_15s_text`, `ending_20s_text`, `language`, `duration_s` | Local speech-to-text (faster-whisper "small") run on the ad's real audio, streamed from the library |
| `silent_open_3s`, `speech_starts_at_s` | Whisper word timestamps; a silent open means the hook is visual or musical |
| `n_cuts_3s`, `motion_3s`, `face_max_frac_3s`, `face_closeup_3s`, `audio_rms_db_3s`, `audio_peak_db_3s` | Plain code (OpenCV / numpy) on the real first 3 s of video and audio; no AI model. Keyframes at 0.5, 1.5 and 2.5 s are saved in `data/keyframes/` |
| `creative_group`, `n_ads_same_creative` | Same video run as several ads (matched on transcript + duration). Use `is_first_of_creative` to avoid double counting |
| `is_short_drama_ad` | 1 if the title names a drama app or says drama/episode, or if the advertiser publishes such ads; 0 otherwise (e.g. the "short drama" search also matched car-insurance and chocolate ads). A hand-checked exclusion list removes Coty, BBC, Save My Exams, Sony Music and Pocket Gems |
| `ad_url` | Link to the ad's public page, so any row can be checked |

**Not available:** ad spend. The library's `spent` field is empty for every commercial ad, and the `impression` field is 0. Reach and days shown are the only public performance signals.

## Limits
- Reach measures exposure, not conversion. It also depends on budget, targeting and the visuals.
- The UK filter does not restrict language. Non-English ads are kept as metadata (`non_english_skipped`); only English ads are transcribed.
- Some "ads" are 10–30 minute compilations. Use `is_short_promo_15_120s` to keep only promo-length clips.
- Whisper transcripts can contain small errors; they are not hand-corrected.
- In the second collection round, the ending (`ending_20s_text`) is transcribed only for ads up to 3 minutes long. Endings of longer compilations are mid-story, not cliffhangers.
- In production, replace this with Kuku's own Ads Manager data (3s hold rate, CTR, CPI).

## Rebuild
```
.venv\Scripts\python -I scripts\collect_tiktok.py data\raw\tiktok GB 3
.venv\Scripts\python -I scripts\collect_tiktok.py data\raw\tiktok_expand_high GB 15 "ReelShort,short drama,DramaWave,NetShort,ShortMax,DramaBox" "100K+"
.venv\Scripts\python -I scripts\collect_tiktok.py data\raw\tiktok_expand_low GB 4 "ReelShort,short drama" "0-10K"
.venv\Scripts\python -I scripts\collect_tiktok.py data\raw\tiktok_win_high GB 4 "ReelShort,short drama" "100K+" 12
.venv\Scripts\python -I scripts\collect_tiktok.py data\raw\tiktok_win_low GB 1 "short drama,ReelShort" "0-10K" 12
.venv\Scripts\python -I scripts\parse_tiktok.py data\raw data\ads_metadata.csv
.venv\Scripts\python -I scripts\transcribe_ads.py data\ads_metadata.csv data\transcripts small
.venv\Scripts\python -I scripts\visual_hook_features.py data\ads_metadata.csv data\transcripts data\visual data\keyframes
.venv\Scripts\python -I scripts\build_dataset.py data\ads_metadata.csv data\transcripts data\ads_dataset.csv data\visual
```
