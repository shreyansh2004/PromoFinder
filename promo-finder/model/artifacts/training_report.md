# Ad-success model: training report

Source `ads_dataset_tagged.csv`: 596 ads, 308 unique English short-drama videos (105 high, 138 low, 65 mid reach).

## 1. Confounding check: app vs label

| app | high | low |
|---|---|---|
| DramaBox | 7 | 2 |
| DramaWave | 3 | 0 |
| FlexTV | 0 | 2 |
| GoodShort | 0 | 9 |
| MoboReels | 0 | 2 |
| My Drama | 6 | 0 |
| NetShort | 1 | 10 |
| ReelShort | 5 | 66 |
| ShortMax | 7 | 11 |
| unspecified | 76 | 36 |

High and low ads come from different apps (a collection artefact). A model trained naively learns the app, not the hook. So training uses only apps with both labels, with weights that give every app equal high and low weight, and is scored by **within-app AUC**.

Training set: 221 ads from 5 apps (DramaBox, NetShort, ReelShort, ShortMax, unspecified); 96 high / 125 low.

## 2. Feature audit

- **Leakage:** none. Features come only from the hook transcript, measured video signals and AI tags of the hook; nothing from reach, dates, app, advertiser or title.
- **Missing values:** AI hook tags present for 87% of training ads (the rest were tags that contradicted the transcript and were blanked; their tag columns are 0).
- **Near-zero variance:** dropped 11 binary features with fewer than 6 positives or negatives. **Reference levels** (one per categorical tag, the most common level, plus 'unclear'/'none') removed to avoid the dummy-variable trap: `t_genre_family_drama`, `t_genre_unclear`, `t_hook_calm_dialogue`, `t_setting_modern`, `t_setting_unclear`, `t_trope_none`, `t_vis_calm_talking`.
- **Cramer's V between categorical tags:** hook_type/visual_hook 0.29, hook_type/genre 0.29, hook_type/setting 0.43, visual_hook/genre 0.32, visual_hook/setting 0.43, genre/setting 0.74. genre/setting is above 0.7 (fantasy and historical genres imply the setting), so **setting is dropped**.
- **Pairwise correlation (|r| > 0.7):** 6 pairs
  - `f_first3_words` / `f_hook15_words`: r = 0.71
  - `f_first_line_question` / `t_hook_is_question`: r = 0.84
  - `f_speech_start_s` / `f_silent_open_3s`: r = 0.84
  - `f_speech_start_s` / `t_hook_no_speech`: r = 0.78
  - `t_hook_narrator_setup` / `t_narrator_voiceover`: r = 0.92
  - `t_genre_billionaire_ceo_romance` / `t_trope_contract_or_fake_marriage`: r = 0.76
  Dropped the weaker member of each: `f_first3_words`, `f_first_line_question`, `f_speech_start_s`, `t_genre_billionaire_ceo_romance`, `t_narrator_voiceover`.
- **Multicollinearity (VIF > 5):** none above 5; max VIF now 4.68.
- **Class balance:** 43% high overall; the within-app weights make each app 50/50.

47 features pass the audit.

## 3. Which signals predict reach within an app?

Within-app ROC AUC, 5-fold x 10 repeats (0.50 = coin flip).

| feature group | features | within-app AUC |
|---|---|---|
| measured signals (no AI) | 8 | 0.514 ± 0.102 |
| hook type (AI tag) | 6 | 0.481 ± 0.081 |
| on-screen visual hook (AI tag) | 11 | 0.526 ± 0.133 |
| genre + tropes (AI tag) | 19 | 0.471 ± 0.105 |
| tone flags (AI tag) | 3 | 0.541 ± 0.079 |

Kept the groups whose AUC is clearly above chance: **tone flags (AI tag)**. Final model: 3 features, within-app AUC **0.541 ± 0.079** (C = 0.01). Note: choosing groups on the same CV slightly flatters this number.

**Weight in the final ranking** = min(0.5, 2 x (AUC - 0.5)) = **0.08** for the full model; the basic model (no AI) gets 0.03, i.e. it is not used.

## 4. External check
On 65 mid-reach ads never used in training, the model's score vs the ad's real active days: Spearman rho = -0.02 (p = 0.90), no clear link.

## 5. What the model learned (standardised coefficients, + = more likely high reach)

- `t_hook_is_question`: -0.17
- `t_open_mid_conflict`: +0.04
- `t_emotion_intensity`: +0.19
## 6. Every ranking signal tested the same way (model/validate_signals.py)

Within-app AUC on 221 real ads (no training involved for the stranger test and rules, so no leakage):

| signal | within-app AUC | 95% CI | search ads | named apps | earned weight |
|---|---|---|---|---|---|
| AI stranger test (Claude, same prompt as the tool) | 0.611 | 0.515-0.691 | 0.439 | 0.718 | 0.222 |
| ad-success model (this report) | 0.541 | | | | 0.082 |
| script signals (text-only version) | 0.514 | 0.391-0.628 | 0.438 | 0.594 | 0.028 |

Final ranking = earned weights normalised (floor 0.02): stranger test ~67%, model ~25%, rules ~8%.
