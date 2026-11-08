"""Merge ad metadata + Whisper transcripts into the final dataset CSV.

Every value is copied from the TikTok Ad Library or computed from the real audio
(Whisper). Labels come only from TikTok's own reach band. Nothing is generated.

usage: python build_dataset.py <ads_metadata.csv> <transcripts_dir> <out_csv>
"""
import json, math, re, sys
from pathlib import Path
import pandas as pd

meta = pd.read_csv(sys.argv[1], dtype={"ad_id": str})
tdir, out = Path(sys.argv[2]), Path(sys.argv[3])
vdir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
VIS_COLS = ["n_cuts_3s", "motion_3s", "face_max_frac_3s", "face_closeup_3s",
            "audio_rms_db_3s", "audio_peak_db_3s"]


def visual(ad_id):
    f = vdir / f"{ad_id}.json" if vdir else None
    if f and f.exists():
        v = json.loads(f.read_text(encoding="utf-8"))
        if v.get("status") == "ok":
            return {k: v.get(k) for k in VIS_COLS}
    return {k: "" for k in VIS_COLS}


def text(segs):
    return " ".join(s["text"] for s in segs or []).strip()


def words_before(segs, t):
    return " ".join(w["w"].strip() for s in segs or [] for w in s.get("words", []) if w["s"] < t).strip()


rows = []
for _, m in meta.iterrows():
    f = tdir / f"{m.ad_id}.json"
    if not f.exists():
        continue
    t = json.loads(f.read_text(encoding="utf-8"))
    if t.get("status") != "ok":
        continue
    head, tail = t.get("head_segments", []), t.get("tail_segments", [])
    head_en = t.get("head_segments_en", head)
    tail_en = t.get("tail_segments_en", tail)
    lo, hi = m.reach_low, m.reach_high
    rows.append({
        "ad_id": m.ad_id,
        "ad_url": m.ad_url,
        "app": m.app_in_title if isinstance(m.app_in_title, str)
               else ("unspecified" if m.search_query == "short drama" else m.search_query),
        "advertiser_legal_name": m.advertiser_legal_name,
        "ad_title": m.ad_title,
        "first_shown": m.first_shown, "last_shown": m.last_shown, "days_shown": m.days_shown,
        "unique_users_seen": m.unique_users_seen,
        "reach_low": lo, "reach_high": hi,
        "reach_mid_log10": round(math.log10((lo + hi) / 2), 3) if pd.notna(lo) and pd.notna(hi) and hi > 0 else "",
        "reach_tier": {"100K+": "high", "10K-100K": "mid", "0-10K": "low"}[m.reach_band_filter],
        "winner": {"100K+": 1, "0-10K": 0}.get(m.reach_band_filter, ""),
        "duration_s": t.get("duration_s"),
        "is_short_promo_15_120s": int(15 <= (t.get("duration_s") or 0) <= 120),
        "language": t.get("language"),
        "first_3s_text": words_before(head, 3.0),
        "first_line": head[0]["text"] if head else "",
        "hook_15s_text": text(head),
        "last_line": tail[-1]["text"] if tail else "",
        "ending_20s_text": text(tail),
        "first_3s_text_en": words_before(head_en, 3.0),
        "hook_15s_text_en": text(head_en),
        "ending_20s_text_en": text(tail_en),
        "speech_starts_at_s": head[0]["start"] if head else "",
        "silent_open_3s": int(not head or head[0]["start"] > 3.0),
        **visual(m.ad_id),
        "transcriber": f"faster-whisper {t.get('whisper_model')}",
        "removed_note": m.removed_note if isinstance(m.removed_note, str) else "",
    })

df = pd.DataFrame(rows)

# Real tags written by the advertiser in the ad title (no AI): hashtags and quoted show names.
df["title_hashtags"] = df.ad_title.fillna("").map(lambda s: " ".join(re.findall(r"#\w+", s)))
df["show_name_in_title"] = df.ad_title.fillna("").map(
    lambda s: (re.findall(r"[\"“《]\s*([^\"”》]{4,80}?)\s*[\"”》]", s) or [""])[0])

# Second success metric: active days. Ads still running at collection time are right-censored.
coll = pd.to_datetime(meta.set_index("ad_id").collected_at.str[:10])
df["collected_on"] = df.ad_id.map(coll).dt.strftime("%Y-%m-%d")
df["still_running"] = (pd.to_datetime(df.last_shown) >= pd.to_datetime(df.collected_on) - pd.Timedelta(days=1)).astype(int)

# Keyword search ("short drama", "My Drama"...) also matched unrelated ads (car insurance,
# chocolate...). Rule: the title names a drama app / says drama or episode, OR the advertiser
# has other ads with such titles (a drama publisher). Plus a hand-checked exclusion list of
# advertisers whose titles use "drama" but are not short-drama apps.
DRAMA_RE = r"drama|episode|reelshort|dramabox|short ?max|netshort|dramawave|goodshort|flex ?tv|moboreels|kuku tv|finale"
NON_DRAMA_ADVERTISERS = ["COTY", "BRITISH BROADCASTING", "SAVE MY EXAMS", "SONY MUSIC", "POCKET GEMS"]
title_hit = meta.ad_title.fillna("").str.lower().str.contains(DRAMA_RE)
publishers = set(meta.loc[title_hit, "advertiser_legal_name"].dropna()) - {""}
df_title_hit = df.ad_title.fillna("").str.lower().str.contains(DRAMA_RE)
excluded = df.advertiser_legal_name.fillna("").str.upper().str.contains("|".join(NON_DRAMA_ADVERTISERS))
df["is_short_drama_ad"] = ((df_title_hit | df.advertiser_legal_name.isin(publishers)) & ~excluded).astype(int)

# Same video run as several ads -> one creative group (avoid double counting in training).
import re
key = df.hook_15s_text.fillna("").str.lower().map(lambda s: re.sub(r"[^a-z0-9 ]", "", s)[:120]) \
      + "|" + df.duration_s.round(0).astype(str)
df["creative_group"] = pd.factorize(key)[0]
df["n_ads_same_creative"] = df.groupby("creative_group").ad_id.transform("count")
tier_rank = df.reach_tier.map({"low": 0, "mid": 1, "high": 2})
df["creative_best_tier"] = tier_rank.groupby(df.creative_group).transform("max").map({0: "low", 1: "mid", 2: "high"})
df["is_first_of_creative"] = (~df.creative_group.duplicated()).astype(int)

df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(df)} ads -> {out}")
print(df.groupby("reach_tier").size().to_string())
print("short promos (15-120s):", int(df.is_short_promo_15_120s.sum()), "| english:", int((df.language == "en").sum()))
