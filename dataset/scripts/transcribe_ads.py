"""Transcribe the hook (first 15 s) and ending (last 20 s) of each ad video with local Whisper.

Downloads each video once to a temp file (deleted right after), transcribes it and, for
English ads, also computes the visual/audio hook features (visual_hook_features.analyse).
Real speech-to-text only: what the ad actually says. Non-English ads also get Whisper's
English translation. One JSON per ad in <out_dir> (resumable).

usage: python transcribe_ads.py <ads_metadata.csv> <out_dir> [model_size] [limit]
"""
import json, sys, time
from pathlib import Path
import av, numpy as np, pandas as pd
from faster_whisper import WhisperModel

meta_csv, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
model_size = sys.argv[3] if len(sys.argv) > 3 else "small"
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
shard, n_shards = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
out_dir.mkdir(parents=True, exist_ok=True)

HEAD_S, TAIL_S, SR = 15.0, 20.0, 16000
HDRS = "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0\r\nReferer: https://library.tiktok.com/\r\n"
ENGLISH_ONLY = True  # the Kuku show is English for a US audience; others are kept as metadata only
model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)
# language check only: the small "base" model is ~6x faster and reliable for this
det_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=8)


def read_audio(url, start, length):
    """Decode [start, start+length) seconds of audio as 16 kHz mono float32 (local file or URL)."""
    opts = {"headers": HDRS, "timeout": "30000000"} if str(url).startswith("http") else {}
    c = av.open(str(url), options=opts)
    try:
        if not c.streams.audio:
            return None, (c.duration or 0) / 1e6
        st = c.streams.audio[0]
        dur = (c.duration or 0) / 1e6
        if start > 0:
            c.seek(int(start * 1e6), any_frame=False, backward=True)
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks, got_start = [], None
        for frame in c.decode(st):
            t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            if t + frame.samples / frame.sample_rate < start:
                continue
            if got_start is None:
                got_start = t
            for fr in rs.resample(frame):
                chunks.append(fr.to_ndarray().reshape(-1))
            if t > start + length:
                break
        if not chunks:
            return None, dur
        a = np.concatenate(chunks).astype(np.float32)
        skip = int(max(0.0, start - (got_start or start)) * SR)
        return a[skip: skip + int(length * SR)], dur
    finally:
        c.close()


def run(audio, offset, task="transcribe", lang=None):
    segs, info = model.transcribe(audio, task=task, language=lang, word_timestamps=True,
                                  vad_filter=True, beam_size=1)
    out = []
    for s in segs:
        out.append({"start": round(s.start + offset, 2), "end": round(s.end + offset, 2),
                    "text": s.text.strip(),
                    "words": [{"w": w.word, "s": round(w.start + offset, 2), "e": round(w.end + offset, 2)}
                              for w in (s.words or [])]})
    return out, info.language, round(info.language_probability, 3)


df = pd.read_csv(meta_csv)
df = df[df.video_url.notna()]
order = {"100K+": 0, "10K-100K": 1, "0-10K": 2}
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0


def title_not_en(t):  # ordering only: ads with English titles are processed first
    try:
        return int(detect(str(t)) != "en")
    except Exception:
        return 1


df = df.assign(_o=df.reach_band_filter.map(order), _l=df.ad_title.map(title_not_en))
df = df.sort_values(["_l", "_o", "ad_id"]).head(limit)
df = df.iloc[shard::n_shards]
import os
if os.environ.get("REVERSE") == "1":  # second worker walks the list from the other end
    df = df.iloc[::-1]

# Downloading the whole file once is ~10x faster than PyAV's many small HTTP range reads.
# The video is deleted right after; visual hook features are computed from it in the same pass.
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_hook_features import analyse as visual_analyse

DATA = out_dir.parent
TMP, VIS, KF = DATA / "tmp_video", DATA / "visual", DATA / "keyframes"
for p in (TMP, VIS):
    p.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 400 * 1024 * 1024
MAX_TAIL_DUR = 180  # endings of 3+ min compilations are mid-story, not promo cliffhangers: skip


def fetch(url, ad_id):
    path = TMP / f"{ad_id}.mp4"
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0 Chrome/130.0", "Referer": "https://library.tiktok.com/"},
                      stream=True, timeout=60) as resp:
        resp.raise_for_status()
        if int(resp.headers.get("content-length") or 0) > MAX_BYTES:
            return url  # too big: fall back to streaming
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    return path


done = 0
for _, r in df.iterrows():
    fn = out_dir / f"{r.ad_id}.json"
    if fn.exists():
        continue
    t0 = time.time()
    rec = {"ad_id": str(r.ad_id), "whisper_model": model_size}
    src = None
    try:
        src = fetch(r.video_url, r.ad_id)
        head, dur = read_audio(src, 0.0, HEAD_S)
        rec["duration_s"] = round(dur, 2)
        if head is None or len(head) < SR:
            rec["status"] = "no_audio"
        elif ENGLISH_ONLY and (lang_det := det_model.detect_language(head))[0] != "en":
            rec["language"], rec["language_prob"] = lang_det[0], round(lang_det[1], 3)
            rec["status"] = "non_english_skipped"
        else:
            rec["head_segments"], lang, p = run(head, 0.0)
            rec["language"], rec["language_prob"] = lang, p
            if HEAD_S + 2 < dur <= MAX_TAIL_DUR:
                ts = max(HEAD_S, dur - TAIL_S)
                tail, _ = read_audio(src, ts, TAIL_S)
                rec["tail_segments"] = run(tail, ts, lang=lang)[0] if tail is not None and len(tail) > SR else []
            else:
                rec["tail_segments"] = []
            if lang != "en":
                rec["head_segments_en"] = run(head, 0.0, task="translate", lang=lang)[0]
                if dur > HEAD_S + 2 and rec["tail_segments"]:
                    rec["tail_segments_en"] = run(tail, ts, task="translate", lang=lang)[0]
            rec["status"] = "ok"
            vf = VIS / f"{r.ad_id}.json"
            if lang == "en" and not vf.exists():
                vrec = {"ad_id": str(r.ad_id)}
                try:
                    vrec.update(visual_analyse(src, str(r.ad_id), KF))
                    vrec["status"] = "ok"
                except Exception as e:
                    vrec["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
                vf.write_text(json.dumps(vrec), encoding="utf-8")
    except Exception as e:
        rec["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        if isinstance(src, Path) and src.exists():
            src.unlink()
    rec["secs"] = round(time.time() - t0, 1)
    fn.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    done += 1
    print(f"{done} {r.ad_id} {r.reach_band_filter} {rec.get('status')} lang={rec.get('language')} "
          f"dur={rec.get('duration_s')} {rec['secs']}s", flush=True)
