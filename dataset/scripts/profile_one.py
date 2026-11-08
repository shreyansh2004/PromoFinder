"""Time each step (download, decode, language detect, transcribe) for one pending ad."""
import os, sys, time
import av, numpy as np, pandas as pd, requests
from faster_whisper import WhisperModel

meta, tdir, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(meta, dtype={"ad_id": str})
d = d[d.video_url.notna()]
done = {f[:-5] for f in os.listdir(tdir)}
r = d[~d.ad_id.isin(done)].iloc[5]

t = time.time()
x = requests.get(r.video_url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://library.tiktok.com/"}, timeout=60)
path = os.path.join(tmp, "profile.mp4")
open(path, "wb").write(x.content)
print("download", round(time.time() - t, 1), "s", len(x.content) // 1024, "KB")

t = time.time()
m = WhisperModel(os.environ.get("WM","small"), device="cpu", compute_type="int8", cpu_threads=8)
print("model load", round(time.time() - t, 1))

t = time.time()
c = av.open(path)
rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
chunks = []
for fr in c.decode(c.streams.audio[0]):
    if fr.pts * fr.time_base > 15:
        break
    chunks += [f.to_ndarray().reshape(-1) for f in rs.resample(fr)]
c.close()
a = np.concatenate(chunks)
print("decode", round(time.time() - t, 1), "audio s", len(a) / 16000)

t = time.time()
print("lang", m.detect_language(a)[:2], round(time.time() - t, 1))
t = time.time()
segs, info = m.transcribe(a, word_timestamps=True, vad_filter=True, beam_size=1)
segs = list(segs)
print("transcribe (words)", round(time.time() - t, 1))
t = time.time()
segs, info = m.transcribe(a, word_timestamps=False, vad_filter=True, beam_size=1)
segs = list(segs)
print("transcribe (no words)", round(time.time() - t, 1))
os.remove(path)

