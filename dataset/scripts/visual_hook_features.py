"""Measure the first 3 seconds of each ad from the real video and audio (no AI models).

Features: audio loudness (rms / peak dB), number of shot cuts, motion energy,
largest face as a share of the frame (OpenCV Haar detector), plus 3 saved keyframes.
Only ads already transcribed in English (status ok) are processed. Resumable.

usage: python visual_hook_features.py <ads_metadata.csv> <transcripts_dir> <out_dir> <keyframes_dir>
"""
import json, sys, time
from pathlib import Path
import av, cv2, numpy as np, pandas as pd

HDRS = "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0\r\nReferer: https://library.tiktok.com/\r\n"
WIN = 3.0
KEY_T = [0.5, 1.5, 2.5]
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def analyse(url, ad_id, kf_dir):
    """url may be a local file path (fast) or an http URL."""
    kf_dir = Path(kf_dir)
    kf_dir.mkdir(parents=True, exist_ok=True)
    opts = {"headers": HDRS, "timeout": "30000000"} if str(url).startswith("http") else {}
    c = av.open(str(url), options=opts)
    try:
        feats = {}
        # ---- video: every frame in the first 3 s
        vs = c.streams.video[0]
        prev_hist, prev_small, cuts, motion, face_frac, n = None, None, 0, [], 0.0, 0
        saved = set()
        for fr in c.decode(vs):
            t = float(fr.pts * fr.time_base) if fr.pts is not None else 0.0
            if t > WIN:
                break
            img = fr.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (90, 160))
            hist = cv2.calcHist([small], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.35:
                    cuts += 1
                motion.append(float(np.mean(cv2.absdiff(small, prev_small))))
            prev_hist, prev_small = hist, small
            if n % 6 == 0:  # face check ~5x per second
                h, w = gray.shape
                faces = face_det.detectMultiScale(cv2.resize(gray, (w // 2, h // 2)), 1.1, 5, minSize=(30, 30))
                for (_, _, fw, fh) in faces:
                    face_frac = max(face_frac, (fw * fh * 4) / (w * h))
            for kt in KEY_T:
                if kt not in saved and t >= kt:
                    cv2.imwrite(str(kf_dir / f"{ad_id}_{kt:.1f}s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    saved.add(kt)
            n += 1
        feats.update({"n_frames_3s": n, "n_cuts_3s": cuts,
                      "motion_3s": round(float(np.mean(motion)), 2) if motion else None,
                      "face_max_frac_3s": round(face_frac, 4),
                      "face_closeup_3s": int(face_frac >= 0.08),
                      "keyframes_saved": len(saved)})
        # ---- audio: first 3 s
        c.seek(0)
        if c.streams.audio:
            rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
            chunks = []
            for fr in c.decode(c.streams.audio[0]):
                t = float(fr.pts * fr.time_base) if fr.pts is not None else 0.0
                if t > WIN:
                    break
                for o in rs.resample(fr):
                    chunks.append(o.to_ndarray().reshape(-1))
            if chunks:
                a = np.concatenate(chunks)[: int(WIN * 16000)]
                eps = 1e-9
                feats["audio_rms_db_3s"] = round(float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + eps)), 2)
                feats["audio_peak_db_3s"] = round(float(20 * np.log10(np.max(np.abs(a)) + eps)), 2)
        return feats
    finally:
        c.close()


def main():
    meta_csv, tdir, out_dir, kf_dir = map(Path, sys.argv[1:5])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(meta_csv, dtype={"ad_id": str}).set_index("ad_id")
    done = 0
    for tf in sorted(tdir.glob("*.json")):
        t = json.loads(tf.read_text(encoding="utf-8"))
        if t.get("status") != "ok" or t.get("language") != "en":
            continue
        ad_id = t["ad_id"]
        of = out_dir / f"{ad_id}.json"
        if of.exists() or ad_id not in meta.index:
            continue
        t0 = time.time()
        rec = {"ad_id": ad_id}
        try:
            rec.update(analyse(meta.loc[ad_id, "video_url"], ad_id, kf_dir))
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
        of.write_text(json.dumps(rec), encoding="utf-8")
        done += 1
        print(done, ad_id, rec.get("status"), rec.get("n_cuts_3s"), rec.get("motion_3s"),
              rec.get("face_max_frac_3s"), rec.get("audio_rms_db_3s"), f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
