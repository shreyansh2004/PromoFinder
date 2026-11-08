"""Make upload files for tagging ads in the ChatGPT web app (no API key needed).

Writes:
  gpt/text_batches/text_batch_XX.md   ~40 ads each: title, hashtags, transcribed hook + ending
  gpt/image_sheets/sheet_XX.jpg       8 ads each: 3 real keyframes (0.5s, 1.5s, 2.5s) + ad_id label
  gpt/image_sheets/sheet_XX.txt       the ad_ids on that sheet (used to validate answers)

Only English, short-drama, first-of-creative ads are included (no duplicates).
usage: python make_gpt_batches.py <ads_dataset.csv> <keyframes_dir> <gpt_dir> [text_batch_size]
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

csv_path, kf_dir, gpt_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 40
# optional argv[5]: file with ad_ids -> writes only a redo batch (gpt/redo/redo_text.md)
REDO_IDS = Path(sys.argv[5]).read_text(encoding="utf-8").split() if len(sys.argv) > 5 else None
d = pd.read_csv(csv_path, dtype={"ad_id": str})
d = d[(d.language == "en") & (d.is_short_drama_ad == 1) & (d.is_first_of_creative == 1)].reset_index(drop=True)


def ending_text(r):
    if isinstance(r.ending_20s_text, str) and r.ending_20s_text.strip():
        return r.ending_20s_text.strip()
    if (r.duration_s or 0) > 180:
        return "[not transcribed: ad is longer than 3 minutes -> use ending_type \"unclear\"]"
    return "[no speech]"


def ad_block(r):
    return [f"## ad_id: {r.ad_id}",
            f"- app: {s(r.app)} | duration: {s(r.duration_s)} s",
            f"- ad title (written by advertiser): {s(r.ad_title)}",
            f"- hashtags in title: {s(r.title_hashtags) or 'none'}",
            f"- first 3 seconds (spoken): {s(r.first_3s_text) or '[no speech in first 3 s]'}",
            f"- first 15 seconds (spoken) -> judge hook_type from THIS: {s(r.hook_15s_text) or '[no speech]'}",
            f"- last 20 seconds (spoken): {ending_text(r)}", ""]


def s(v):
    return "" if pd.isna(v) else str(v).strip()


if REDO_IDS is not None:
    part = d[d.ad_id.isin(REDO_IDS)]
    out = gpt_dir / "redo"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Redo batch ({len(part)} ads)\n"] + [x for _, r in part.iterrows() for x in ad_block(r)]
    (out / "redo_text.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"redo batch: {len(part)} ads -> {out / 'redo_text.md'}")
    sys.exit(0)

tb, sh = gpt_dir / "text_batches", gpt_dir / "image_sheets"
for p in (tb, sh, gpt_dir / "answers"):
    p.mkdir(parents=True, exist_ok=True)

# ---------- text batches
for b, start in enumerate(range(0, len(d), BATCH), 1):
    part = d.iloc[start:start + BATCH]
    lines = [f"# Text batch {b:02d} ({len(part)} ads)\n"] + [x for _, r in part.iterrows() for x in ad_block(r)]
    (tb / f"text_batch_{b:02d}.md").write_text("\n".join(lines), encoding="utf-8")
print(f"text batches: {b} files, {len(d)} ads")

# ---------- image contact sheets
TW, TH, BAR, PER = 180, 320, 44, 8
have = [r.ad_id for _, r in d.iterrows() if (kf_dir / f"{r.ad_id}_1.5s.jpg").exists()]
n_sheets = 0
for k, start in enumerate(range(0, len(have), PER), 1):
    ids = have[start:start + PER]
    tiles = []
    for ad_id in ids:
        frames = []
        for t in ("0.5", "1.5", "2.5"):
            img = cv2.imread(str(kf_dir / f"{ad_id}_{t}s.jpg"))
            img = np.full((TH, TW, 3), 40, np.uint8) if img is None else cv2.resize(img, (TW, TH))
            frames.append(img)
        tile = np.hstack(frames)
        bar = np.full((BAR, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"ad_id {ad_id}", (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile, np.full((10, tile.shape[1], 3), 255, np.uint8)]))
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([tiles[i], np.full((tiles[i].shape[0], 16, 3), 255, np.uint8), tiles[i + 1]])
            for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(sh / f"sheet_{k:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (sh / f"sheet_{k:02d}.txt").write_text("\n".join(ids), encoding="utf-8")
    n_sheets = k
print(f"image sheets: {n_sheets} files, {len(have)} ads with keyframes")
