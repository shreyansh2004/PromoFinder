"""Flatten raw TikTok Ad Library responses into one CSV. Values are copied, never invented.

usage: python parse_tiktok.py <raw_dir> <out_csv>
"""
import csv, json, sys, datetime as dt
from pathlib import Path

raw_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
rows, seen = [], {}

APP_KEYS = {"reelshort": "ReelShort", "dramabox": "DramaBox", "shortmax": "ShortMax",
            "goodshort": "GoodShort", "flextv": "FlexTV", "netshort": "NetShort",
            "my drama": "My Drama", "mydrama": "My Drama", "dramawave": "DramaWave",
            "moboreels": "MoboReels", "kuku": "Kuku TV"}


def to_date(ms):
    return dt.datetime.fromtimestamp(int(ms) / 1000, dt.UTC).strftime("%Y-%m-%d") if ms else ""


def reach_bounds(s):
    s = (s or "").upper().replace(" ", "")
    if "-" not in s:
        return "", ""
    lo, hi = s.split("-", 1)
    f = lambda x: int(float(x[:-1]) * (1000 if x.endswith("K") else 1_000_000)) if x[-1] in "KM" else int(x)
    return f(lo), f(hi)


for f in sorted(raw_dir.rglob("*.json")):
    blob = json.loads(f.read_text(encoding="utf-8"))
    for pg in blob["pages"]:
        for ad in pg["response"].get("data") or []:
            ad_id = str(ad.get("id", ""))
            if not ad_id:
                continue
            if ad_id in seen:
                seen[ad_id]["also_found_via"] += f";{blob['query']}"
                continue
            vids = ad.get("videos") or []
            v0 = vids[0] if vids else {}
            first, last = to_date(ad.get("first_shown_date")), to_date(ad.get("last_shown_date"))
            days = (dt.date.fromisoformat(last) - dt.date.fromisoformat(first)).days + 1 if first and last else ""
            lo, hi = reach_bounds(ad.get("estimated_audience"))
            title = ad.get("title") or ""
            app = next((v for k, v in APP_KEYS.items() if k in title.lower()), "")
            row = {
                "ad_id": ad_id,
                "source": "TikTok Ad Library (EU/UK DSA)",
                "search_region": blob["region"],
                "search_query": blob["query"],
                "also_found_via": "",
                "reach_band_filter": blob["reach_band"],
                "unique_users_seen": ad.get("estimated_audience", ""),
                "reach_low": lo, "reach_high": hi,
                "first_shown": first, "last_shown": last, "days_shown": days,
                "advertiser_legal_name": ad.get("name", ""),
                "ad_title": title,
                "app_in_title": app,
                "subject": ad.get("subject", ""),
                "audit_status": ad.get("audit_status", ""),
                "removed_note": "; ".join((ad.get("rejection_info") or {}).get("reasons") or []),
                "n_videos": len(vids),
                "video_url": v0.get("video_url", ""),
                "ad_url": f"https://library.tiktok.com/ads/detail/?ad_id={ad_id}",
                "collected_at": blob["collected_at"],
            }
            seen[ad_id] = row
            rows.append(row)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} unique ads -> {out_csv}")
