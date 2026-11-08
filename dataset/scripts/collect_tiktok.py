"""Collect real ads from the public TikTok Ad Library (EU/UK DSA repository).

Loads library.tiktok.com in headless Chromium (the site signs its API calls in-page),
captures the raw JSON returned by /api/v1/search, and clicks "View more" to paginate.
Queries each official reach band separately (100K+, 10K-100K, 0-10K), video ads only.
Raw responses are saved unchanged to <out_dir> so every row is traceable.

usage: python collect_tiktok.py <out_dir> [region] [max_pages_per_band]
"""
import json, sys, time, urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1])
REGION = sys.argv[2] if len(sys.argv) > 2 else "GB"
MAX_PAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 3

QUERIES = ["ReelShort", "DramaBox", "ShortMax", "GoodShort", "FlexTV", "NetShort",
           "My Drama", "DramaWave", "MoboReels", "Kuku"]
BANDS = ["100K+", "10K-100K", "0-10K"]
# optional overrides: argv[4] = comma-separated queries, argv[5] = comma-separated bands
if len(sys.argv) > 4:
    QUERIES = [q.strip() for q in sys.argv[4].split(",") if q.strip()]
if len(sys.argv) > 5:
    BANDS = [b.strip() for b in sys.argv[5].split(",") if b.strip()]
SORT = "last_shown_date,desc"

OUT.mkdir(parents=True, exist_ok=True)
end_ms = int(time.time() * 1000)
start_ms = end_ms - 365 * 86400 * 1000
# optional argv[6] = number of date windows: the library caps each search at ~2-3 pages,
# so splitting the year into windows returns more (different) ads.
N_WIN = int(sys.argv[6]) if len(sys.argv) > 6 else 1
WINDOWS = [(start_ms + i * (end_ms - start_ms) // N_WIN, start_ms + (i + 1) * (end_ms - start_ms) // N_WIN)
           for i in range(N_WIN)]


def run_one(page, query, band, win=(start_ms, end_ms)):
    captured = []

    def on_resp(resp):
        if "/api/v1/search" in resp.url and resp.request.method == "POST":
            try:
                captured.append({"request": resp.request.post_data_json, "response": resp.json()})
            except Exception:
                pass

    page.on("response", on_resp)
    qs = urllib.parse.urlencode({
        "region": REGION, "start_time": win[0], "end_time": win[1],
        "adv_name": query, "adv_biz_ids": "", "query_type": "1", "sort_type": SORT,
        "ad_type": "1", "ad_status": "0", "ages": "all", "ad_reach": band, "gender": "ALL",
    })
    try:
        page.goto(f"https://library.tiktok.com/ads?{qs}", wait_until="networkidle", timeout=60000)
        for _ in range(20):
            if captured:
                break
            page.wait_for_timeout(500)
        for _ in range(MAX_PAGES - 1):
            if not captured or not captured[-1]["response"].get("has_more"):
                break
            btn = page.get_by_text("View more", exact=True)
            if btn.count() == 0:
                break
            n = len(captured)
            btn.first.click()
            for _ in range(30):
                page.wait_for_timeout(500)
                if len(captured) > n:
                    break
    finally:
        page.remove_listener("response", on_resp)
    return captured


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(locale="en-GB", user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"))
    page = ctx.new_page()
    for q in QUERIES:
        for band in BANDS:
          for wi, win in enumerate(WINDOWS):
            suffix = f"_w{wi:02d}" if N_WIN > 1 else ""
            fn = OUT / f"{REGION}_{q.replace(' ', '_')}_{band.replace('+', 'plus')}{suffix}.json"
            if fn.exists():
                continue
            pages = []
            for attempt in range(3):
                try:
                    pages = run_one(page, q, band, win)
                except Exception as e:
                    print(f"[{q} | {band}] attempt {attempt+1} error: {e}")
                if pages:
                    break
                time.sleep(3)
            n_ads = sum(len(x["response"].get("data") or []) for x in pages)
            total = pages[0]["response"].get("total") if pages else None
            sent = pages[0]["request"] if pages else None
            print(f"[{q} | {band}{suffix}] pages={len(pages)} ads={n_ads} total={total} "
                  f"sent_reach={sent and sent.get('ad_reach')} sent_order={sent and sent.get('order')}")
            if pages:
                fn.write_text(json.dumps({"query": q, "reach_band": band, "region": REGION,
                                          "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "pages": pages}, ensure_ascii=False), encoding="utf-8")
            time.sleep(2)
    browser.close()
