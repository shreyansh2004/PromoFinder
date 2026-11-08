import time, urllib.parse, sys
from playwright.sync_api import sync_playwright

end_ms = int(time.time() * 1000); start_ms = end_ms - 365 * 86400 * 1000
qs = urllib.parse.urlencode({"region": "GB", "start_time": start_ms, "end_time": end_ms,
                             "adv_name": "short drama", "adv_biz_ids": "", "query_type": "1",
                             "sort_type": "last_shown_date,desc", "ad_type": "1", "ad_status": "0",
                             "ages": "all", "ad_reach": "100K+", "gender": "ALL"})
log = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    pg.on("response", lambda r: log.append((r.status, r.request.method, r.url[:90])) if "/api/v1/" in r.url else None)
    pg.goto(f"https://library.tiktok.com/ads?{qs}", wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout(3000)
    print("after load:", log)
    for txt in ["View more", "Load more", "More"]:
        print(txt, pg.get_by_text(txt).count())
    btns = pg.locator("button").all_inner_texts()
    print("buttons:", [t for t in btns if t.strip()][:30])
    for i in range(3):
        pg.mouse.wheel(0, 20000)
        pg.wait_for_timeout(2500)
    print("after scroll:", log[-4:])
    vm = pg.get_by_text("View more", exact=True)
    if vm.count():
        vm.first.scroll_into_view_if_needed(); vm.first.click(); pg.wait_for_timeout(4000)
        print("after click:", log[-4:])
    pg.screenshot(path=sys.argv[1], full_page=False)
    b.close()
