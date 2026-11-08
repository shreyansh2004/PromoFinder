import json, sys, time, requests

BASE = "https://library.tiktok.com"
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": BASE,
    "Referer": BASE + "/ads?region=GB",
})
S.get(BASE + "/ads?region=GB", timeout=30)

q = sys.argv[1] if len(sys.argv) > 1 else "ReelShort"
qt = sys.argv[2] if len(sys.argv) > 2 else "1"
order = sys.argv[3] if len(sys.argv) > 3 else "last_shown_date,desc"
end = int(time.time())
start = end - 365 * 86400
params = {"region": "GB", "type": "1", "start_time": start, "end_time": end}
data = {"query": q, "query_type": qt, "adv_biz_ids": "", "order": order, "offset": 0,
        "search_id": "", "limit": 12, "ad_type": 0, "ad_status": 0}
r = S.post(BASE + "/api/v1/search", params=params, json=data, timeout=30)
print(r.status_code)
j = r.json()
print({k: j.get(k) for k in ("code", "err_msg", "total", "has_more")})
for ad in (j.get("data") or [])[:3]:
    print(json.dumps(ad, ensure_ascii=False)[:1500])
