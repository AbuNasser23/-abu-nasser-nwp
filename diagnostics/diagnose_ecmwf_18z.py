#!/usr/bin/env python3
"""Read-only diagnosis of ECMWF 0.1 degree 18Z index availability."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import os
import time

RUN_TEXT = os.getenv("ECMWF_RUN", "20261107T18Z")
run = datetime.strptime(RUN_TEXT, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
hours = list(range(0, 145, 3))
UA = "abusnasir-obs/1.0 diagnostic"

def index_url(hour, level):
    stamp = run.strftime("%Y%m%d%H") + "0000"
    return (
        f"https://data.ecmwf.int/forecasts/{run:%Y%m%d}/{run:%H}z/"
        f"ifs/0p1/oper/{stamp}-{hour}h-oper-fc-{level}.index"
    )

def probe(item):
    hour, level = item
    url = index_url(hour, level)
    req = Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urlopen(req, timeout=30) as response:
            return hour, level, response.status, response.headers.get("Content-Length"), None
    except HTTPError as exc:
        return hour, level, exc.code, None, str(exc.reason)
    except (URLError, TimeoutError, OSError) as exc:
        return hour, level, "NETWORK_ERROR", None, str(exc)

items = [(hour, level) for hour in hours for level in ("sfc", "pl")]
print(f"ECMWF 0.1 index diagnostic run={run.isoformat()} steps={len(hours)} files={len(items)}")
results = {}
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(probe, item) for item in items]
    for future in as_completed(futures):
        row = future.result()
        results[(row[0], row[1])] = row

missing = []
for hour in hours:
    pair = [results[(hour, level)] for level in ("sfc", "pl")]
    states = " ".join(f"{level}={status}" for _, level, status, _, _ in pair)
    if any(status not in (200, 206) for _, _, status, _, _ in pair):
        missing.append(hour)
        print(f"MISSING_STEP +{hour:03d}h {states}")
print(f"SUMMARY available={len(items)-sum(1 for r in results.values() if r[2] not in (200,206))}/{len(items)} missing_steps={len(missing)}")
if missing:
    print("FAILED_URLS:")
    for hour in missing:
        for level in ("sfc", "pl"):
            _, _, status, size, error = results[(hour, level)]
            if status not in (200, 206):
                print(f"+{hour:03d}h {level} status={status} size={size} error={error} url={index_url(hour,level)}")
else:
    print("ALL_INDEX_FILES_PRESENT: publisher HEAD readiness gate should accept this cycle.")
