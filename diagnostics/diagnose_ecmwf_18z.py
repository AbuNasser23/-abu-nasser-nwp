#!/usr/bin/env python3
"""Read-only side-by-side ECMWF 0.1 index diagnosis for 12Z and 18Z."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RAW_RUN = os.environ.get("ECMWF_RUN", "20261107T18Z").strip().upper()
try:
    RUN_DATE, RUN_HOUR = RAW_RUN.replace("Z", "").split("T", 1)
    datetime.strptime(RUN_DATE, "%Y%m%d")
    if RUN_HOUR not in ("00", "06", "12", "18"):
        raise ValueError("run hour must be 00, 06, 12, or 18")
except (ValueError, TypeError) as exc:
    raise SystemExit(f"Invalid ECMWF_RUN={RAW_RUN!r}; expected YYYYMMDDTHHZ") from exc

RUNS = ("12", "18")
UA = "abusnasir-obs/1.0 diagnostic"

def forecast_hours(run_hour):
    if run_hour in ("06", "18"):
        return list(range(0, 145, 3))
    return list(range(0, 145, 3)) + list(range(150, 241, 6))

def index_url(run_hour, hour, level):
    stamp = f"{RUN_DATE}{run_hour}0000"
    return (
        f"https://data.ecmwf.int/forecasts/{RUN_DATE}/{run_hour}z/"
        f"ifs/0p1/oper/{stamp}-{hour}h-oper-fc-{level}.index"
    )

def probe(item):
    run_hour, hour, level = item
    url = index_url(run_hour, hour, level)
    req = Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urlopen(req, timeout=30) as response:
            return run_hour, hour, level, response.status, response.headers.get("Content-Length"), None
    except HTTPError as exc:
        return run_hour, hour, level, exc.code, None, str(exc.reason)
    except (URLError, TimeoutError, OSError) as exc:
        return run_hour, hour, level, "NETWORK_ERROR", None, str(exc)

items = [
    (run_hour, hour, level)
    for run_hour in RUNS
    for hour in forecast_hours(run_hour)
    for level in ("sfc", "pl")
]
print(f"ECMWF 0.1 index comparison date={RUN_DATE} runs=12Z,18Z requests={len(items)}")
print("Provider note: ECMWF states the 0.1-degree tier is published about 2 hours after 0.25-degree data.")
results = {}
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(probe, item) for item in items]
    for future in as_completed(futures):
        row = future.result()
        results[row[:3]] = row

for run_hour in RUNS:
    hours = forecast_hours(run_hour)
    rows = [results[(run_hour, hour, level)] for hour in hours for level in ("sfc", "pl")]
    available = sum(row[3] in (200, 206) for row in rows)
    unavailable = len(rows) - available
    print(f"SUMMARY run={RUN_DATE}T{run_hour}Z horizon={hours[-1]}h steps={len(hours)} "
          f"available={available}/{len(rows)} unavailable={unavailable}")
    for hour in hours:
        for level in ("sfc", "pl"):
            row = results[(run_hour, hour, level)]
            if row[3] not in (200, 206):
                print(f"UNAVAILABLE run={run_hour}Z +{hour:03d}h {level} "
                      f"status={row[3]} error={row[5]} url={index_url(run_hour,hour,level)}")

for run_hour in RUNS:
    for hour in (0, 3, 144, 150, 240):
        if hour not in forecast_hours(run_hour):
            continue
        for level in ("sfc", "pl"):
            row = results[(run_hour, hour, level)]
            print(f"SAMPLE run={run_hour}Z +{hour:03d}h {level} status={row[3]} "
                  f"url={index_url(run_hour,hour,level)}")
