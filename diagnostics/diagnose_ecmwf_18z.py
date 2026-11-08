#!/usr/bin/env python3
"""Read-only side-by-side ECMWF 0.1 index URL diagnosis for 12Z and 18Z."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATE = "20261107"
RUNS = ("12", "18")
UA = "abusnasir-obs/1.0 diagnostic"

def forecast_hours(run_hour):
    if run_hour in ("06", "18"):
        return list(range(0, 145, 3))
    return list(range(0, 145, 3)) + list(range(150, 241, 6))

def index_url(run_hour, hour, level):
    stamp = f"{DATE}{run_hour}0000"
    return (
        f"https://data.ecmwf.int/forecasts/{DATE}/{run_hour}z/"
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
print(f"ECMWF 0.1 side-by-side index diagnostic date={DATE} total_requests={len(items)}")
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
    bad_hours = [
        hour for hour in hours
        if any(results[(run_hour, hour, level)][3] not in (200, 206) for level in ("sfc", "pl"))
    ]
    print(f"SUMMARY run={DATE}T{run_hour}Z horizon={hours[-1]}h steps={len(hours)} "
          f"available={available}/{len(rows)} missing_steps={len(bad_hours)}")
    for hour in bad_hours:
        for level in ("sfc", "pl"):
            row = results[(run_hour, hour, level)]
            if row[3] not in (200, 206):
                print(f"UNAVAILABLE run={run_hour}Z +{hour:03d}h {level} "
                      f"status={row[3]} error={row[5]} url={index_url(run_hour,hour,level)}")

# Compact endpoint samples help distinguish a bad URL convention from a cycle-wide delay.
for run_hour in RUNS:
    for hour in (0, 3, 144, 150, 240):
        if hour not in forecast_hours(run_hour):
            continue
        for level in ("sfc", "pl"):
            row = results[(run_hour, hour, level)]
            print(f"SAMPLE run={run_hour}Z +{hour:03d}h {level} status={row[3]} "
                  f"url={index_url(run_hour,hour,level)}")
