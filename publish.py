#!/usr/bin/env python3
"""Publish one complete ECMWF or GFS cycle as static float32 packs.

The observatory is not modified. This writer keeps two finished cycles and
replaces current.json only after the new cycle has been checked on disk.
A failed model does not delete the other model's pointer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from eccodes import codes_get, codes_get_values, codes_new_from_message, codes_release

SOUTH, NORTH, WEST, EAST = -2.0, 32.0, 32.0, 75.0
NX, NY = 173, 137
UA = "abusnasir-obs/1.0"
HORIZON = 240
KEEP_DEFAULT = 2
CORE = (
    "t2m", "td2m", "rh2m", "mslp", "u10", "v10", "gusts", "precip", "clouds", "cape",
    "t850", "rh850", "td850", "z850", "u850", "v850",
    "t700", "rh700", "td700", "z700", "u700", "v700",
    "t500", "rh500", "z500", "u500", "v500",
    "t300", "rh300", "z300", "u300", "v300",
)
LEVELS = (850, 700, 500, 300)
ECMWF_SURFACE = {
    "2t": "t2m",
    "2d": "td2m",
    "msl": "mslp",
    "10u": "u10",
    "10v": "v10",
    "10fg": "gusts",
    "tp": "precip",
    "tcc": "clouds",
    "mucape": "cape",
}
ECMWF_PL = {"t": "t", "r": "rh", "gh": "z", "u": "u", "v": "v"}


def hours(model: str) -> list[int]:
    """Native deterministic output steps; never synthesize missing forecast hours."""
    if model == "ecmwf":
        return list(range(0, 145, 3)) + list(range(150, HORIZON + 1, 6))
    if model == "gfs":
        return list(range(0, 121, 3)) + list(range(126, HORIZON + 1, 6))
    raise RuntimeError(f"unknown model {model}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stamp_of(run: datetime) -> str:
    return run.strftime("%Y%m%dT%HZ")


def iso_of(run: datetime) -> str:
    return run.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    raw = text.replace(".000Z", "Z")
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def candidates(model: str, now: datetime, lookback: int = 8) -> list[datetime]:
    step = 12 if model == "ecmwf" else 6
    hour = (now.hour // step) * step
    cur = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return [cur - timedelta(hours=step * i) for i in range(lookback)]


def expected_times(run: datetime, model: str) -> list[str]:
    return [iso_of(run + timedelta(hours=hour)) for hour in hours(model)]


def head_ok(url: str, timeout: int = 25) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return 200 <= res.status < 300
    except urllib.error.HTTPError as exc:
        return exc.code in (200, 206)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def fetch(url: str, timeout: int = 90) -> bytes:
    delay = 2.0
    last = "fetch failed"
    for _ in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.read()
        except urllib.error.HTTPError as exc:
            last = f"{exc.code} {url}"
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        import time
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(last)


def fetch_range(url: str, start: int, end: int) -> bytes:
    delay = 2.0
    last = "range failed"
    for _ in range(4):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": UA, "Range": f"bytes={start}-{end - 1}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                if res.status not in (200, 206):
                    raise RuntimeError(f"{res.status} {url}")
                return res.read()
        except urllib.error.HTTPError as exc:
            last = f"{exc.code} {url}"
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        import time
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(last)


def ecmwf_index_url(run: datetime, hour: int) -> str:
    ymd = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    stamp = run.strftime("%Y%m%d%H") + "0000"
    return f"https://data.ecmwf.int/forecasts/{ymd}/{hh}z/ifs/0p25/oper/{stamp}-{hour}h-oper-fc.index"


def gfs_idx_url(run: datetime, hour: int) -> str:
    ymd = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    fff = f"{hour:03d}"
    return f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fff}.idx"


def step_marker(model: str, run: datetime, hour: int) -> str:
    return ecmwf_index_url(run, hour) if model == "ecmwf" else gfs_idx_url(run, hour)


def cycle_steps_ready(model: str, run: datetime) -> bool:
    urls = [step_marker(model, run, hour) for hour in hours(model)]
    ok = 0
    lock = threading.Lock()

    def one(url: str) -> bool:
        return head_ok(url)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ready in pool.map(one, urls):
            if ready:
                with lock:
                    ok += 1
    return ok == len(urls)


def discover(model: str, now: datetime | None = None) -> datetime:
    """Newest cycle with every native forecast step through +240h actually listed."""
    now = now or utcnow()
    if model not in ("ecmwf", "gfs"):
        raise RuntimeError(f"unknown model {model}")
    for run in candidates(model, now):
        if not head_ok(step_marker(model, run, HORIZON)):
            continue
        if cycle_steps_ready(model, run):
            return run
    raise RuntimeError(f"no complete {model} cycle through +{HORIZON}h at native steps")


def ecmwf_want(row: dict) -> str | None:
    param = row.get("param")
    lev = row.get("levtype")
    if lev == "sfc" and param in ECMWF_SURFACE:
        return ECMWF_SURFACE[param]
    if lev == "pl" and param in ECMWF_PL:
        level = int(row.get("levelist") or 0)
        if level in LEVELS:
            return f"{ECMWF_PL[param]}{level}"
    return None


def crop_meta(gid) -> tuple:
    return (
        float(codes_get(gid, "latitudeOfFirstGridPointInDegrees")),
        float(codes_get(gid, "longitudeOfFirstGridPointInDegrees")),
        float(codes_get(gid, "iDirectionIncrementInDegrees")),
        float(codes_get(gid, "jDirectionIncrementInDegrees")),
        int(codes_get(gid, "jScansPositively")),
        int(codes_get(gid, "Ni")),
        int(codes_get(gid, "Nj")),
    )


def indexes(lat1: float, lon1: float, di: float, dj: float, jpos: int, ni: int, nj: int):
    if jpos:
        j0 = int(round((SOUTH - lat1) / dj))
        j1 = int(round((NORTH - lat1) / dj))
    else:
        j0 = int(round((lat1 - NORTH) / dj))
        j1 = int(round((lat1 - SOUTH) / dj))

    def lon_i(lon: float) -> int:
        shift = lon - lon1
        if shift < -1:
            shift += 360
        return int(round(shift / di))

    i0 = lon_i(WEST)
    i1 = lon_i(EAST)
    if not (0 <= j0 <= j1 < nj and 0 <= i0 <= i1 < ni):
        raise RuntimeError(f"crop outside grid j {j0}-{j1} i {i0}-{i1} ni {ni} nj {nj}")
    lats = [round((lat1 + (dj if jpos else -dj) * j) * 1000) / 1000 for j in range(j0, j1 + 1)]
    lons = [round(((lon1 + di * i + 180) % 360 - 180) * 1000) / 1000 for i in range(i0, i1 + 1)]
    return j0, j1, i0, i1, lats, lons


def decode(buf: bytes, j0: int, j1: int, i0: int, i1: int, ni: int) -> np.ndarray:
    gid = codes_new_from_message(buf)
    try:
        vals = np.asarray(codes_get_values(gid), dtype=np.float32)
    finally:
        codes_release(gid)
    return np.array(vals.reshape((-1, ni))[j0 : j1 + 1, i0 : i1 + 1], dtype=np.float32, copy=True)


def rh_from_dew(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    def es(c):
        return 6.112 * np.exp((17.67 * c) / (c + 243.5))

    rh = 100.0 * es(td_c) / np.maximum(es(t_c), 1e-6)
    return np.clip(np.rint(rh), 0, 100).astype(np.float32)


def dewpoint_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rhc = np.clip(rh, 0.5, 100.0)
    a, b = 17.67, 243.5
    gamma = np.log(rhc / 100.0) + (a * t_c) / (b + t_c)
    return np.round(((b * gamma) / np.maximum(a - gamma, 1e-6)) * 10) / 10


def convert_ecmwf(name: str, field: np.ndarray) -> np.ndarray:
    if name.startswith("t") or name == "td2m":
        return np.round((field - 273.15) * 10) / 10
    if name.startswith("u") or name.startswith("v") or name == "gusts":
        return np.round(field * 36) / 10
    if name == "mslp":
        return np.round(field / 10) / 10
    if name == "clouds":
        scale = 100.0 if float(np.nanmax(field)) <= 1.5 else 1.0
        return np.clip(np.rint(field * scale), 0, 100)
    if name == "precip":
        return field
    if name.startswith("rh"):
        return np.clip(np.rint(field), 0, 100)
    if name.startswith("z") or name == "cape":
        out = field.astype(np.float32, copy=True)
        if name.startswith("z") and np.nanmedian(out) > 20000:
            out = out / 9.80665
        return np.rint(out)
    return field


def convert_gfs(name: str, field: np.ndarray) -> np.ndarray:
    if name.startswith("t") or name == "td2m":
        return np.round((field - 273.15) * 10) / 10
    if name.startswith("u") or name.startswith("v") or name == "gusts":
        return np.round(field * 36) / 10
    if name == "mslp":
        return np.round(field / 10) / 10
    if name.startswith("rh") or name == "clouds":
        return np.clip(np.rint(field), 0, 100)
    if name.startswith("z") or name == "cape":
        out = field.astype(np.float32, copy=True)
        if name.startswith("z") and np.nanmedian(out) > 20000:
            out = out / 9.80665
        return np.rint(out)
    if name == "precip":
        return np.round(field * 100) / 100
    return field


def fill_missing_ecmwf_gusts(fields: dict[str, np.ndarray]) -> bool:
    """Estimate missing gusts from 10 m wind; return True when an estimate is added.

    ECMWF Open Data does not expose 10fg at every forecast step. The 1.3
    gust factor is a documented fallback proxy, not a model-provided gust.
    Wind components are already converted to km/h by convert_ecmwf().
    """
    if "gusts" in fields:
        return False
    if "u10" not in fields or "v10" not in fields:
        return False
    speed = np.hypot(fields["u10"], fields["v10"])
    fields["gusts"] = (np.round(speed * 1.3 * 10) / 10).astype(np.float32)
    return True


def finish_derived(fields: dict[str, np.ndarray]) -> None:
    if "t2m" in fields and "td2m" in fields and "rh2m" not in fields:
        fields["rh2m"] = rh_from_dew(fields["t2m"], fields["td2m"])
    if "t850" in fields and "rh850" in fields:
        fields["td850"] = dewpoint_c(fields["t850"], fields["rh850"])
    if "t700" in fields and "rh700" in fields:
        fields["td700"] = dewpoint_c(fields["t700"], fields["rh700"])


def fill_missing_analysis_precip(model: str, hour: int, fields: dict, geo: dict) -> bool:
    """GFS f000 has no APCP message. Store zeros only for that hour."""
    if model != "gfs" or hour != 0 or "precip" in fields:
        return False
    sample = next((value for value in fields.values() if isinstance(value, np.ndarray) and value.ndim == 2), None)
    if sample is None:
        return False
    fields["precip"] = np.zeros(sample.shape, dtype=np.float32)
    geo["window"] = (0, 0)
    return True


def orient(lats: list[float], lons: list[float], fields: dict[str, np.ndarray]):
    if lats[0] > lats[-1]:
        lats = list(reversed(lats))
        fields = {name: np.flipud(field) for name, field in fields.items()}
    if lons[0] > lons[-1]:
        lons = list(reversed(lons))
        fields = {name: np.fliplr(field) for name, field in fields.items()}
    return lats, lons, fields


def gfs_name(gid) -> str | None:
    try:
        short = str(codes_get(gid, "shortName"))
    except Exception:
        return None
    try:
        level_type = str(codes_get(gid, "typeOfLevel"))
    except Exception:
        level_type = ""
    try:
        level = int(codes_get(gid, "level"))
    except Exception:
        level = 0
    surface = {
        "2t": "t2m",
        "2d": "td2m",
        "2r": "rh2m",
        "10u": "u10",
        "10v": "v10",
        "10fg": "gusts",
        "gust": "gusts",
        "msl": "mslp",
        "prmsl": "mslp",
        "tcc": "clouds",
        "cape": "cape",
        "mucape": "cape",
    }
    if short in surface and level_type != "isobaricInhPa":
        return surface[short]
    if short == "tp":
        return "precip"
    if level_type == "isobaricInhPa" and level in LEVELS and short in {"t", "r", "gh", "u", "v"}:
        prefix = {"t": "t", "r": "rh", "gh": "z", "u": "u", "v": "v"}[short]
        return f"{prefix}{level}"
    return None


def gfs_window(gid) -> tuple[int, int] | None:
    try:
        end_step = int(codes_get(gid, "endStep"))
        start_step = int(codes_get(gid, "startStep"))
    except Exception:
        try:
            span = str(codes_get(gid, "stepRange"))
            start_s, end_s = span.split("-")
            start_step, end_step = int(start_s), int(end_s)
        except Exception:
            return None
    hours_span = end_step - start_step
    if hours_span <= 0 or hours_span > HORIZON:
        return None
    return hours_span, end_step


def gfs_index_spans(text: str) -> list[tuple[str, int, int]]:
    lines = [line for line in text.splitlines() if line.strip()]
    spans = []
    for i, line in enumerate(lines):
        if ":APCP:" not in line and not any(
            key in line
            for key in (
                ":TMP:2 m above ground:",
                ":DPT:2 m above ground:",
                ":RH:2 m above ground:",
                ":UGRD:10 m above ground:",
                ":VGRD:10 m above ground:",
                ":GUST:",
                ":PRMSL:",
                ":TCDC:",
                ":CAPE:",
                ":TMP:850 mb:",
                ":RH:850 mb:",
                ":HGT:850 mb:",
                ":UGRD:850 mb:",
                ":VGRD:850 mb:",
                ":TMP:700 mb:",
                ":RH:700 mb:",
                ":HGT:700 mb:",
                ":UGRD:700 mb:",
                ":VGRD:700 mb:",
                ":TMP:500 mb:",
                ":RH:500 mb:",
                ":HGT:500 mb:",
                ":UGRD:500 mb:",
                ":VGRD:500 mb:",
                ":TMP:300 mb:",
                ":RH:300 mb:",
                ":HGT:300 mb:",
                ":UGRD:300 mb:",
                ":VGRD:300 mb:",
            )
        ):
            continue
        start = int(line.split(":")[1])
        end = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
        if end is None or end <= start:
            continue
        kind = "precip" if ":APCP:" in line else "field"
        spans.append((kind, start, end))
    return spans


def pull_gfs_blob(run: datetime, hour: int) -> bytes:
    idx_url = gfs_idx_url(run, hour)
    text = fetch(idx_url).decode("utf-8", "replace")
    base = idx_url[: -len(".idx")]
    spans = gfs_index_spans(text)
    if not spans:
        raise RuntimeError(f"gfs index has no fields at +{hour}h")
    parts = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_range, base, start, end): (kind, start) for kind, start, end in spans}
        ordered = []
        for fut in as_completed(futs):
            ordered.append((futs[fut][1], fut.result()))
        ordered.sort()
        parts = [buf for _, buf in ordered]
    return b"".join(parts)


def decode_gfs_step(blob: bytes) -> tuple[dict[str, np.ndarray], dict]:
    fields: dict[str, np.ndarray] = {}
    rains: list[tuple[int, int, np.ndarray]] = []
    geo = None
    offset = 0
    raw = blob
    while offset + 16 <= len(raw):
        if raw[offset : offset + 4] != b"GRIB":
            nxt = raw.find(b"GRIB", offset + 1)
            if nxt < 0:
                break
            offset = nxt
            continue
        length = int.from_bytes(raw[offset + 8 : offset + 16], "big")
        chunk = raw[offset : offset + length]
        offset += length
        gid = codes_new_from_message(chunk)
        try:
            name = gfs_name(gid)
            if not name:
                continue
            meta = crop_meta(gid)
            j0, j1, i0, i1, lats, lons = indexes(*meta)
            grid = decode(chunk, j0, j1, i0, i1, meta[5])
            if geo is None:
                geo = {"lats": lats, "lons": lons}
            if name == "precip":
                window = gfs_window(gid)
                if window:
                    rains.append((window[0], window[1], grid))
            elif name not in fields:
                fields[name] = grid
        finally:
            codes_release(gid)
    if geo is None:
        raise RuntimeError("gfs step produced no grid")
    if rains:
        rains.sort(key=lambda item: item[0])
        hours_span, end_step, grid = rains[0]
        fields["precip"] = convert_gfs("precip", grid)
        fields["_precip_window"] = np.array([hours_span, end_step], dtype=np.float32)
    for name in list(fields):
        if name.startswith("_") or name == "precip":
            continue
        fields[name] = convert_gfs(name, fields[name])
    window = None
    if "_precip_window" in fields:
        window_arr = fields.pop("_precip_window")
        window = (int(window_arr[0]), int(window_arr[1]))
    lats, lons, fields = orient(geo["lats"], geo["lons"], fields)
    return fields, {"lats": lats, "lons": lons, "window": window}


def pull_ecmwf_step(run: datetime, hour: int) -> tuple[dict[str, np.ndarray], dict]:
    base = ecmwf_index_url(run, hour)[: -len(".index")]
    rows = []
    for line in fetch(base + ".index").decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = ecmwf_want(row)
        if name:
            rows.append((name, int(row["_offset"]), int(row["_length"])))
    if not any(name == "t2m" for name, _, _ in rows):
        raise RuntimeError(f"ecmwf missing 2t at +{hour}h")
    blobs: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_range, base + ".grib2", off, off + ln): name for name, off, ln in rows}
        for fut in as_completed(futs):
            blobs[futs[fut]] = fut.result()
    gid = codes_new_from_message(blobs["t2m"])
    try:
        meta = crop_meta(gid)
    finally:
        codes_release(gid)
    j0, j1, i0, i1, lats, lons = indexes(*meta)
    fields = {name: convert_ecmwf(name, decode(buf, j0, j1, i0, i1, meta[5])) for name, buf in blobs.items()}
    lats, lons, fields = orient(lats, lons, fields)
    return fields, {"lats": lats, "lons": lons, "window": None}


def reduce_windows(steps: list[tuple[np.ndarray, int, int]], run: datetime) -> list[np.ndarray]:
    """Match reduceApcpWindow: subtract rain already stored inside a longer bucket."""
    out: list[np.ndarray] = []
    stored: list[tuple[int, np.ndarray]] = []
    run_ms = int(run.timestamp() * 1000)
    for raw, span, end_step in steps:
        end_ms = run_ms + end_step * 3_600_000
        start_ms = end_ms - span * 3_600_000
        covered = np.zeros(raw.shape, dtype=np.float64)
        for prev_end, prev in stored:
            if start_ms + 60_000 < prev_end < end_ms - 60_000:
                covered += prev
        piece = np.round(np.clip(raw.astype(np.float64) - covered, 0, None) * 100) / 100
        piece = piece.astype(np.float32)
        out.append(piece)
        stored.append((end_ms, piece))
    return out


def assemble(model: str, run: datetime) -> dict:
    frames = []
    lats = lons = None
    prev_tp = None
    windows = []
    notes = []
    for hour in hours(model):
        fields, geo = pull_ecmwf_step(run, hour) if model == "ecmwf" else decode_gfs_step(pull_gfs_blob(run, hour))
        if lats is None:
            lats, lons = geo["lats"], geo["lons"]
        elif geo["lats"] != lats or geo["lons"] != lons:
            raise RuntimeError("grid changed inside one cycle")
        if fill_missing_analysis_precip(model, hour, fields, geo):
            notes.append("gfs +0h APCP is absent at analysis; precip stored as zeros")
        if model == "ecmwf":
            if fill_missing_ecmwf_gusts(fields):
                notes.append(f"ECMWF gusts estimated from 10 m wind at +{hour}h (1.3 factor; not provider gust data)")
            tp = fields.get("precip")
            if tp is not None:
                step_mm = tp * 1000.0 if prev_tp is None else (tp - prev_tp) * 1000.0
                fields["precip"] = np.round(np.clip(step_mm, 0, None) * 100) / 100
                prev_tp = tp
        else:
            windows.append(geo.get("window"))
        finish_derived(fields)
        missing = [name for name in CORE if name not in fields]
        if missing:
            raise RuntimeError(f"+{hour}h missing {', '.join(missing)}")
        frames.append(fields)
        print(f"{model} +{hour}h", file=sys.stderr, flush=True)
    if model == "gfs":
        if any(item is None for item in windows):
            raise RuntimeError("gfs precipitation window missing")
        reduced = reduce_windows(
            [(frames[i]["precip"], windows[i][0], windows[i][1]) for i in range(len(frames))],
            run,
        )
        for frame, rain in zip(frames, reduced):
            frame["precip"] = rain
    stack = {name: np.stack([frame[name] for frame in frames]).astype(np.float32) for name in CORE}
    empty = np.full((len(frames), len(lats), len(lons)), np.nan, dtype=np.float32)
    stack["lit"] = empty.copy()
    stack["lifted"] = empty.copy()
    pack = {"model": model, "run": run, "lats": lats, "lons": lons, "fields": stack}
    if notes:
        pack["notes"] = notes
    return pack


def sanity(pack: dict) -> None:
    lats, lons = pack["lats"], pack["lons"]
    if len(lons) != NX or len(lats) != NY:
        raise RuntimeError(f"grid {len(lons)}x{len(lats)} is not {NX}x{NY}")
    if abs(lats[0] - SOUTH) > 0.02 or abs(lats[-1] - NORTH) > 0.02:
        raise RuntimeError("latitude window mismatch")
    if abs(lons[0] - WEST) > 0.02 or abs(lons[-1] - EAST) > 0.02:
        raise RuntimeError("longitude window mismatch")
    if any(abs((lats[i + 1] - lats[i]) - 0.25) > 0.02 for i in range(0, len(lats) - 1, 10)):
        raise RuntimeError("latitude spacing is not 0.25")
    t2m = pack["fields"]["t2m"][0]
    finite = np.isfinite(t2m)
    if finite.mean() < 0.9:
        raise RuntimeError("t2m is mostly missing")
    med = float(np.nanmedian(t2m))
    if not -50 <= med <= 55:
        raise RuntimeError(f"t2m median {med} is not Celsius")
    z500 = pack["fields"]["z500"][0]
    zmed = float(np.nanmedian(z500))
    if not 4000 <= zmed <= 6500:
        raise RuntimeError(f"z500 median {zmed} is not geopotential metres")
    mslp = float(np.nanmedian(pack["fields"]["mslp"][0]))
    if not 900 <= mslp <= 1100:
        raise RuntimeError(f"mslp median {mslp} is not hPa")
    wind = float(np.nanmedian(np.abs(pack["fields"]["u10"][0])))
    if wind > 250:
        raise RuntimeError(f"u10 median {wind} is not km/h")


def gzip_f32(values: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(values, dtype="<f4").tobytes()
    return gzip.compress(raw, compresslevel=6)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def bundle_files(pack: dict) -> tuple[dict[str, bytes], dict]:
    run = pack["run"]
    times = expected_times(run, pack["model"])
    files: dict[str, bytes] = {}
    digests = {}
    for name, values in pack["fields"].items():
        if values.shape != (len(times), NY, NX):
            raise RuntimeError(f"{name} shape {values.shape}")
        payload = gzip_f32(values)
        files[f"{name}.f32.bin"] = payload
        digests[name] = {"sha256": sha256(payload), "bytes": len(payload)}
    return files, digests


def meta_of(model: str, pack: dict, digests: dict) -> dict:
    run = pack["run"]
    times = expected_times(run, model)
    source = "ECMWF Open Data · IFS 0.25° مباشرة" if model == "ecmwf" else "NOAA GFS 0.25° مباشرة"
    label = "ECMWF IFS" if model == "ecmwf" else "GFS"
    return {
        "model": model,
        "nx": NX,
        "ny": NY,
        "nT": len(times),
        "nP": NX * NY,
        "lats": pack["lats"],
        "lons": pack["lons"],
        "times": times,
        "run": iso_of(run),
        "modelLabel": label,
        "source": source,
        "litReady": False,
        "fields": list(pack["fields"]),
        "digests": digests,
        "qualityNotes": pack.get("notes", []),
        "horizonHours": HORIZON,
    }


def validate_meta(meta: dict, model: str) -> None:
    if meta.get("model") != model:
        raise RuntimeError("model mismatch")
    if meta.get("nx") != NX or meta.get("ny") != NY or meta.get("nT") != len(hours(model)):
        raise RuntimeError("grid or step count mismatch")
    run = parse_iso(meta["run"])
    if meta.get("times") != expected_times(run, model):
        raise RuntimeError("valid times do not match the run")
    if meta.get("horizonHours") != HORIZON:
        raise RuntimeError("horizon is not 144h")
    for name in CORE:
        if name not in meta.get("fields", []):
            raise RuntimeError(f"meta missing {name}")


def cycle_dir(site: Path, model: str, run: datetime) -> Path:
    return site / model / "cycles" / stamp_of(run)


def pointer_matches(site: Path, model: str, run: datetime) -> bool:
    current = read_json(site / model / "current.json")
    if not current or current.get("complete") is not True:
        return False
    try:
        return parse_iso(current.get("run", "")) == run
    except ValueError:
        return False


def disk_cycle_ok(path: Path, model: str, run: datetime) -> bool:
    meta = read_json(path / "meta.json")
    if not meta:
        return False
    try:
        validate_meta(meta, model)
    except RuntimeError:
        return False
    if parse_iso(meta["run"]) != run:
        return False
    for name in list(CORE) + ["lit", "lifted"]:
        file = path / f"{name}.f32.bin"
        if not file.exists() or file.stat().st_size < 32:
            return False
        if file.read_bytes()[:2] != b"\x1f\x8b":
            return False
    return True


def write_cycle(site: Path, model: str, pack: dict) -> Path:
    sanity(pack)
    run = pack["run"]
    files, digests = bundle_files(pack)
    meta = meta_of(model, pack, digests)
    validate_meta(meta, model)
    dest = cycle_dir(site, model, run)
    partial = dest.parent / f".{dest.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    atomic_write(partial / "meta.json", json.dumps(meta, separators=(",", ":")).encode())
    for name, payload in files.items():
        atomic_write(partial / name, payload)
    manifest = {
        "model": model,
        "run": iso_of(run),
        "complete": True,
        "horizonHours": HORIZON,
        "nx": NX,
        "ny": NY,
        "nT": len(hours(model)),
        "source": meta["source"],
        "fetchedAt": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": digests,
    }
    if pack.get("notes"):
        manifest["notes"] = pack["notes"]
    atomic_write(partial / "manifest.json", json.dumps(manifest, indent=2).encode())
    if not disk_cycle_ok(partial, model, run):
        shutil.rmtree(partial, ignore_errors=True)
        raise RuntimeError("cycle failed re-read")
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    if not disk_cycle_ok(dest, model, run):
        raise RuntimeError("cycle missing after rename")
    return dest


def flip_pointer(site: Path, model: str, run: datetime, source: str) -> None:
    dest = cycle_dir(site, model, run)
    if not disk_cycle_ok(dest, model, run):
        raise RuntimeError("refusing to point at an incomplete cycle")
    previous = read_json(site / model / "current.json")
    prev_run = previous.get("run") if previous and previous.get("run") != iso_of(run) else (previous or {}).get("previous")
    body = {
        "model": model,
        "run": iso_of(run),
        "path": f"{model}/cycles/{stamp_of(run)}",
        "complete": True,
        "nT": len(hours(model)),
        "nx": NX,
        "ny": NY,
        "horizonHours": HORIZON,
        "publishedAt": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "previous": prev_run,
    }
    atomic_write(site / model / "current.json", json.dumps(body, indent=2).encode())
    written = read_json(site / model / "current.json")
    if not written or written.get("run") != iso_of(run) or written.get("complete") is not True:
        raise RuntimeError("pointer was not replaced")


def prune(site: Path, model: str, keep: int) -> None:
    current = read_json(site / model / "current.json") or {}
    keep_stamps = set()
    for key in ("run", "previous"):
        raw = current.get(key)
        if not raw:
            continue
        try:
            keep_stamps.add(stamp_of(parse_iso(raw)))
        except ValueError:
            continue
    root = site / model / "cycles"
    if not root.exists():
        return
    dirs = sorted([path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")])
    extras = [path for path in dirs if path.name not in keep_stamps]
    # If the pointer has no previous yet, retain the newest leftover so two cycles survive.
    while len(keep_stamps) + len(extras) > keep and extras:
        doomed = extras.pop(0)
        shutil.rmtree(doomed)
    for path in root.iterdir():
        if path.name.startswith(".") and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def write_status(site: Path, model: str, *, ok: bool, run: str | None, error: str | None, fail_count: int) -> None:
    delay_min = 0 if ok else min(360, 30 * (2 ** max(0, fail_count - 1)))
    nxt = utcnow() + timedelta(minutes=delay_min)
    body = {
        "model": model,
        "ok": ok,
        "run": run,
        "error": error,
        "failCount": 0 if ok else fail_count,
        "nextAttempt": nxt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "at": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write(site / model / "status.json", json.dumps(body, indent=2).encode())


def backoff_active(site: Path, model: str, run: datetime) -> bool:
    status = read_json(site / model / "status.json")
    if not status or status.get("ok") is not False:
        return False
    if status.get("run") and status["run"] != iso_of(run):
        return False
    try:
        nxt = parse_iso(status.get("nextAttempt", ""))
    except ValueError:
        return False
    return utcnow() < nxt


def publish_model(site: Path, model: str, *, keep: int = KEEP_DEFAULT, now: datetime | None = None) -> str:
    site.mkdir(parents=True, exist_ok=True)
    (site / model).mkdir(parents=True, exist_ok=True)
    status = read_json(site / model / "status.json") or {}
    fail_count = int(status.get("failCount") or 0)
    run = None
    try:
        run = discover(model, now)
        if pointer_matches(site, model, run):
            print(f"{model} already published {iso_of(run)}")
            return "skip"
        if backoff_active(site, model, run):
            print(f"{model} backoff until {(read_json(site / model / 'status.json') or {}).get('nextAttempt')}")
            return "backoff"
        dest = cycle_dir(site, model, run)
        if not disk_cycle_ok(dest, model, run):
            pack = assemble(model, run)
            if pack["run"] != run:
                raise RuntimeError("assembled run does not match the discovered run")
            dest = write_cycle(site, model, pack)
        source = "ECMWF Open Data IFS 0.25" if model == "ecmwf" else "NOAA GFS 0.25 AWS"
        flip_pointer(site, model, run, source)
        prune(site, model, keep)
        write_status(site, model, ok=True, run=iso_of(run), error=None, fail_count=0)
        print(f"{model} published {iso_of(run)}")
        return "published"
    except Exception as exc:
        write_status(
            site,
            model,
            ok=False,
            run=iso_of(run) if run is not None else None,
            error=str(exc)[-500:],
            fail_count=fail_count + 1,
        )
        print(f"{model} failed: {exc}", file=sys.stderr)
        return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("ecmwf", "gfs"))
    parser.add_argument("--site", required=True)
    parser.add_argument("--keep", type=int, default=KEEP_DEFAULT)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if args.discover_only:
        run = discover(args.model)
        print(iso_of(run))
        return
    result = publish_model(Path(args.site), args.model, keep=args.keep)
    if result == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
