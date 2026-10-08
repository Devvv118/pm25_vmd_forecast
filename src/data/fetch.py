"""Fetch PM2.5 for a city from OpenAQ and save one cleaned CSV per station.

Time handling: the requested window is interpreted in the city's local time
(default Asia/Kolkata, UTC+05:30) - "2026-09-01" to "2026-09-30" means
2026-09-01 00:00 IST through 2026-09-30 23:00 IST inclusive (720 hours).
Saved timestamps are *naive local time* (IST), which keeps the daily cycle
aligned with the clock for the forecasting models.
"""
import os
import re

import pandas as pd

from .openaq_client import OpenAQClient, PM25_PARAMETER_ID

DEFAULT_TZ = "Asia/Kolkata"


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return text.strip("_") or "station"


def make_window(date_from: str, date_to: str, tz: str = DEFAULT_TZ):
    """Inclusive date range -> (start, end_exclusive, hourly DatetimeIndex), tz-aware."""
    start = pd.Timestamp(date_from).tz_localize(tz)
    end = (pd.Timestamp(date_to) + pd.Timedelta(days=1)).tz_localize(tz)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    return start, end, index


def hours_to_series(results: list, tz: str = DEFAULT_TZ, window=None,
                    max_value: float = 1000.0, min_coverage_pct: float = 0.0) -> pd.Series:
    """OpenAQ /sensors/{id}/hours results -> clean hourly pd.Series (naive local time).

    Drops: missing values, negatives / values above max_value (sensor
    faults), hours with reported coverage below min_coverage_pct (when the
    API reports it), duplicates, and anything outside `window` (start, end).
    """
    rows = []
    for r in results:
        dt = (r.get("period") or {}).get("datetimeFrom", {}).get("utc")
        value = r.get("value")
        if dt is None or value is None:
            continue
        cov = (r.get("coverage") or {}).get("percentCoverage")
        if min_coverage_pct and cov is not None and cov < min_coverage_pct:
            continue
        rows.append((dt, float(value)))
    if not rows:
        return pd.Series(dtype=float, name="pm25")

    df = pd.DataFrame(rows, columns=["datetime", "pm25"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz)
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= max_value)]
    if window is not None:
        start, end = window
        df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df["pm25"].rename("pm25")


def _iso_utc_field(obj):
    """datetimeFirst/Last -> tz-aware Timestamp or None."""
    utc = (obj or {}).get("utc") if isinstance(obj, dict) else None
    return pd.Timestamp(utc) if utc else None


def find_pm25_sensor_ids(client: OpenAQClient, location: dict) -> list:
    """All PM2.5 sensor ids at a station. Uses the sensors embedded in the
    /locations result; falls back to /locations/{id}/sensors if absent."""
    sensors = location.get("sensors") or client.get_location_sensors(location["id"])
    ids = []
    for s in sensors:
        p = s.get("parameter") or {}
        if p.get("id") == PM25_PARAMETER_ID or p.get("name") == "pm25":
            ids.append(s.get("id"))
    return ids


def fetch_city_pm25(client: OpenAQClient, lat: float, lon: float, radius_m: int,
                    datetime_from: str, datetime_to: str, out_dir: str,
                    city: str = "chennai", tz: str = DEFAULT_TZ,
                    min_station_coverage: float = 0.5, min_coverage_pct: float = 0.0,
                    max_value: float = 1000.0, dry_run: bool = False):
    """Fetch PM2.5 for every station near (lat, lon); save one CSV per usable station.

    Returns (csv_paths, report_df). report_df has one row per candidate
    station (status, hours observed, coverage) - also written to
    <out_dir>/fetch_report.csv so you can see exactly why a station was kept
    or dropped.
    """
    os.makedirs(out_dir, exist_ok=True)
    start, end, full_index = make_window(datetime_from, datetime_to, tz)
    n_expected = len(full_index)
    print(f"Window: {start} -> {end} (exclusive), {n_expected} hourly slots")

    locations = client.find_locations(lat, lon, radius_m=radius_m)
    print(f"Found {len(locations)} station(s) within {radius_m / 1000:.0f} km")

    report, written = [], []
    for loc in locations:
        loc_id = loc.get("id")
        loc_name = loc.get("name") or f"station_{loc_id}"
        row = {"location_id": loc_id, "name": loc_name, "sensor_ids": "",
               "hours": 0, "coverage": 0.0, "status": ""}
        report.append(row)

        first, last = _iso_utc_field(loc.get("datetimeFirst")), _iso_utc_field(loc.get("datetimeLast"))
        if last is not None and last < start:
            row["status"] = f"skipped: last data {last.date()} is before window"
            print(f"  [skip] {loc_name} (id={loc_id}) - {row['status']}")
            continue
        if first is not None and first >= end:
            row["status"] = f"skipped: first data {first.date()} is after window"
            print(f"  [skip] {loc_name} (id={loc_id}) - {row['status']}")
            continue

        sensor_ids = find_pm25_sensor_ids(client, loc)
        row["sensor_ids"] = " ".join(map(str, sensor_ids))
        if not sensor_ids:
            row["status"] = "skipped: no PM2.5 sensor"
            print(f"  [skip] {loc_name} (id={loc_id}) - no PM2.5 sensor")
            continue
        if dry_run:
            row["status"] = "dry-run (not fetched)"
            print(f"  [dry-run] {loc_name} (id={loc_id}, sensors={sensor_ids})")
            continue

        # Ask for the exact window, with explicit UTC offset.
        dt_from, dt_to = start.isoformat(), end.isoformat()
        series_list = []
        for sid in sensor_ids:
            print(f"  Fetching {loc_name} (id={loc_id}, sensor={sid}) ...")
            try:
                results = client.get_sensor_hours(sid, dt_from, dt_to)
            except Exception as exc:  # one bad sensor shouldn't kill the run
                print(f"    [error] sensor {sid}: {exc}")
                continue
            s = hours_to_series(results, tz=tz, window=(start, end),
                                max_value=max_value, min_coverage_pct=min_coverage_pct)
            if not s.empty:
                series_list.append(s)

        if not series_list:
            row["status"] = "skipped: no data in window"
            print(f"  [skip] {loc_name} - no data in window")
            continue

        series = pd.concat(series_list, axis=1).mean(axis=1).rename("pm25")
        series = series.reindex(full_index.tz_localize(None)).dropna()
        coverage = len(series) / n_expected
        row["hours"], row["coverage"] = len(series), round(coverage, 3)

        if coverage < min_station_coverage:
            row["status"] = f"skipped: coverage {coverage:.0%} < {min_station_coverage:.0%}"
            print(f"  [skip] {loc_name} - {row['status']}")
            continue

        out_path = os.path.join(out_dir, f"{city}_{loc_id}_{_slugify(loc_name)}.csv")
        series.to_csv(out_path, index_label="datetime")
        row["status"] = "saved"
        print(f"  -> {len(series)}/{n_expected} hours ({coverage:.0%}) saved to {out_path}")
        written.append(out_path)

    report_df = pd.DataFrame(report)
    report_df.to_csv(os.path.join(out_dir, "fetch_report.csv"), index=False)
    return written, report_df
