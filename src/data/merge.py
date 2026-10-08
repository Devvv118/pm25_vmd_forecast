"""Fuse several per-station PM2.5 series into one composite city-level series.

Averaging across stations at each hour smooths over any single sensor's
dropouts and noise. Caveat: when the set of reporting stations changes from
hour to hour, a plain mean can jump if stations read at different levels.
fetch_city_pm25 already drops very gappy stations (min_station_coverage) to
limit this; the output also records `n_stations` per hour so you can check.
"""
import glob
import os

import pandas as pd


def load_station_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.set_index("datetime")["pm25"]


def merge_stations(csv_paths: list, method: str = "mean", min_stations: int = 1,
                   full_index: pd.DatetimeIndex = None) -> pd.DataFrame:
    """Combine per-station series -> DataFrame[pm25, n_stations] on an hourly index."""
    if not csv_paths:
        raise ValueError("No station CSVs to merge")
    combined = pd.concat([load_station_csv(p) for p in csv_paths], axis=1).sort_index()
    if full_index is not None:
        combined = combined.reindex(full_index)

    if method == "mean":
        merged = combined.mean(axis=1, skipna=True)
    elif method == "median":
        merged = combined.median(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown merge method: {method!r} (use 'mean' or 'median')")

    n = combined.notna().sum(axis=1)
    out = pd.DataFrame({"pm25": merged, "n_stations": n})
    return out[out["n_stations"] >= max(1, min_stations)]


def merge_csvs(csv_paths: list, out_path: str, method: str = "mean", min_stations: int = 1,
               full_index: pd.DatetimeIndex = None) -> str:
    """Merge an explicit list of station CSVs (no globbing, so stale files
    from earlier runs can't leak in) and write the composite."""
    merged = merge_stations(csv_paths, method=method, min_stations=min_stations,
                            full_index=full_index)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index_label="datetime")
    print(f"Merged {len(csv_paths)} station(s) ({method}) -> {len(merged)} hourly points "
          f"(stations/hour: min {int(merged.n_stations.min())}, "
          f"mean {merged.n_stations.mean():.1f}) -> {out_path}")
    return out_path


def merge_city_dir(raw_dir: str, city_prefix: str, out_path: str, method: str = "mean") -> str:
    """Legacy helper: merge every '<city_prefix>_*.csv' in raw_dir."""
    csv_paths = sorted(glob.glob(os.path.join(raw_dir, f"{city_prefix}_*.csv")))
    csv_paths = [p for p in csv_paths if os.path.basename(p) != os.path.basename(out_path)]
    if not csv_paths:
        raise FileNotFoundError(f"No '{city_prefix}_*.csv' station files found in {raw_dir}")
    return merge_csvs(csv_paths, out_path, method=method)
