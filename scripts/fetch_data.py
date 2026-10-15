"""
Fetch hourly PM2.5 for a city from OpenAQ (default: Chennai, 1 Apr - 30 Sep 2026),
save one cleaned CSV per station, then fuse them into one composite series.

Requires a free OpenAQ API key: https://explore.openaq.org/register
    put OPENAQ_API_KEY=your_key_here in a .env file in the project root

Usage:
    python scripts/fetch_data.py                 # uses configs/default.yaml
    python scripts/fetch_data.py --dry_run       # list stations/sensors only, download nothing
    python scripts/fetch_data.py --datetime_from 2025-10-01 --datetime_to 2026-09-30 \
        --min_station_coverage 0.3

Dates are inclusive local (Asia/Kolkata) dates. Outputs:
    data/raw/chennai_<id>_<name>.csv   per-station hourly PM2.5
    data/raw/fetch_report.csv          why each station was kept/dropped
    data/raw/chennai_pm25.csv          merged composite (pm25, n_stations)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.openaq_client import OpenAQClient
from src.data.fetch import fetch_city_pm25
from src.data.fetch import make_window
from src.data.merge import merge_csvs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--city", default=None, help="Override configs/default.yaml openaq.city")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--radius_m", type=int, default=None)
    parser.add_argument("--datetime_from", default=None, help="ISO date, e.g. 2023-01-01")
    parser.add_argument("--datetime_to", default=None, help="ISO date, e.g. 2025-01-01")
    parser.add_argument("--merge_method", default=None, choices=["mean", "median"])
    parser.add_argument("--min_station_coverage", type=float, default=None)
    parser.add_argument("--min_stations", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="list stations only")
    parser.add_argument("--verbose", action="store_true", help="log API retries/throttling")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    cfg = load_config(args.config)
    oaq = cfg.openaq
    city = args.city or oaq["city"]
    lat = args.lat if args.lat is not None else oaq["lat"]
    lon = args.lon if args.lon is not None else oaq["lon"]
    radius_m = args.radius_m or oaq["radius_m"]
    datetime_from = args.datetime_from or oaq["datetime_from"]
    datetime_to = args.datetime_to or oaq["datetime_to"]
    merge_method = args.merge_method or oaq["merge_method"]

    tz = oaq["timezone"]
    min_cov = args.min_station_coverage if args.min_station_coverage is not None else oaq["min_station_coverage"]
    min_stations = args.min_stations or oaq["min_stations"]

    client = OpenAQClient(verbose=args.verbose)  # reads OPENAQ_API_KEY from .env

    print(f"Fetching PM2.5 for {city} ({lat}, {lon}), {datetime_from} -> {datetime_to} ({tz}) ...")
    station_csvs, report = fetch_city_pm25(
        client, lat=lat, lon=lon, radius_m=radius_m,
        datetime_from=datetime_from, datetime_to=datetime_to, out_dir=args.out_dir,
        city=city, tz=tz, min_station_coverage=min_cov,
        min_coverage_pct=oaq["min_coverage_pct"], max_value=oaq["max_value"],
        dry_run=args.dry_run,
    )
    print("\nStation report:\n" + report.to_string(index=False) if len(report) else "\nNo stations found.")

    if args.dry_run:
        return
    if not station_csvs:
        print("\nNo usable stations - nothing to merge. Check data/raw/fetch_report.csv; "
              "try --min_station_coverage 0.2 or a different date range.")
        return

    _, _, full_index = make_window(datetime_from, datetime_to, tz)
    merged_path = os.path.join(args.out_dir, f"{city}_pm25.csv")
    merge_csvs(station_csvs, merged_path, method=merge_method, min_stations=min_stations,
               full_index=full_index.tz_localize(None))

    print(f"\nDone. Run the pipeline with:\n  python main.py --csv {merged_path}")


if __name__ == "__main__":
    main()
