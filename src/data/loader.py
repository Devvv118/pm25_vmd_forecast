"""Load a clean "datetime,pm25" CSV (a per-station file or the merged
composite city series) into an hourly-indexed, gap-filled pd.Series.
"""
import pandas as pd


def load_pm25_series(csv_path: str, interpolate_limit: int = 6) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    # Time-interpolate only short gaps; longer gaps are left as NaN and
    # dropped below rather than silently invented.
    s = df["pm25"].interpolate(method="time", limit=interpolate_limit, limit_area="inside")
    s = s.dropna()

    # Lag windows assume consecutive hours; warn if dropped gaps break that.
    if len(s) > 1:
        n_breaks = int((s.index.to_series().diff() > pd.Timedelta(hours=1)).sum())
        if n_breaks:
            print(f"[loader] WARNING: {n_breaks} gap(s) longer than {interpolate_limit}h remain in "
                  f"{csv_path}; lag windows will span them.")
    return s
