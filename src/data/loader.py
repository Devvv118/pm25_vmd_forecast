"""Load a clean "datetime,pm25[,n_stations]" CSV (a per-station file or the
merged composite city series) into an hourly-indexed, gap-filled frame.
"""
import pandas as pd


def load_pm25_frame(csv_path: str, interpolate_limit: int = 6) -> pd.DataFrame:
    """Returns a DataFrame with 'pm25' and, if present in the CSV, 'n_stations'.

    Short gaps (<= interpolate_limit hours) are time-interpolated; longer gaps
    are dropped, not invented.
    """
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    out = pd.DataFrame({"pm25": df["pm25"].interpolate(
        method="time", limit=interpolate_limit, limit_area="inside")})
    if "n_stations" in df.columns:
        out["n_stations"] = df["n_stations"].ffill().bfill()
    out = out.dropna(subset=["pm25"])

    if len(out) > 1:
        n_breaks = int((out.index.to_series().diff() > pd.Timedelta(hours=1)).sum())
        if n_breaks:
            print(f"[loader] WARNING: {n_breaks} gap(s) longer than {interpolate_limit}h remain in "
                  f"{csv_path}; lag windows will span them.")
    return out

