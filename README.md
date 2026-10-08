# Chennai Hourly PM2.5 Forecasting — Hybrid VMD + Neural Network

Fetches Chennai PM2.5 from the OpenAQ API, fuses all nearby stations into one
composite hourly series, then compares: persistence baseline, single BP/LSTM,
and hybrid VMD+BP / VMD+LSTM (one model per IMF, forecasts summed).

## Quick start

```bash
pip install -r requirements.txt

# 1. Get a free API key: https://explore.openaq.org/register
#    then copy .env.example to .env and paste your key in it (loaded automatically)
cp .env.example .env

# 2. Fetch + merge Chennai data (1 Apr - 30 Sep 2026)  ->  data/raw/chennai_pm25.csv
python scripts/fetch_data.py --dry_run   # optional: just list stations/sensors
python scripts/fetch_data.py

# 3. Train + evaluate everything
python main.py

# 4. Plots -> outputs/plots/chennai_pm25/
python scripts/plot_results.py --station chennai_pm25

# Clean generated outputs (keeps data/raw)
bash scripts/clean.sh
```

## What changed vs. the Beijing project

| Before (Beijing/PRSA) | Now (Chennai/OpenAQ) |
|---|---|
| Local CSVs, 12 stations | Data pulled live from OpenAQ v3 API |
| Each station run and compared separately | All stations near Chennai **fused into one composite series** (mean or median per hour) |
| Station-vs-station summary table | One city-level forecast; `run_all.py` kept only for optional multi-series runs |

Why fuse: OpenAQ stations inside one city are typically gappy. Averaging across
them fills individual dropouts and reduces single-sensor noise, and the goal is
one good Chennai forecast, not a station leaderboard. Per-station raw CSVs are
still saved (`data/raw/chennai_<id>_<name>.csv`) so you can run any one of them:
`python main.py --csv data/raw/chennai_<id>_<name>.csv`.

Models evaluated are unchanged: Persistence, Single_BP, Single_LSTM, VMD_BP,
VMD_LSTM, plus per-IMF errors.

## Data fetching (30-day window)

Default window is **1 Apr - 30 Sep 2026, inclusive, in local IST time** (4,392 hourly
slots, 6 months). Timestamps in all CSVs are naive IST. Changes from the first version:

- Window is sent to the API with an explicit `+05:30` offset (previously bare
  dates were read as UTC, shifting the window by 5.5 h and cutting off the 30th).
- Retries on 429 **and** 5xx/network errors; honours `Retry-After` and
  `x-ratelimit-*` headers; clear errors for bad API keys / bad parameters.
- `/locations` is paginated; sensors are read from the `/locations` response
  (one fewer API call per station); stations whose data ends before / starts
  after the window are skipped without any download.
- Cleaning: negatives and values > `max_value` dropped, hours outside the window
  dropped, optional per-hour API coverage filter, duplicate hours removed.
- Stations with < `min_station_coverage` (50%) of the window's hours are
  excluded so a gappy station can't make the merged mean jump around.
- Merge uses the explicit file list from the current run (stale CSVs from earlier
  runs can't leak in) and records `n_stations` per hour.
- `data/raw/fetch_report.csv` shows why each station was kept or dropped.
- `--dry_run` lists stations/sensors without downloading data.

## Configuration

Everything lives in `configs/default.yaml` (partial `openaq:` blocks are merged
with defaults). Any value can be overridden on `scripts/fetch_data.py`
(`--datetime_from`, `--radius_m`, `--min_station_coverage`, ...).

## Layout

```
configs/default.yaml
scripts/  fetch_data.py  run_all.py  plot_results.py  clean.sh
main.py
src/
  config.py
  data/        openaq_client.py  fetch.py  merge.py  loader.py
  decomposition/vmd.py
  features/windowing.py
  models/      base.py  bp.py  lstm.py  registry.py
  evaluation/metrics.py
  pipeline/    baseline.py  single_model.py  hybrid_vmd.py  orchestrator.py
  utils/io.py
```

## Notes and caveats

- **OpenAQ coverage varies.** Chennai may have few stations and gaps in the
  date range you choose. If `fetch_data.py` reports no stations/data, widen
  `--radius_m` (max 25000, an OpenAQ limit) or change the date range. Hours
  missing from *every* station stay missing; gaps up to 6 hours are
  time-interpolated, longer gaps are dropped (see `src/data/loader.py`).
- **Leakage safety:** VMD runs separately on the train and test segments, and
  the split is chronological. VMD returns one sample fewer for odd-length
  input; `hybrid_vmd.py` handles this by trimming the series to match.
- **LSTM** uses a GPU automatically if PyTorch detects one, else CPU.
- With ~4,400 hours the test set is ~880 hours; a longer range gives more stable comparisons.
