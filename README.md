# Evaluating the Impact of Causal Variational Mode Decomposition on Hourly PM2.5 Forecasts in Chennai

Fetches Chennai PM2.5 from the OpenAQ API, fuses all nearby stations into one
composite hourly series, then compares: persistence baseline, single BP/LSTM,
and VMD hybrids (VMD+BP / VMD+LSTM, one model per IMF, forecasts summed), built with both retrospective and causal decomposition.

## Quick start

```bash
pip install -r requirements.txt

# 1. Get a free API key: https://explore.openaq.org/register
#    then copy .env.example to .env and paste your key in it (loaded automatically)
cp .env.example .env

# 2. Fetch + merge Chennai data (1 Oct 2025 - 30 Sep 2026)  ->  data/raw/chennai_pm25.csv
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

## Evaluation design (reviewer-driven changes)

| Review comment | What the code does now |
|---|---|
| Retrospective VMD leaks future info | `VMD_*_causal`: rolling VMD at every forecast origin (`src/decomposition/rolling.py`, window `causal_window`, cached in `outputs/cache/`). `VMD_*_retrospective` is kept for side-by-side comparison. Modes are sorted by centre frequency so mode i is the same band everywhere. |
| Train/test modes are different signals | Causal modes come from one identical procedure for train and test. `results["_meta"]["mode_diagnostics"]` reports dominant period per mode (train vs test) and reconstruction RMSE (train vs test). |
| One run, no seed, no validation, fixed epochs | `n_seeds` (default 10) runs per stochastic model; scores are mean over seeds with `<metric>_std`. The last `val_frac` of the training samples is a chronological validation block used for early stopping and LSTM tuning. Seed spread is drawn in the prediction plot and as error bars. |
| One season only / composite changes | Default fetch window is 12 months (2025-10-01 to 2026-09-30). `n_stations` is used as a covariate (`use_n_stations`). Station-level runs: `python scripts/run_all.py --pattern "chennai_*.csv"`. |
| Thin baselines | Added `SeasonalNaive_24`, `SARIMA`, `Single_GBM` and `Single_LSTM_tuned` (grid searched on the validation block). |
| Metric fixes | Persistence scored on the same hours as all models; MASE added; MAPE excludes zero actuals, with `n_zero_actual` / `n_zero_hours_in_test` reported; `top_decile` gives MAE/RMSE only (no R2); residual lag-1 autocorrelation (`resid_acf1`) added. |

Flags: `--n_seeds 3`, `--no_causal`, `--no_retrospective`. The causal run costs one VMD
per hour (parallelised, cached after the first run).

## Data fetching (30-day window)

Default window is **1 Oct 2025 - 30 Sep 2026, inclusive, in local IST time** (12 months). Timestamps in all CSVs are naive IST. Changes from the first version:

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
- **Leakage:** decomposing train and test separately does NOT make VMD causal
  (each mode value depends on later samples in its segment). Only `*_causal` models
  are leakage-free; treat `*_retrospective` as an upper bound.
- Everything runs on CPU (no GPU needed). Install the smaller CPU build of PyTorch with
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- With ~4,400 hours the test set is ~880 hours; a longer range gives more stable comparisons.
