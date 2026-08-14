# Hourly PM2.5 Forecasting — Hybrid VMD + Neural Network

Decomposes the PM2.5 series with VMD, forecasts each mode with BP (MLP) /
LSTM, fuses the mode forecasts, and benchmarks against single-model and
persistence baselines.

## Repo structure

```
pm25_vmd_forecast/
├── configs/
│   └── default.yaml         # every hyperparameter lives here
├── data/raw/                 # put PRSA_Data_*.csv (or CPCB/OpenAQ) files here
├── outputs/
│   ├── models/               # (optional) saved checkpoints
│   └── results/              # metrics JSON / summary.csv written here
├── src/
│   ├── config.py             # Config dataclass + YAML loader
│   ├── data/
│   │   ├── preprocessing.py  # hourly reindex + gap interpolation
│   │   └── loader.py         # CSV -> clean PM2.5 series, station discovery
│   ├── decomposition/
│   │   └── vmd.py            # VMD wrapper
│   ├── features/
│   │   └── windowing.py      # chronological split + lag-window arrays
│   ├── models/
│   │   ├── base.py           # Forecaster interface (fit/predict/name)
│   │   ├── bp.py              # MLPRegressor implementing the interface
│   │   ├── lstm.py            # PyTorch LSTM implementing the interface
│   │   └── registry.py        # config -> list of model factories
│   ├── evaluation/
│   │   └── metrics.py         # MAE / RMSE / MAPE / R2 + persistence
│   ├── pipeline/
│   │   ├── baseline.py        # stage 1: persistence baseline
│   │   ├── single_model.py    # stage 2: models on raw PM2.5
│   │   ├── hybrid_vmd.py      # stage 3: VMD -> per-IMF models -> fusion
│   │   └── orchestrator.py    # wires stages 1-3 together for one station
│   └── utils/
│       └── io.py              # save/load JSON
├── scripts/
│   └── run_all.py             # loop the orchestrator over every station CSV
├── main.py                    # thin CLI: run the orchestrator for one station
├── requirements.txt
└── README.md
```

**Why split it this way:** `main.py` and `scripts/run_all.py` both just call
`src/pipeline/orchestrator.py::run_pipeline_for_station`, so the single-station
and batch entry points can never drift apart. Each pipeline stage
(`baseline.py`, `single_model.py`, `hybrid_vmd.py`) does one thing. BP and
LSTM both implement the same `Forecaster` interface (`fit`/`predict`/`name`),
so the pipeline code loops over models generically instead of duplicating
logic for each model type — adding a GRU or ARIMA model later means writing
one new file in `src/models/` and adding it to `registry.py`, nothing else
changes.

## Setup

```bash
pip install -r requirements.txt
```

Place station CSVs into `data/raw/` (UCI Beijing Multi-Site Air Quality
dataset, or CPCB/OpenAQ data reshaped to the same columns: `year, month,
day, hour, PM2.5, ...`).

## Run — one station

```bash
python main.py --csv data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv
```

Optional flags: `--config configs/default.yaml` (hyperparameters),
`--out_dir outputs/results`.

## Run — every station + cross-city summary

```bash
python scripts/run_all.py --raw_dir data/raw
```

Writes one JSON per station plus `outputs/results/summary.csv`, a flat
table of MAE/RMSE/MAPE/R2 per model per station — this is the benchmarking
artifact for comparing hybrid vs single models vs persistence, and (once
you add non-Beijing CSVs) across cities.

## What each run produces

- **Persistence** — naive PM2.5(t+1) = PM2.5(t)
- **Single_BP / Single_LSTM** — model trained directly on raw PM2.5
- **VMD_BP / VMD_LSTM** — VMD decomposition into `k` IMFs, one BP/LSTM per
  IMF, forecasts summed (fusion) into a single PM2.5 prediction
- **per_IMF** — MAE/RMSE/MAPE/R2 for each individual IMF forecast

## Leakage-safety note

Objective 1 requires VMD to be fit on training data only. This is enforced
by calling VMD **separately** on the train segment and the test segment
(`src/decomposition/vmd.py`, invoked twice in `hybrid_vmd.py`) — each call
only ever sees data within its own segment. The train/test split itself is
strictly chronological (`src/features/windowing.py`), never shuffled.

## Configuration

All tunable values (`n_lags`, `k`, `test_frac`, VMD/BP/LSTM hyperparameters)
live in `configs/default.yaml`, loaded into a `Config` dataclass
(`src/config.py`). Copy the file, edit it, and pass `--config your_file.yaml`
rather than touching code.

## Plotting results

Every run now saves two files per station instead of one:

- `outputs/results/<station>_results.json` — the metrics (unchanged from before)
- `outputs/results/<station>_artifacts.npz` — actual/predicted arrays (with
  timestamps) for every model, plus the raw train/test VMD decomposition

Turn those into plots with:

```bash
python scripts/plot_results.py --station PRSA_Data_Aotizhongxin_20130301-20170228
# or, for every station you've already run:
python scripts/plot_results.py --all
```

This writes to `outputs/plots/<station>/`:

- `predictions_vs_actual.png` — actual PM2.5 vs every model's prediction, last N test hours
- `metrics_comparison.png` — grouped MAE/RMSE/MAPE bars across models
- `r2_comparison.png` — R2 bars (kept separate since it's a different scale)
- `imf_decomposition.png` — one subplot per IMF, train segment vs test segment
- `per_imf_error.png` — MAE per IMF, grouped by model (BP vs LSTM)

`--n_points` controls how many test hours the actual-vs-predicted plot shows (default 500).

## Cleaning up

```bash
bash scripts/clean.sh
```

Removes everything a run generates — `outputs/results/*`, `outputs/models/*`,
`outputs/plots/*`, and all `__pycache__`/`*.pyc` files — while keeping the
folder structure (`.gitkeep` files) intact.

## Extending

- **New model type** (e.g. GRU, ARIMA): implement `Forecaster` in
  `src/models/`, add it to `build_model_factories()` in
  `src/models/registry.py`. The pipeline stages need no changes.
- **New decomposition method** (EMD/EEMD): add a module under
  `src/decomposition/` with the same `(signal, ...) -> (K, len(signal))`
  contract as `vmd_decompose`, then swap it into `hybrid_vmd.py`.
- **Exogenous features** (TEMP, PRES, DEWP, WSPM, RAIN): extend
  `make_supervised` in `src/features/windowing.py` to stack extra columns
  into the lag window.
