"""
Hourly PM2.5 forecasting for one station: VMD + BP/LSTM hybrid vs single
models vs a persistence baseline.

Usage:
    python main.py --csv data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv
    python main.py --csv data/raw/<file>.csv --config configs/default.yaml
"""
import argparse
import os

from src.config import load_config
from src.data.loader import station_name_from_path
from src.pipeline.orchestrator import run_pipeline_for_station
from src.utils.io import save_json, save_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to a station CSV file")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config")
    parser.add_argument("--out_dir", default="outputs/results", help="Where to write results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results, artifacts = run_pipeline_for_station(args.csv, cfg)

    station = station_name_from_path(args.csv)
    results_path = os.path.join(args.out_dir, f"{station}_results.json")
    artifacts_path = os.path.join(args.out_dir, f"{station}_artifacts.npz")
    save_json(results, results_path)
    save_artifacts(artifacts, artifacts_path)

    print(f"\nResults saved to {results_path}")
    print(f"Prediction/IMF arrays saved to {artifacts_path}")
    print(f"Plot them with: python scripts/plot_results.py --station {station}\n")
    for name, metrics in results.items():
        if name != "per_IMF":
            print(f"{name:15s} {metrics}")


if __name__ == "__main__":
    main()
