"""
Run the pipeline over every station CSV in a directory and write one
combined summary table (MAE/RMSE/MAPE/R2 per model, per station) — useful
for the cross-city benchmarking called for in the problem statement.

Usage:
    python scripts/run_all.py --raw_dir data/raw --config configs/default.yaml
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import discover_station_csvs, station_name_from_path
from src.pipeline.orchestrator import run_pipeline_for_station
from src.utils.io import save_json, save_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="outputs/results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_paths = discover_station_csvs(args.raw_dir)
    if not csv_paths:
        print(f"No CSV files found in {args.raw_dir}")
        return

    rows = []
    for csv_path in csv_paths:
        station = station_name_from_path(csv_path)
        print(f"Running pipeline for {station} ...")
        results, artifacts = run_pipeline_for_station(csv_path, cfg)

        save_json(results, os.path.join(args.out_dir, f"{station}_results.json"))
        save_artifacts(artifacts, os.path.join(args.out_dir, f"{station}_artifacts.npz"))

        for model_name, metrics in results.items():
            if model_name == "per_IMF":
                continue
            rows.append({"station": station, "model": model_name, **metrics})

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    print(summary.pivot(index="station", columns="model", values="MAE"))
    print("\nPlot any station with: python scripts/plot_results.py --station <name>")
    print("Or plot all of them with: python scripts/plot_results.py --all")


if __name__ == "__main__":
    main()
