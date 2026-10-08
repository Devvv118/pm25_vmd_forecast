"""
Run the pipeline over every "*_pm25.csv" file in a directory (by default,
just the merged Chennai composite — but this also works if you later add
more cities, or want to compare individual station CSVs against it) and
write one combined summary table.

Usage:
    python scripts/run_all.py --raw_dir data/raw
    python scripts/run_all.py --raw_dir data/raw --pattern "chennai_*.csv"  # every station too
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.pipeline.orchestrator import run_pipeline
from src.utils.io import save_json, save_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--pattern", default="*_pm25.csv",
                         help="Glob pattern (relative to raw_dir) for the CSVs to run")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="outputs/results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_paths = sorted(glob.glob(os.path.join(args.raw_dir, args.pattern)))
    if not csv_paths:
        print(f"No files matching '{args.pattern}' found in {args.raw_dir}")
        return

    rows = []
    for csv_path in csv_paths:
        name = os.path.basename(csv_path).replace(".csv", "")
        print(f"Running pipeline for {name} ...")
        results, artifacts = run_pipeline(csv_path, cfg)

        save_json(results, os.path.join(args.out_dir, f"{name}_results.json"))
        save_artifacts(artifacts, os.path.join(args.out_dir, f"{name}_artifacts.npz"))

        for model_name, metrics in results.items():
            if model_name == "per_IMF":
                continue
            rows.append({"series": name, "model": model_name, **metrics})

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    print(summary.pivot(index="series", columns="model", values="MAE"))
    print("\nPlot any series with: python scripts/plot_results.py --station <name>")
    print("Or plot all of them with: python scripts/plot_results.py --all")


if __name__ == "__main__":
    main()
