"""
Hourly PM2.5 forecasting: VMD + BP/LSTM hybrid vs single models vs a
persistence baseline, run on a single PM2.5 series (by default, the merged
Chennai composite produced by scripts/fetch_data.py).

Usage:
    python main.py                                   # uses data/raw/chennai_pm25.csv
    python main.py --csv data/raw/chennai_pm25.csv
    python main.py --csv data/raw/chennai_5234_some_station.csv   # a single station instead
"""
import argparse
import os

from src.config import load_config
from src.pipeline.orchestrator import run_pipeline
from src.utils.io import save_json, save_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/raw/chennai_pm25.csv",
                         help="PM2.5 series CSV with 'datetime,pm25' columns")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="outputs/results")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"'{args.csv}' not found. Run scripts/fetch_data.py first to pull "
            f"data from OpenAQ, or pass --csv pointing at an existing file."
        )

    cfg = load_config(args.config)
    results, artifacts = run_pipeline(args.csv, cfg)

    name = os.path.basename(args.csv).replace(".csv", "")
    results_path = os.path.join(args.out_dir, f"{name}_results.json")
    artifacts_path = os.path.join(args.out_dir, f"{name}_artifacts.npz")
    save_json(results, results_path)
    save_artifacts(artifacts, artifacts_path)

    print(f"\nResults saved to {results_path}")
    print(f"Prediction/IMF arrays saved to {artifacts_path}")
    print(f"Plot them with: python scripts/plot_results.py --station {name}\n")
    for model_name, metrics in results.items():
        if model_name != "per_IMF":
            print(f"{model_name:15s} {metrics}")


if __name__ == "__main__":
    main()
