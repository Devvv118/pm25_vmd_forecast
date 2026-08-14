"""
Generate plots from a station's saved results (*_results.json) and
artifacts (*_artifacts.npz): actual-vs-predicted curves, metric comparison
bars, the VMD decomposition, and per-IMF error.

Usage:
    python scripts/plot_results.py --station PRSA_Data_Aotizhongxin_20130301-20170228
    python scripts/plot_results.py --all
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed; just write PNG files
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_json, load_artifacts


def plot_predictions_vs_actual(artifacts: dict, out_path: str, n_points: int = 500):
    """Overlay each model's last `n_points` test-hour predictions against actual PM2.5."""
    model_keys = [k for k in artifacts if k != "imfs"]

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted_actual = False
    for key in model_keys:
        ts = pd.to_datetime(artifacts[key]["timestamps"])[-n_points:]
        actual = artifacts[key]["actual"][-n_points:]
        pred = artifacts[key]["predicted"][-n_points:]
        if not plotted_actual:
            ax.plot(ts, actual, label="Actual", color="black", linewidth=1.6)
            plotted_actual = True
        ax.plot(ts, pred, label=key, alpha=0.8, linewidth=1.1)

    ax.set_title(f"Actual vs predicted PM2.5 (last {n_points} test hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(results: dict, out_path: str):
    """Grouped bar chart of MAE/RMSE/MAPE across every top-level model."""
    model_keys = [k for k in results if k != "per_IMF"]
    metric_names = ["MAE", "RMSE", "MAPE"]

    x = range(len(model_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metric_names):
        values = [results[m][metric] for m in model_keys]
        ax.bar([xi + i * width for xi in x], values, width, label=metric)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(model_keys, rotation=30, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Model comparison (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_r2_bar(results: dict, out_path: str):
    """R2 needs its own chart since it's on a different scale than MAE/RMSE."""
    model_keys = [k for k in results if k != "per_IMF"]
    values = [results[m]["R2"] for m in model_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_keys, values, color="teal")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel("R2")
    ax.set_title("R2 comparison (higher is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_imf_decomposition(artifacts: dict, out_path: str):
    """One subplot per IMF, showing the train and test segments."""
    imf_data = artifacts.get("imfs")
    if imf_data is None:
        return

    train_ts = pd.to_datetime(imf_data["train_timestamps"])
    test_ts = pd.to_datetime(imf_data["test_timestamps"])
    train_imfs = imf_data["train_imfs"]
    test_imfs = imf_data["test_imfs"]
    k = train_imfs.shape[0]

    fig, axes = plt.subplots(k, 1, figsize=(14, 2.2 * k), sharex=False)
    axes = [axes] if k == 1 else list(axes)
    for i in range(k):
        axes[i].plot(train_ts, train_imfs[i], color="steelblue", linewidth=0.7, label="train")
        axes[i].plot(test_ts, test_imfs[i], color="darkorange", linewidth=0.7, label="test")
        axes[i].set_ylabel(f"IMF{i + 1}")
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time")
    fig.suptitle("VMD decomposition (train fit separately from test — see README)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_imf_error(results: dict, out_path: str):
    """Bar chart of per-IMF MAE, grouped by model."""
    per_imf = results.get("per_IMF")
    if not per_imf:
        return

    rows = []
    for key, metrics in per_imf.items():
        imf_num, model = re.match(r"IMF(\d+)_(\w+)", key).groups()
        rows.append({"IMF": int(imf_num), "model": model, "MAE": metrics["MAE"]})
    pivot = pd.DataFrame(rows).sort_values("IMF").pivot(index="IMF", columns="model", values="MAE")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE")
    ax.set_xlabel("IMF")
    ax.set_title("Per-IMF forecast error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_station(station: str, results_dir: str, plots_dir: str, n_points: int):
    results = load_json(os.path.join(results_dir, f"{station}_results.json"))
    artifacts = load_artifacts(os.path.join(results_dir, f"{station}_artifacts.npz"))

    out_dir = os.path.join(plots_dir, station)
    os.makedirs(out_dir, exist_ok=True)

    plot_predictions_vs_actual(artifacts, os.path.join(out_dir, "predictions_vs_actual.png"), n_points)
    plot_metrics_bar(results, os.path.join(out_dir, "metrics_comparison.png"))
    plot_r2_bar(results, os.path.join(out_dir, "r2_comparison.png"))
    plot_imf_decomposition(artifacts, os.path.join(out_dir, "imf_decomposition.png"))
    plot_per_imf_error(results, os.path.join(out_dir, "per_imf_error.png"))

    print(f"Plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", help="Station name, matching <station>_results.json")
    parser.add_argument("--all", action="store_true", help="Plot every station found in results_dir")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--plots_dir", default="outputs/plots")
    parser.add_argument("--n_points", type=int, default=500,
                         help="Hours to show in the actual-vs-predicted plot")
    args = parser.parse_args()

    if not args.station and not args.all:
        parser.error("pass --station <name> or --all")

    if args.all:
        result_files = glob.glob(os.path.join(args.results_dir, "*_results.json"))
        stations = [os.path.basename(f).replace("_results.json", "") for f in result_files]
        if not stations:
            print(f"No *_results.json files found in {args.results_dir}")
            return
        for station in stations:
            plot_station(station, args.results_dir, args.plots_dir, args.n_points)
    else:
        plot_station(args.station, args.results_dir, args.plots_dir, args.n_points)


if __name__ == "__main__":
    main()
