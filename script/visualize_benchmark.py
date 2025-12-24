import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_benchmark(benchmark, output_path: Path):
    node_stats = benchmark["node_stats"]
    node_names = list(node_stats.keys())
    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6), sharex=True)
    axes = np.array(axes).reshape(rows, cols)
    x = np.linspace(0, 100, len(next(iter(node_stats.values()))["mean"]))

    for idx, name in enumerate(node_names):
        r = 0 if idx < cols else 1
        c = idx % cols
        ax = axes[r, c]
        stats = node_stats[name]
        median = np.array(stats.get("median") or stats["mean"])
        lower = np.array(stats.get("p10") or (median - np.array(stats["std"])))
        upper = np.array(stats.get("p90") or (median + np.array(stats["std"])))
        ax.fill_between(x, lower, upper, color="#1d3557", alpha=0.25, label="Range (P10-P90)")
        ax.plot(x, median, color="#1d3557", linewidth=2, label="Group median")
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#b0b0b0", linestyle="-", linewidth=0.5)
        if r == rows - 1:
            ax.set_xlabel("Phase (%)")
        if c == 0:
            ax.set_ylabel("LRP (0-1)")
    for idx in range(len(node_names), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
    fig.suptitle(
        f"Benchmark Range skill_level={benchmark['skill_level']} (n={benchmark['sample_count']})",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sample_vs_benchmark(record, benchmark, output_path: Path):
    node_stats = benchmark["node_stats"]
    node_names = list(node_stats.keys())
    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6), sharex=True)
    axes = np.array(axes).reshape(rows, cols)
    x = np.linspace(0, 100, len(next(iter(node_stats.values()))["mean"]))

    for idx, name in enumerate(node_names):
        r = 0 if idx < cols else 1
        c = idx % cols
        ax = axes[r, c]
        stats = node_stats[name]
        median = np.array(stats.get("median") or stats["mean"])
        lower = np.array(stats.get("p10") or (median - np.array(stats["std"])))
        upper = np.array(stats.get("p90") or (median + np.array(stats["std"])))
        sample_series = np.array(record["node_time_series"][name])
        ax.fill_between(x, lower, upper, color="#1d3557", alpha=0.25, label="Range (P10-P90)")
        ax.plot(x, median, color="#1d3557", linewidth=1.5, label="Group median")
        ax.plot(x, sample_series, color="#e63946", linewidth=2, label="Sample")
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#b0b0b0", linestyle="-", linewidth=0.5)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        if r == rows - 1:
            ax.set_xlabel("Phase (%)")
        if c == 0:
            ax.set_ylabel("LRP (0-1)")
    for idx in range(len(node_names), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
    fig.suptitle(
        f"Sample vs Benchmark | pred={record['pred_label']} true={record['true_label']}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize LRP benchmarks.")
    parser.add_argument("--benchmark", default="outputs/figures/benchmark_skill2.json")
    parser.add_argument("--record", help="Optional: compare a single-sample JSON to the benchmark.")
    parser.add_argument("--output_dir", default="outputs/figures/benchmark_viz")
    args = parser.parse_args()
    benchmark = load_json(Path(args.benchmark))
    output_dir = Path(args.output_dir)
    plot_benchmark(benchmark, output_dir / "benchmark_range.png")
    if args.record:
        record = load_json(Path(args.record))
        if isinstance(record, list):
            record = record[0]
        plot_sample_vs_benchmark(record, benchmark, output_dir / "sample_vs_benchmark.png")


if __name__ == "__main__":
    main()
