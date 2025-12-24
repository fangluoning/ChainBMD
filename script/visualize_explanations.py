import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Invalid JSON format: expected a list of records.")
    return data


def plot_record(record, output_dir: Path, index: int, events=None):
    node_series = record["node_time_series"]
    node_names = list(node_series.keys())
    num_nodes = len(node_names)
    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    x = np.linspace(0, 100, len(next(iter(node_series.values()))))
    curve_color = "#219EBC"

    for idx, name in enumerate(node_names):
        r = 0 if idx < cols else 1
        c = idx % cols
        ax = axes[r, c]
        series = np.array(node_series[name])
        ax.plot(x, series, color=curve_color, linewidth=2)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#b0b0b0", linestyle="-", linewidth=0.5)
        if events:
            for pos, style in events:
                ax.axvline(pos, color=style["color"], linestyle=style["linestyle"], linewidth=1.2)
        if r == rows - 1:
            ax.set_xlabel("Phase (%)")
        if c == 0:
            ax.set_ylabel("LRP (0-1)")

    # Turn off unused subplot (bottom row, 4th axis if only 7 nodes)
    if num_nodes < rows * cols:
        for idx in range(num_nodes, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis("off")

    fig.suptitle(
        f"Sample {index} | Pred={record['pred_label']} True={record['true_label']}",
        fontsize=14,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"explanation_{index}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LRP explanation JSON.")
    parser.add_argument("--input", default="outputs/figures/explanations.json", help="Input JSON path.")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--output_dir", default="outputs/figures/viz", help="Output image directory.")
    parser.add_argument("--events", nargs="*", type=float, help="Event percent positions, e.g., 60 85.")
    args = parser.parse_args()

    event_styles = []
    if args.events:
        colors = ["black", "green", "blue"]
        linestyles = ["-", "--", ":"]
        for i, ev in enumerate(args.events):
            event_styles.append(
                {"pos": ev, "color": colors[i % len(colors)], "linestyle": linestyles[i % len(linestyles)]}
            )

    records = load_records(Path(args.input))
    output_dir = Path(args.output_dir)
    events = [(e["pos"], {"color": e["color"], "linestyle": e["linestyle"]}) for e in event_styles]
    for idx, record in enumerate(records[: args.limit]):
        plot_record(record, output_dir, idx, events=events)


if __name__ == "__main__":
    main()
