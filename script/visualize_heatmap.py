import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_heatmap(record, output_dir: Path, index: int):
    node_series = record["node_time_series"]
    node_names = list(node_series.keys())
    matrix = np.stack([np.array(node_series[name]) for name in node_names], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(node_names)))
    ax.set_yticklabels(node_names)
    ax.set_xlabel("Phase (%)")
    ax.set_title(f"Sample {index} Heatmap | Pred={record['pred_label']} True={record['true_label']}")
    fig.colorbar(im, ax=ax, label="LRP")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"heatmap_{index}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate node-time heatmaps.")
    parser.add_argument("--input", default="outputs/figures/explanations.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/figures/heatmaps")
    args = parser.parse_args()
    records = load_records(Path(args.input))
    output_dir = Path(args.output_dir)
    for idx, record in enumerate(records[: args.limit]):
        plot_heatmap(record, output_dir, idx)


if __name__ == "__main__":
    main()
