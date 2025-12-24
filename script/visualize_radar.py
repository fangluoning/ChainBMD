import argparse
import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_radar(record, output_dir: Path, index: int):
    node_names = list(record["node_contributions"].keys())
    values = list(record["node_contributions"].values())
    values.append(values[0])
    angles = np.linspace(0, 2 * pi, len(node_names), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles, values, color="crimson", linewidth=2)
    ax.fill(angles, values, color="crimson", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(node_names)
    ax.set_title(f"Sample {index} Radar | Pred={record['pred_label']} True={record['true_label']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"radar_{index}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Node contribution radar plots.")
    parser.add_argument("--input", default="outputs/figures/explanations.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/figures/radar")
    args = parser.parse_args()
    records = load_records(Path(args.input))
    output_dir = Path(args.output_dir)
    for idx, record in enumerate(records[: args.limit]):
        plot_radar(record, output_dir, idx)


if __name__ == "__main__":
    main()
