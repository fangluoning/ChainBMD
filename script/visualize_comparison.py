import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(records, group_key="true_label"):
    groups = defaultdict(list)
    for rec in records:
        groups[rec[group_key]].append(rec)
    return groups


def plot_group_average(records, output_path: Path):
    if not records:
        return
    node_names = list(records[0]["node_time_series"].keys())
    length = len(records[0]["time_importance"])
    x = np.linspace(0, 100, length)
    fig, axes = plt.subplots(len(node_names), 1, figsize=(8, 2 * len(node_names)), sharex=True)
    if len(node_names) == 1:
        axes = [axes]
    for idx, name in enumerate(node_names):
        series = np.array([rec["node_time_series"][name] for rec in records])
        mean = series.mean(axis=0)
        std = series.std(axis=0)
        band = np.where(std == 0, 1e-4, std)
        axes[idx].plot(x, mean, color="navy")
        axes[idx].fill_between(x, mean - band, mean + band, color="navy", alpha=0.2)
        axes[idx].set_ylabel(name)
    axes[-1].set_xlabel("Phase (%)")
    fig.suptitle(
        f"Group {records[0]['true_label']} ({len(records)} samples)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare LRP means across groups.")
    parser.add_argument("--input", default="outputs/figures/explanations.json")
    parser.add_argument("--group", default="true_label", choices=["true_label", "pred_label"])
    parser.add_argument("--output_dir", default="outputs/figures/group_compare")
    args = parser.parse_args()
    records = load_records(Path(args.input))
    groups = aggregate(records, group_key=args.group)
    output_dir = Path(args.output_dir)
    for key, recs in groups.items():
        plot_group_average(recs, output_dir / f"group_{key}.png")


if __name__ == "__main__":
    main()
