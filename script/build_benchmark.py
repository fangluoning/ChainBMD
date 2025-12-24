"""
Build a node-time LRP benchmark (skill_level=2) from explanations.json.
Outputs a JSON with per-node mean and standard deviation curves.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compute_phase_metrics(curve: np.ndarray):
    peak_idx = int(curve.argmax())
    peak_phase = peak_idx / max(len(curve) - 1, 1) * 100
    cumulative = np.cumsum(curve)
    total = cumulative[-1] + 1e-6
    reach50 = np.argmax(cumulative >= 0.5 * total)
    rise_phase = reach50 / max(len(curve) - 1, 1) * 100
    return peak_phase, rise_phase


def build_benchmark(records, target_skill=2):
    high_records = [rec for rec in records if rec["true_label"] == target_skill]
    if not high_records:
        raise ValueError(
            f"No samples found for skill_level={target_skill}. Collect expert explanations first."
        )
    node_names = list(high_records[0]["node_time_series"].keys())
    node_stats = {}
    component_stats = {}
    for name in node_names:
        series = np.array([np.array(rec["node_time_series"][name]) for rec in high_records])
        peak_phase, rise_phase = compute_phase_metrics(series.mean(axis=0))
        median = np.median(series, axis=0)
        p10 = np.percentile(series, 10, axis=0)
        p90 = np.percentile(series, 90, axis=0)
        node_stats[name] = {
            "mean": series.mean(axis=0).tolist(),
            "std": series.std(axis=0).tolist(),
            "median": median.tolist(),
            "p10": p10.tolist(),
            "p90": p90.tolist(),
            "peak_phase": peak_phase,
            "rise_phase": rise_phase,
        }
        comp_map = high_records[0].get("node_components", {}).get(name, {})
        comp_stats = {}
        for comp_name in comp_map:
            comp_series = np.array(
                [np.array(rec["node_components"][name][comp_name]) for rec in high_records]
            )
            comp_stats[comp_name] = {
                "mean": comp_series.mean(axis=0).tolist(),
                "std": comp_series.std(axis=0).tolist(),
                "median": np.median(comp_series, axis=0).tolist(),
                "p10": np.percentile(comp_series, 10, axis=0).tolist(),
                "p90": np.percentile(comp_series, 90, axis=0).tolist(),
            }
        if comp_stats:
            component_stats[name] = comp_stats
    return {
        "skill_level": target_skill,
        "sample_count": len(high_records),
        "node_stats": node_stats,
        "component_stats": component_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Build expert LRP benchmark.")
    parser.add_argument("--input", default="outputs/figures/explanations.json")
    parser.add_argument("--output", default="outputs/figures/benchmark_skill2.json")
    parser.add_argument("--skill_level", type=int, default=2)
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    benchmark = build_benchmark(records, target_skill=args.skill_level)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark to {args.output} (n={benchmark['sample_count']})")


if __name__ == "__main__":
    main()
