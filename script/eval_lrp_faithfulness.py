import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import available_models, build_model  # noqa: E402
from train.config import FEATURE_SUBSETS, get_config  # noqa: E402
from train.utils import build_dataloaders  # noqa: E402


def build_importance_matrix(
    node_time_series: Dict[str, List[float]],
    sequence: torch.Tensor,
    node_specs,
) -> torch.Tensor:
    seq_len, feat_dim = sequence.shape
    importance = torch.zeros((seq_len, feat_dim), device=sequence.device)
    for spec in node_specs:
        series = node_time_series.get(spec.name)
        if series is None:
            continue
        series_tensor = torch.tensor(series, device=sequence.device, dtype=sequence.dtype)
        if series_tensor.numel() != seq_len:
            raise ValueError(f"Node {spec.name} series length {series_tensor.numel()} != {seq_len}")
        importance[:, spec.start:spec.end] = series_tensor.unsqueeze(1).expand(-1, spec.end - spec.start)
    return importance


def extract_metrics(logits: torch.Tensor, target_idx: int) -> Tuple[float, float, float]:
    probs = torch.softmax(logits, dim=-1)
    target_logit = logits[0, target_idx].item()
    target_prob = probs[0, target_idx].item()
    mask = torch.ones_like(probs[0], dtype=torch.bool)
    mask[target_idx] = False
    other_max = probs[0][mask].max().item()
    margin = target_prob - other_max
    return target_logit, target_prob, margin


def auc_trapz(fractions: List[float], values: List[float]) -> float:
    x = torch.tensor(fractions, dtype=torch.float32)
    y = torch.tensor(values, dtype=torch.float32)
    return torch.trapz(y, x).item()


def perturb_and_trace(
    model,
    sequence: torch.Tensor,
    baseline: torch.Tensor,
    order: torch.Tensor,
    fractions: List[float],
    target_idx: int,
) -> Dict[str, List[float]]:
    total = order.numel()
    original_flat = sequence.flatten()
    baseline_flat = baseline.flatten()
    deletion_flat = original_flat.clone()
    insertion_flat = baseline_flat.clone()
    del_logits, del_probs, del_margins = [], [], []
    ins_logits, ins_probs, ins_margins = [], [], []
    prev_k = 0
    for frac in fractions:
        k = int(round(frac * total))
        if k > prev_k:
            idx = order[prev_k:k]
            deletion_flat[idx] = baseline_flat[idx]
            insertion_flat[idx] = original_flat[idx]
            prev_k = k
        del_seq = deletion_flat.view_as(sequence).unsqueeze(0)
        ins_seq = insertion_flat.view_as(sequence).unsqueeze(0)
        with torch.no_grad():
            del_logits_raw, _, _ = model(del_seq)
            ins_logits_raw, _, _ = model(ins_seq)
        d_logit, d_prob, d_margin = extract_metrics(del_logits_raw, target_idx)
        i_logit, i_prob, i_margin = extract_metrics(ins_logits_raw, target_idx)
        del_logits.append(d_logit)
        del_probs.append(d_prob)
        del_margins.append(d_margin)
        ins_logits.append(i_logit)
        ins_probs.append(i_prob)
        ins_margins.append(i_margin)
    return {
        "del_logit": del_logits,
        "del_prob": del_probs,
        "del_margin": del_margins,
        "ins_logit": ins_logits,
        "ins_prob": ins_probs,
        "ins_margin": ins_margins,
    }


def run_evaluation(
    split: str,
    samples: int,
    output_path: Path,
    model_name: Optional[str],
    checkpoint: Optional[Path],
    subject_split: bool,
    feature_subset: Optional[str],
    steps: int,
    baseline_mode: str,
) -> None:
    cfg = get_config(model_name=model_name)
    if subject_split:
        cfg.split_by_subject = True
    if feature_subset:
        cfg.feature_subset = feature_subset
        cfg.feature_indices = FEATURE_SUBSETS[feature_subset]
        cfg.raw_feature_dim = len(cfg.feature_indices)
    if cfg.feature_indices:
        if hasattr(cfg.model_config, "raw_feature_dim"):
            cfg.model_config.raw_feature_dim = len(cfg.feature_indices)
        if hasattr(cfg.model_config, "input_dim"):
            cfg.model_config.input_dim = len(cfg.feature_indices)
    train_loader, val_loader, test_loader = build_dataloaders(
        hdf5_path=str(cfg.data_path),
        batch_size=1,
        val_split=cfg.subject_val_split if cfg.split_by_subject else cfg.val_split,
        test_split=cfg.subject_test_split if cfg.split_by_subject else cfg.test_split,
        target_field=cfg.target_field,
        random_seed=cfg.random_seed,
        split_by_subject=cfg.split_by_subject,
        feature_indices=cfg.feature_indices,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}', choose from train/val/test")
    loader = loaders[split]
    device = torch.device(cfg.device)
    checkpoint_path = checkpoint or cfg.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found, train the model first.")
    model = build_model(cfg.model_name, cfg.model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    fractions = [round(i / (steps - 1), 6) for i in range(steps)]
    base = output_path.stem
    summary_path = output_path.parent / f"{base}_summary.csv"
    per_sample_path = output_path.parent / f"{base}_per_sample.csv"
    curves_path = output_path.parent / f"{base}_curves.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    curve_rows = []
    collected = 0
    for idx, batch in enumerate(loader):
        if samples >= 0 and collected >= samples:
            break
        sequence = batch["sequence"].to(device)
        label = batch["label"].item()
        with torch.no_grad():
            base_logits, _, _ = model(sequence)
            base_probs = torch.softmax(base_logits, dim=-1)[0].cpu().tolist()
        explanations, pred_tensor = model.explain(sequence)
        pred_label = pred_tensor[0].item()
        node_time_series = explanations[0]["node_time_series"]
        seq = sequence[0]
        importance = build_importance_matrix(node_time_series, seq, model.node_specs)
        order = torch.argsort(importance.flatten(), descending=True)
        if baseline_mode == "zero":
            baseline = torch.zeros_like(seq)
        elif baseline_mode == "mean":
            mean_feat = seq.mean(dim=0, keepdim=True)
            baseline = mean_feat.repeat(seq.size(0), 1)
        else:
            raise ValueError(f"Unsupported baseline '{baseline_mode}'")

        for target_mode, target_idx in (("true", label),):
            base_logit, base_prob, base_margin = extract_metrics(base_logits, target_idx)
            traces = perturb_and_trace(
                model,
                seq,
                baseline,
                order,
                fractions,
                target_idx,
            )
            del_auc_prob = auc_trapz(fractions, traces["del_prob"])
            ins_auc_prob = auc_trapz(fractions, traces["ins_prob"])
            del_auc_logit = auc_trapz(fractions, traces["del_logit"])
            ins_auc_logit = auc_trapz(fractions, traces["ins_logit"])
            del_auc_margin = auc_trapz(fractions, traces["del_margin"])
            ins_auc_margin = auc_trapz(fractions, traces["ins_margin"])

            p20_index = min(range(len(fractions)), key=lambda i: abs(fractions[i] - 0.2))
            del_delta_prob_p20 = base_prob - traces["del_prob"][p20_index]
            del_delta_logit_p20 = base_logit - traces["del_logit"][p20_index]
            del_delta_margin_p20 = base_margin - traces["del_margin"][p20_index]

            per_sample_rows.append(
                {
                    "sample_index": idx,
                    "target_mode": target_mode,
                    "true_label": label,
                    "pred_label": pred_label,
                    "base_prob_target": base_prob,
                    "base_logit_target": base_logit,
                    "base_margin_target": base_margin,
                    "del_auc_prob": del_auc_prob,
                    "ins_auc_prob": ins_auc_prob,
                    "del_auc_logit": del_auc_logit,
                    "ins_auc_logit": ins_auc_logit,
                    "del_auc_margin": del_auc_margin,
                    "ins_auc_margin": ins_auc_margin,
                    "del_delta_prob_p20": del_delta_prob_p20,
                    "del_delta_logit_p20": del_delta_logit_p20,
                    "del_delta_margin_p20": del_delta_margin_p20,
                    "baseline_mode": baseline_mode,
                }
            )

            for metric_key, label_key in (
                ("del_prob", "deletion_prob"),
                ("ins_prob", "insertion_prob"),
                ("del_logit", "deletion_logit"),
                ("ins_logit", "insertion_logit"),
                ("del_margin", "deletion_margin"),
                ("ins_margin", "insertion_margin"),
            ):
                for frac, value in zip(fractions, traces[metric_key]):
                    curve_rows.append(
                        {
                            "sample_index": idx,
                            "target_mode": target_mode,
                            "metric": label_key,
                            "fraction": frac,
                            "value": value,
                            "baseline_mode": baseline_mode,
                        }
                    )

        collected += 1

    if not per_sample_rows:
        raise ValueError("No samples processed; check split or filters.")

    fieldnames = list(per_sample_rows[0].keys())
    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    curve_fields = list(curve_rows[0].keys())
    with open(curves_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    summary_rows = []
    for target_mode in ("true",):
        rows = [r for r in per_sample_rows if r["target_mode"] == target_mode]
        summary = {"target_mode": target_mode, "num_samples": len(rows), "baseline_mode": baseline_mode}
        for key in fieldnames:
            if key in ("sample_index", "target_mode", "true_label", "pred_label", "baseline_mode"):
                continue
            values = [r[key] for r in rows]
            summary[f"{key}_mean"] = float(torch.tensor(values).mean().item())
            summary[f"{key}_std"] = float(torch.tensor(values).std(unbiased=False).item())
        summary_rows.append(summary)

    summary_fields = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-sample metrics to {per_sample_path}")
    print(f"Saved curve data to {curves_path}")
    print(f"Saved summary metrics to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LRP faithfulness via deletion/insertion on ChainBMD."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=-1, help="Number of sequences to evaluate.")
    parser.add_argument(
        "--output",
        default="outputs/metrics/lrp_faithfulness.csv",
        help="Base path for CSV outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to load. Options: {', '.join(available_models())}",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--subject-split", action="store_true")
    parser.add_argument(
        "--feature-subset",
        type=str,
        choices=tuple(FEATURE_SUBSETS.keys()),
        help="Evaluate only a feature subset.",
    )
    parser.add_argument("--steps", type=int, default=11, help="Number of fractions in curves.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="zero",
        choices=("zero", "mean"),
        help="Baseline for deletion/insertion.",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Plot mean + IQR curves from the curves CSV.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="outputs/metrics/curves",
        help="Directory to save curve plots.",
    )
    parser.add_argument(
        "--plot-target",
        choices=("true",),
        default="true",
        help="Target mode for single-plot rendering when --plot-all is not set.",
    )
    parser.add_argument(
        "--plot-metric",
        choices=(
            "deletion_prob",
            "insertion_prob",
            "deletion_logit",
            "insertion_logit",
            "deletion_margin",
            "insertion_margin",
        ),
        default="deletion_prob",
        help="Metric for single-plot rendering when --plot-all is not set.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot all target/metric combinations.",
    )
    args = parser.parse_args()
    run_evaluation(
        args.split,
        args.samples,
        Path(args.output),
        args.model,
        Path(args.checkpoint) if args.checkpoint else None,
        args.subject_split,
        args.feature_subset,
        args.steps,
        args.baseline,
    )
    if args.plot_curves:
        curves_path = Path(args.output).parent / f"{Path(args.output).stem}_curves.csv"
        plot_curves_from_csv(
            curves_path,
            Path(args.plot_output_dir),
            args.plot_target,
            args.plot_metric,
            args.plot_all,
        )


def plot_curves_from_csv(
    curves_path: Path,
    output_dir: Path,
    target_mode: str,
    metric: str,
    plot_all: bool,
) -> None:
    df = pd.read_csv(curves_path)
    if df.empty:
        raise ValueError(f"No curve data found in {curves_path}")
    df["target_mode"] = df["target_mode"].astype(str).str.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_all:
        for path in output_dir.glob("*_pred.png"):
            path.unlink(missing_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
        }
    )
    metrics = (
        "deletion_prob",
        "insertion_prob",
        "deletion_logit",
        "insertion_logit",
        "deletion_margin",
        "insertion_margin",
    )
    targets = ("true",)
    plot_tasks = []
    if plot_all:
        for t in targets:
            for m in metrics:
                plot_tasks.append((t, m))
    else:
        plot_tasks.append((target_mode, metric))

    for t_mode, m_key in plot_tasks:
        sub = df[(df["target_mode"] == t_mode) & (df["metric"] == m_key)].copy()
        if sub.empty:
            continue
        grouped = sub.groupby("fraction")["value"]
        stats = grouped.agg(["mean", "std", "count"])
        x = stats.index.to_numpy()
        y = stats["mean"].to_numpy()
        sem = stats["std"].to_numpy() / np.sqrt(stats["count"].to_numpy().clip(min=1))
        ci = 1.96 * sem
        y_low = y - ci
        y_high = y + ci
        if "prob" in m_key:
            y_low = np.clip(y_low, 0.0, 1.0)
            y_high = np.clip(y_high, 0.0, 1.0)
        if "margin" in m_key:
            y_low = np.clip(y_low, -1.0, 1.0)
            y_high = np.clip(y_high, -1.0, 1.0)
        color = "#1f77b4" if "insertion" in m_key else "#ff7f0e"
        plt.figure(figsize=(4.2, 3.0))
        plt.plot(x, y, color=color, label="Mean")
        plt.fill_between(x, y_low, y_high, color=color, alpha=0.2, label="95% CI")
        plt.xlabel("Fraction of features perturbed")
        plt.ylabel(m_key.replace("_", " ").title())
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        plt.grid(True, axis="y", alpha=0.2, linewidth=0.6)
        plt.legend(frameon=False)
        plt.tight_layout()
        out_path = output_dir / f"{m_key}_{t_mode}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
