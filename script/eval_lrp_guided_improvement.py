import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import available_models, build_model  # noqa: E402
from script.analyze_deviation import detect_deviation  # noqa: E402
from train.config import FEATURE_SUBSETS, get_config  # noqa: E402
from train.utils import build_dataloaders  # noqa: E402


def compute_expert_mean(
    hdf5_path: Path,
    target_field: str,
    skill_level: int,
    feature_indices: Optional[List[int]],
) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        labels = f[target_field][:]
        indices = np.where(labels == skill_level)[0]
        if indices.size == 0:
            raise ValueError(f"No samples found with skill_level={skill_level}")
        seq_len = f["feature_matrices"].shape[1]
        feature_dim = (
            len(feature_indices) if feature_indices is not None else f["feature_matrices"].shape[2]
        )
        total = np.zeros((seq_len, feature_dim), dtype=np.float64)
        count = 0
        for idx in indices:
            seq = f["feature_matrices"][idx]
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            if feature_indices is not None:
                seq = seq[:, feature_indices]
            total += seq
            count += 1
    return (total / max(count, 1)).astype(np.float32)


def extract_metrics(logits: torch.Tensor, target_idx: int) -> Tuple[float, float, float]:
    probs = torch.softmax(logits, dim=-1)
    target_logit = logits[0, target_idx].item()
    target_prob = probs[0, target_idx].item()
    mask = torch.ones_like(probs[0], dtype=torch.bool)
    mask[target_idx] = False
    other_max = probs[0][mask].max().item()
    margin = target_prob - other_max
    return target_logit, target_prob, margin


def optimize_sequence(
    model,
    base_seq: torch.Tensor,
    mask: torch.Tensor,
    target_idx: int,
    steps: int,
    lr: float,
    objective: str,
    patience: int,
    min_delta: float,
) -> torch.Tensor:
    delta = torch.zeros_like(base_seq, requires_grad=True)
    optimizer = optim.Adam([delta], lr=lr)
    best_loss = None
    bad_steps = 0
    for _ in range(steps):
        optimizer.zero_grad()
        seq = base_seq + delta * mask
        logits, _, _ = model(seq.unsqueeze(0))
        if objective == "logit":
            loss = -logits[0, target_idx]
        elif objective == "margin":
            probs = torch.softmax(logits, dim=-1)[0]
            target_prob = probs[target_idx]
            other_max = probs[torch.arange(probs.size(0)) != target_idx].max()
            loss = -(target_prob - other_max)
        else:
            probs = torch.softmax(logits, dim=-1)[0]
            loss = -probs[target_idx]
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss - min_delta:
            best_loss = loss_val
            bad_steps = 0
        else:
            bad_steps += 1
            if bad_steps >= patience:
                break
    return (base_seq + delta * mask).detach()


def load_benchmark(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_issue_nodes(report: Dict) -> List[str]:
    nodes = set()
    for item in report.get("chain_reports", []):
        nodes.update(item.get("nodes", {}).keys())
    for alert in report.get("component_alerts", []):
        node = alert.get("node")
        if node:
            nodes.add(node)
    for alert in report.get("timing_alerts", []):
        node = alert.get("node")
        if node:
            nodes.add(node)
    return sorted(nodes)


def build_deviation_mask_for_node(
    series: List[float],
    expert_mean: List[float],
    expert_std: List[float],
    z_threshold: float,
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0,), dtype=bool)
    mean = np.asarray(expert_mean, dtype=np.float32)
    std = np.asarray(expert_std, dtype=np.float32)
    if mean.shape[0] != values.shape[0]:
        raise ValueError("LRP series length mismatch with benchmark mean.")
    z = (values - mean) / (std + 1e-6)
    return np.abs(z) >= abs(z_threshold)


def run_evaluation(
    split: str,
    samples: int,
    output_path: Path,
    model_name: Optional[str],
    checkpoint: Optional[Path],
    subject_split: bool,
    feature_subset: Optional[str],
    target_skill: int,
    deviation_threshold: float,
    alphas: List[float],
    benchmark_path: Path,
    phase_threshold: float,
    only_non_target: bool,
    opt_steps: int,
    opt_lrs: List[float],
    opt_objective: str,
    opt_patience: int,
    opt_min_delta: float,
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

    benchmark = load_benchmark(benchmark_path)
    expert_mean = compute_expert_mean(
        cfg.data_path,
        cfg.target_field,
        target_skill,
        cfg.feature_indices,
    )
    expert_mean_tensor = torch.from_numpy(expert_mean).to(device)

    base = output_path.stem
    per_sample_path = output_path.parent / f"{base}_per_sample.csv"
    summary_path = output_path.parent / f"{base}_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    collected = 0
    for idx, batch in enumerate(loader):
        if samples >= 0 and collected >= samples:
            break
        sequence = batch["sequence"].to(device)
        label = batch["label"].item()
        with torch.no_grad():
            base_logits, _, _ = model(sequence)
        base_logit, base_prob, base_margin = extract_metrics(base_logits, target_skill)
        pred_before = int(base_logits.argmax(dim=-1)[0].item())
        if only_non_target and pred_before == target_skill:
            continue

        explanations, _ = model.explain(sequence)
        record = explanations[0]
        record["true_label"] = label
        record["pred_label"] = pred_before
        deviation_report = detect_deviation(
            record,
            benchmark,
            threshold=deviation_threshold,
            phase_threshold=phase_threshold,
        )
        issue_nodes = collect_issue_nodes(deviation_report)

        seq = sequence[0]
        modified_any = False
        mask = torch.zeros_like(seq)
        for alpha in alphas:
            mod_seq = seq.clone()
            if issue_nodes:
                for spec in model.node_specs:
                    if spec.name not in issue_nodes:
                        continue
                    series = record["node_time_series"].get(spec.name, [])
                    node_stats = benchmark["node_stats"].get(spec.name, {})
                    mean_curve = node_stats.get("mean", [])
                    std_curve = node_stats.get("std", [])
                    dev_mask = build_deviation_mask_for_node(
                        series,
                        mean_curve,
                        std_curve,
                        deviation_threshold,
                    )
                    if dev_mask.size == 0:
                        continue
                    dev_indices = np.where(dev_mask)[0]
                    if dev_indices.size == 0:
                        continue
                    dev_idx_tensor = torch.tensor(dev_indices, device=device, dtype=torch.long)
                    current = mod_seq[dev_idx_tensor, spec.start:spec.end]
                    target = expert_mean_tensor[dev_idx_tensor, spec.start:spec.end]
                    mod_seq[dev_idx_tensor, spec.start:spec.end] = current + alpha * (target - current)
                    modified_any = True
                    mask[dev_idx_tensor, spec.start:spec.end] = 1.0

            with torch.no_grad():
                mod_logits, _, _ = model(mod_seq.unsqueeze(0))
            mod_logit, mod_prob, mod_margin = extract_metrics(mod_logits, target_skill)
            pred_after = int(mod_logits.argmax(dim=-1)[0].item())

            opt_seq = mod_seq
            opt_lr_used = None
            if opt_steps > 0 and mask.sum().item() > 0 and opt_lrs:
                best_prob = -1.0
                best_seq = mod_seq
                for lr in opt_lrs:
                    cand_seq = optimize_sequence(
                        model,
                        mod_seq,
                        mask,
                        target_skill,
                        opt_steps,
                        lr,
                        opt_objective,
                        opt_patience,
                        opt_min_delta,
                    )
                    with torch.no_grad():
                        cand_logits, _, _ = model(cand_seq.unsqueeze(0))
                        cand_prob = torch.softmax(cand_logits, dim=-1)[0, target_skill].item()
                    if cand_prob > best_prob:
                        best_prob = cand_prob
                        best_seq = cand_seq
                        opt_lr_used = lr
                opt_seq = best_seq
            with torch.no_grad():
                opt_logits, _, _ = model(opt_seq.unsqueeze(0))
            opt_logit, opt_prob, opt_margin = extract_metrics(opt_logits, target_skill)
            pred_opt = int(opt_logits.argmax(dim=-1)[0].item())
            per_sample_rows.append(
                {
                    "sample_index": idx,
                    "true_label": label,
                    "pred_before": pred_before,
                    "pred_after": pred_after,
                    "pred_opt": pred_opt,
                    "target_skill": target_skill,
                    "deviation_threshold": deviation_threshold,
                    "alpha": alpha,
                    "issue_nodes": "|".join(issue_nodes),
                    "issue_node_count": len(issue_nodes),
                    "modified": int(modified_any),
                    "base_prob_target": base_prob,
                    "base_logit_target": base_logit,
                    "base_margin_target": base_margin,
                    "mod_prob_target": mod_prob,
                    "mod_logit_target": mod_logit,
                    "mod_margin_target": mod_margin,
                    "opt_prob_target": opt_prob,
                    "opt_logit_target": opt_logit,
                    "opt_margin_target": opt_margin,
                    "delta_prob": mod_prob - base_prob,
                    "delta_logit": mod_logit - base_logit,
                    "delta_margin": mod_margin - base_margin,
                    "delta_prob_opt": opt_prob - base_prob,
                    "delta_logit_opt": opt_logit - base_logit,
                    "delta_margin_opt": opt_margin - base_margin,
                    "pred_improved_to_target": int(pred_before != target_skill and pred_after == target_skill),
                    "pred_changed": int(pred_after != pred_before),
                    "pred_improved_to_target_opt": int(
                        pred_before != target_skill and pred_opt == target_skill
                    ),
                    "pred_changed_opt": int(pred_opt != pred_before),
                    "opt_steps": opt_steps,
                    "opt_lr": opt_lr_used if opt_lr_used is not None else 0.0,
                    "opt_objective": opt_objective,
                    "opt_patience": opt_patience,
                    "opt_min_delta": opt_min_delta,
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

    summary_rows = []
    for alpha in alphas:
        rows = [r for r in per_sample_rows if r["alpha"] == alpha]
        rows_non_target = [r for r in rows if r["pred_before"] != target_skill]
        summary = {
            "alpha": alpha,
            "num_samples": len(rows_non_target),
            "target_skill": target_skill,
            "deviation_threshold": deviation_threshold,
        }
        for key in (
            "delta_prob",
            "delta_logit",
            "delta_margin",
            "delta_prob_opt",
            "delta_logit_opt",
            "delta_margin_opt",
            "pred_improved_to_target",
            "pred_improved_to_target_opt",
            "pred_changed",
            "pred_changed_opt",
            "issue_node_count",
        ):
            values = [r[key] for r in rows_non_target]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            summary[f"{key}_std"] = float(np.std(values)) if values else 0.0
        summary_rows.append(summary)

    summary_fields = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-sample results to {per_sample_path}")
    print(f"Saved summary results to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LRP-guided improvement by modifying low-contribution regions."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=-1, help="Number of sequences to evaluate.")
    parser.add_argument(
        "--output",
        default="outputs/metrics/lrp_guided_improvement.csv",
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
    parser.add_argument("--target-skill", type=int, default=2)
    parser.add_argument("--deviation-threshold", type=float, default=1.0)
    parser.add_argument("--alphas", type=str, default="0.5,1.0")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="outputs/figures/benchmark_skill2.json",
        help="Benchmark JSON for deviation detection.",
    )
    parser.add_argument("--phase-threshold", type=float, default=8.0)
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include samples whose pred_before already equals target_skill.",
    )
    parser.add_argument("--opt-steps", type=int, default=40)
    parser.add_argument("--opt-lrs", type=str, default="0.03,0.05,0.08,0.1")
    parser.add_argument(
        "--opt-objective",
        type=str,
        default="prob",
        choices=("prob", "logit", "margin"),
    )
    parser.add_argument("--opt-patience", type=int, default=5)
    parser.add_argument("--opt-min-delta", type=float, default=1e-4)
    args = parser.parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("No alphas provided.")
    opt_lrs = [float(x.strip()) for x in args.opt_lrs.split(",") if x.strip()]
    run_evaluation(
        args.split,
        args.samples,
        Path(args.output),
        args.model,
        Path(args.checkpoint) if args.checkpoint else None,
        args.subject_split,
        args.feature_subset,
        args.target_skill,
        args.deviation_threshold,
        alphas,
        Path(args.benchmark),
        args.phase_threshold,
        not args.include_target,
        args.opt_steps,
        opt_lrs,
        args.opt_objective,
        args.opt_patience,
        args.opt_min_delta,
    )


if __name__ == "__main__":
    main()
