#!/usr/bin/env python3
"""Automate ChainBMD module ablations with consistent splits/metrics."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import build_model  # noqa: E402
from train.config import (
    FEATURE_SUBSETS,
    TrainingConfig,
    build_node_specs_for_subset,
    get_config,
)  # noqa: E402
from train.utils import build_dataloaders  # noqa: E402


ABLATION_VARIANTS: Dict[str, Dict[str, Dict[str, object]]] = {
    "full": {
        "description": "Default ChainBMD (GCN + Transformer + CLS + LRP)",
        "model": {},
    },
    "no_gcn": {
        "description": "Disable chained GCN (tokenizer + mean pooling only)",
        "model": {"use_gcn": False},
    },
    "no_transformer": {
        "description": "Remove Transformer (node mean + MLP only)",
        "model": {
            "use_transformer": False,
            "use_positional_encoding": False,
            "use_cls_token": False,
        },
    },
    "no_pos_cls": {
        "description": "Keep Transformer but drop positional encoding and CLS token",
        "model": {
            "use_positional_encoding": False,
            "use_cls_token": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChainBMD ablation experiments.")
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_gcn,no_transformer,no_pos_cls",
        help="Comma separated variant keys to run.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per variant.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--subject-split",
        action="store_true",
        help="Match the training script's subject-level split.",
    )
    parser.add_argument(
        "--feature-subset",
        type=str,
        choices=tuple(FEATURE_SUBSETS.keys()),
        help="Run a single node subset (legacy flag).",
    )
    parser.add_argument(
        "--feature-subsets",
        type=str,
        help="Comma-separated node subset keys (see README 3.2); runs all and writes one JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu/cuda), default follows config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/logs/chainbmd_ablation.json"),
        help="Path to save the JSON results.",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _, _ = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / max(total, 1)
    avg_acc = total_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": avg_acc}


def evaluate_metrics(
    model: nn.Module,
    dataloader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            logits, _, _ = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro"),
    }
    return metrics


def apply_variant_overrides(cfg: TrainingConfig, overrides: Dict[str, object]) -> TrainingConfig:
    new_cfg = copy.deepcopy(cfg)
    model_cfg = copy.deepcopy(new_cfg.model_config)
    for key, value in overrides.items():
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
        elif hasattr(new_cfg, key):
            setattr(new_cfg, key, value)
        else:
            raise AttributeError(f"Unknown override '{key}' for ChainBMD config.")
    new_cfg.model_config = model_cfg
    return new_cfg


def run_variant(
    name: str,
    cfg: TrainingConfig,
    loaders: Tuple,
    device: torch.device,
) -> Dict[str, object]:
    train_loader, val_loader, test_loader = loaders
    model = build_model(cfg.model_name, cfg.model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_state = None
    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_stats = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_metrics(model, val_loader, device, criterion)
        elapsed = time.time() - epoch_start
        print(
            f"[{name}] Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"time={elapsed:.1f}s"
        )
        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate_metrics(model, val_loader, device, criterion)
    test_metrics = evaluate_metrics(model, test_loader, device, criterion)
    print(
        f"[{name}] Test | acc={test_metrics['accuracy']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )
    return {
        "variant": name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _resolve_feature_subsets(args: argparse.Namespace) -> List[str]:
    subsets: List[str] = []
    if args.feature_subsets:
        subsets.extend([s.strip().lower() for s in args.feature_subsets.split(",") if s.strip()])
    if args.feature_subset:
        subsets.append(args.feature_subset.lower())
    if not subsets:
        subsets.append("all")
    normalized: List[str] = []
    for key in subsets:
        if key != "all" and key not in FEATURE_SUBSETS:
            raise ValueError(
                f"Unknown feature subset '{key}'. Available: all, {', '.join(FEATURE_SUBSETS.keys())}"
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def _configure_subset(cfg: TrainingConfig, subset_key: str) -> TrainingConfig:
    subset_cfg = copy.deepcopy(cfg)
    if subset_key == "all":
        subset_cfg.feature_subset = None
        subset_cfg.feature_indices = None
        if hasattr(subset_cfg.model_config, "raw_feature_dim"):
            subset_cfg.model_config.raw_feature_dim = subset_cfg.raw_feature_dim
        if hasattr(subset_cfg.model_config, "input_dim"):
            subset_cfg.model_config.input_dim = subset_cfg.raw_feature_dim
        return subset_cfg
    indices = FEATURE_SUBSETS[subset_key]
    subset_cfg.feature_subset = subset_key
    subset_cfg.feature_indices = indices
    subset_cfg.raw_feature_dim = len(indices)
    if hasattr(subset_cfg.model_config, "raw_feature_dim"):
        subset_cfg.model_config.raw_feature_dim = len(indices)
    if hasattr(subset_cfg.model_config, "input_dim"):
        subset_cfg.model_config.input_dim = len(indices)
    if hasattr(subset_cfg.model_config, "node_feature_specs"):
        subset_cfg.model_config.node_feature_specs = build_node_specs_for_subset(indices)
    return subset_cfg


def main() -> None:
    args = parse_args()
    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for key in requested_variants:
        if key not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown variant '{key}'. Available: {', '.join(ABLATION_VARIANTS)}")
    feature_subset_keys = _resolve_feature_subsets(args)

    base_cfg_template = get_config(model_name="chainbmd")
    base_cfg_template.batch_size = args.batch_size
    base_cfg_template.epochs = args.epochs
    base_cfg_template.split_by_subject = args.subject_split
    device = torch.device(args.device or base_cfg_template.device)
    base_cfg_template.device = str(device)
    torch.manual_seed(base_cfg_template.random_seed)

    all_results: List[Dict[str, object]] = []
    for subset_key in feature_subset_keys:
        subset_cfg = _configure_subset(base_cfg_template, subset_key)
        print(f"\n===== Feature subset: {subset_key} =====")
        loaders = build_dataloaders(
            hdf5_path=str(subset_cfg.data_path),
            batch_size=subset_cfg.batch_size,
            val_split=subset_cfg.subject_val_split if subset_cfg.split_by_subject else subset_cfg.val_split,
            test_split=subset_cfg.subject_test_split if subset_cfg.split_by_subject else subset_cfg.test_split,
            target_field=subset_cfg.target_field,
            random_seed=subset_cfg.random_seed,
            split_by_subject=subset_cfg.split_by_subject,
            feature_indices=subset_cfg.feature_indices,
        )
        for variant in requested_variants:
            overrides = ABLATION_VARIANTS[variant]
            desc = overrides.get("description", "")
            overrides_cfg = overrides.get("model", {})
            print(f"\n----- Running variant '{variant}' ({desc}) -----")
            variant_cfg = apply_variant_overrides(subset_cfg, overrides_cfg)
            result = run_variant(variant, variant_cfg, loaders, device)
            result["description"] = desc
            result["feature_subset"] = subset_key
            all_results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAblation summary saved to {args.output.resolve()}")
    for res in all_results:
        metrics = res["test_metrics"]
        print(
            f"[{res.get('feature_subset', 'all')}] {res['variant']}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
