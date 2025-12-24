import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import available_models, build_model  # noqa: E402


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


if __package__ in (None, ""):
    config_module = _load_module("train_config", PROJECT_ROOT / "train" / "config.py")
    utils_module = _load_module("train_utils", PROJECT_ROOT / "train" / "utils.py")
    get_config = config_module.get_config
    FEATURE_SUBSETS = config_module.FEATURE_SUBSETS
    build_node_specs_for_subset = config_module.build_node_specs_for_subset
    build_dataloaders = utils_module.build_dataloaders
    build_kfold_dataloaders = utils_module.build_kfold_dataloaders
else:
    from train.config import FEATURE_SUBSETS, build_node_specs_for_subset, get_config  # type: ignore
    from train.utils import build_dataloaders, build_kfold_dataloaders  # type: ignore


def _parse_args():
    parser = argparse.ArgumentParser(description="Train ChainBMD.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to train (default from config). Options: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--subject-split",
        action="store_true",
        help="Enable subject-level splitting so the same player never appears in multiple splits.",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="Enable k-fold cross-validation (>1); overrides standard train/val/test flow.",
    )
    parser.add_argument(
        "--feature-subset",
        type=str,
        choices=tuple(FEATURE_SUBSETS.keys()),
        help="Keep only the specified node subset features for ablation.",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
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


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            logits, _, _ = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(total, 1)
    avg_acc = total_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": avg_acc}


def _train_single_split(
    cfg,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader=None,
    fold_suffix: Optional[str] = None,
) -> Dict[str, float]:
    model = build_model(cfg.model_name, cfg.model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_val_acc = float("-inf")
    best_state = None
    ckpt_path = cfg.checkpoint_path
    if fold_suffix:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_{fold_suffix}{ckpt_path.suffix}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    patience = getattr(cfg, "early_stopping_patience", 0) or 0
    epochs_without_improve = 0
    tag = fold_suffix or "main"
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)
        print(
            f"[{fold_suffix or 'main'}] Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}"
        )
        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if patience and epochs_without_improve >= patience:
                print(f"[{tag}] Early stopping triggered at epoch {epoch} (patience={patience}).")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    results = {"best_val_acc": best_val_acc}
    if test_loader is not None:
        test_stats = evaluate(model, test_loader, criterion, device)
        print(
            f"[{fold_suffix or 'main'}] Test set | loss={test_stats['loss']:.4f} acc={test_stats['accuracy']:.4f}"
        )
        results.update({"test_loss": test_stats["loss"], "test_acc": test_stats["accuracy"]})
    return results


def main():
    args = _parse_args()
    env_model = os.getenv("CHAINBMD_MODEL")
    selected_model = args.model or env_model
    cfg = get_config(model_name=selected_model)
    if args.subject_split:
        cfg.split_by_subject = True
    if args.kfold and args.kfold > 1:
        cfg.kfold_splits = args.kfold
    if args.feature_subset:
        cfg.feature_subset = args.feature_subset
        cfg.feature_indices = FEATURE_SUBSETS[cfg.feature_subset]
        cfg.raw_feature_dim = len(cfg.feature_indices)
    if cfg.feature_indices:
        if hasattr(cfg.model_config, "raw_feature_dim"):
            cfg.model_config.raw_feature_dim = len(cfg.feature_indices)
        if hasattr(cfg.model_config, "input_dim"):
            cfg.model_config.input_dim = len(cfg.feature_indices)
        if cfg.model_name in ("chainbmd", "chainbmd_model"):
            cfg.model_config.node_feature_specs = build_node_specs_for_subset(cfg.feature_indices)
    override_epochs = os.getenv("CHAINBMD_EPOCHS")
    if override_epochs is not None:
        cfg.epochs = int(override_epochs)
    device = torch.device(cfg.device)

    if cfg.kfold_splits and cfg.kfold_splits > 1:
        folds = build_kfold_dataloaders(
            hdf5_path=str(cfg.data_path),
            batch_size=cfg.batch_size,
            n_splits=cfg.kfold_splits,
            target_field=cfg.target_field,
            random_seed=cfg.random_seed,
            split_by_subject=cfg.split_by_subject,
            feature_indices=cfg.feature_indices,
        )
        fold_results: List[Dict[str, float]] = []
        for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
            print(f"\n===== Fold {fold_idx}/{cfg.kfold_splits} =====")
            result = _train_single_split(
                cfg,
                device,
                train_loader,
                val_loader,
                test_loader=None,
                fold_suffix=f"fold{fold_idx}",
            )
            fold_results.append(result)
        avg_val = sum(res["best_val_acc"] for res in fold_results) / len(fold_results)
        print(f"\nK-fold summary: avg best val acc = {avg_val:.4f}")
        return

    train_loader, val_loader, test_loader = build_dataloaders(
        hdf5_path=str(cfg.data_path),
        batch_size=cfg.batch_size,
        val_split=cfg.subject_val_split if cfg.split_by_subject else cfg.val_split,
        test_split=cfg.subject_test_split if cfg.split_by_subject else cfg.test_split,
        target_field=cfg.target_field,
        random_seed=cfg.random_seed,
        split_by_subject=cfg.split_by_subject,
        feature_indices=cfg.feature_indices,
    )
    _ = _train_single_split(cfg, device, train_loader, val_loader, test_loader, fold_suffix=None)


if __name__ == "__main__":
    main()
