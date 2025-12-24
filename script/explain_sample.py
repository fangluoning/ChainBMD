"""
CLI: load a checkpoint and run inference + LRP for a single sample.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.factory import available_models, build_model  # noqa: E402
from train.config import FEATURE_SUBSETS, get_config  # noqa: E402
from train.utils import build_dataloaders  # noqa: E402


def explain_single(
    split: str,
    index: int,
    output: Path = None,
    model_name: str = None,
    checkpoint: Path = None,
    subject_split: bool = False,
    feature_subset: Optional[str] = None,
):
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
    loader = loaders[split]

    device = torch.device(cfg.device)
    checkpoint_path = checkpoint or cfg.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    model = build_model(cfg.model_name, cfg.model_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    batch = None
    for idx, sample in enumerate(loader):
        if idx == index:
            batch = sample
            break
    if batch is None:
        raise IndexError(f"No sample index={index} in {split} split.")

    seq = batch["sequence"].to(device)
    logits, _, _ = model(seq)
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    explanations, preds = model.explain(seq)
    record = {
        "split": split,
        "index": index,
        "true_label": batch["label"].item(),
        "pred_label": preds[0].item(),
        "probabilities": probs,
        "node_contributions": explanations[0]["node_contributions"],
        "time_importance": explanations[0]["time_importance"],
        "node_time_series": explanations[0]["node_time_series"],
        "node_components": explanations[0].get("node_components", {}),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"Saved explanation to {output}")
    else:
        print(json.dumps(record, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Explain a single sample.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="Sample index within the split.")
    parser.add_argument("--output", type=str, help="Optional: output JSON path.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name (default from config). Options: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: override checkpoint path.",
    )
    parser.add_argument(
        "--subject-split",
        action="store_true",
        help="Match training: split by subject.",
    )
    parser.add_argument(
        "--feature-subset",
        type=str,
        choices=tuple(FEATURE_SUBSETS.keys()),
        help="Explain only the specified node feature subset.",
    )
    args = parser.parse_args()
    explain_single(
        args.split,
        args.index,
        Path(args.output) if args.output else None,
        model_name=args.model,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        subject_split=args.subject_split,
        feature_subset=args.feature_subset,
    )


if __name__ == "__main__":
    main()
