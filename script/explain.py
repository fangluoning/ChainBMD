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


def run_explanations(
    split: str,
    samples: int,
    output_path: Path,
    filter_skill: Optional[int],
    model_name: Optional[str],
    checkpoint: Optional[Path],
    subject_split: bool,
    feature_subset: Optional[str],
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

    explanations = []
    collected = 0
    for batch in loader:
        sequence = batch["sequence"].to(device)
        label = batch["label"].item()
        if filter_skill is not None and label != filter_skill:
            continue
        if samples >= 0 and collected >= samples:
            break
        with torch.no_grad():
            logits, _, _ = model(sequence)
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        exps, preds = model.explain(sequence)
        record = {
            "true_label": label,
            "pred_label": preds[0].item(),
            "probabilities": probs,
            "node_contributions": exps[0]["node_contributions"],
            "time_importance": exps[0]["time_importance"],
            "node_time_series": exps[0]["node_time_series"],
            "node_components": exps[0].get("node_components", {}),
        }
        explanations.append(record)
        collected += 1
    if filter_skill is not None and collected == 0:
        raise ValueError(f"No samples found with skill level {filter_skill} in split '{split}'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(explanations)} explanations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate explanations for ChainBMD or LSTM models.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to use.")
    parser.add_argument("--samples", type=int, default=5, help="Number of sequences to explain.")
    parser.add_argument(
        "--output",
        default="outputs/figures/explanations.json",
        help="Path to save explanation JSON.",
    )
    parser.add_argument(
        "--filter-skill",
        type=int,
        default=None,
        help="Only export samples whose skill level equals this value.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to load. Options: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path.",
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
    run_explanations(
        args.split,
        args.samples,
        Path(args.output),
        args.filter_skill,
        args.model,
        Path(args.checkpoint) if args.checkpoint else None,
        args.subject_split,
        args.feature_subset,
    )


if __name__ == "__main__":
    main()
