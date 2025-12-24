from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from models.chainbmd_model import ChainBMDConfig, DEFAULT_NODE_SPECS, NodeFeatureSpec


DEFAULT_CHECKPOINT = Path("outputs/checkpoints/chainbmd_best.pt")
FEATURE_SUBSETS: Dict[str, List[int]] = {
    "node13": [0, 1, 2, 6, 7, 8, 9],
    "node123": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "node1234": list(range(0, 19)),
    "node12345": list(range(0, 27)),
    "node123456": list(range(0, 35)),
    "node1234567": list(range(0, 38)),
    "all": list(range(0, 38)),
}


@dataclass
class TrainingConfig:
    data_path: Path = Path("data_processed/data_processed_allStreams_60hz_onlyForehand_skill_level.hdf5")
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    epochs: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    target_field: str = "skill_levels"
    raw_feature_dim: int = 38
    seq_len: int = 150
    log_interval: int = 50
    random_seed: int = 42
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    model_name: str = "chainbmd"
    model_config: Optional[ChainBMDConfig] = None
    split_by_subject: bool = False
    early_stopping_patience: int = 20
    subject_val_split: float = 0.2
    subject_test_split: float = 0.1
    kfold_splits: int = 0
    feature_subset: Optional[str] = None
    feature_indices: Optional[List[int]] = None


def get_config(model_name: Optional[str] = None) -> TrainingConfig:
    cfg = TrainingConfig()
    if model_name is not None:
        cfg.model_name = model_name
    cfg.model_name = cfg.model_name.lower()
    if cfg.model_name in ("chainbmd", "chainbmd_model"):
        if not isinstance(cfg.model_config, ChainBMDConfig):
            cfg.model_config = ChainBMDConfig(
                in_features=16,
                gcn_hidden=64,
                gcn_layers=3,
                transformer_dim=128,
                transformer_heads=4,
                transformer_layers=3,
                mlp_hidden=64,
                num_classes=3,
                dropout=0.2,
                raw_feature_dim=cfg.raw_feature_dim,
            )
        if cfg.checkpoint_path == DEFAULT_CHECKPOINT:
            cfg.checkpoint_path = Path("outputs/checkpoints/chainbmd_best.pt")
    else:
        raise ValueError(f"Unsupported model_name '{cfg.model_name}'.")

    if cfg.feature_subset:
        subset_key = cfg.feature_subset.lower()
        if subset_key not in FEATURE_SUBSETS:
            raise ValueError(
                f"Unknown feature_subset '{cfg.feature_subset}'. "
                f"Available: {', '.join(FEATURE_SUBSETS.keys())}"
            )
        cfg.feature_indices = FEATURE_SUBSETS[subset_key]
        cfg.raw_feature_dim = len(cfg.feature_indices)
        if hasattr(cfg.model_config, "raw_feature_dim"):
            cfg.model_config.raw_feature_dim = len(cfg.feature_indices)
        if hasattr(cfg.model_config, "input_dim"):
            cfg.model_config.input_dim = len(cfg.feature_indices)
        if isinstance(cfg.model_config, ChainBMDConfig):
            cfg.model_config.node_feature_specs = build_node_specs_for_subset(cfg.feature_indices)
    else:
        cfg.feature_indices = None
    return cfg


def build_node_specs_for_subset(feature_indices: List[int]) -> List[NodeFeatureSpec]:
    """Rebuild node specs for a subset to keep tokenizer input dimensions aligned."""
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(feature_indices)}
    subset_specs: List[NodeFeatureSpec] = []
    for spec in DEFAULT_NODE_SPECS:
        original_range = range(spec.start, spec.end)
        if all(idx in index_map for idx in original_range):
            new_start = index_map[spec.start]
            components = None
            if spec.components:
                components = [(name, rel_start, rel_end) for name, rel_start, rel_end in spec.components]
            subset_specs.append(
                NodeFeatureSpec(
                    name=spec.name,
                    start=new_start,
                    end=new_start + spec.dim,
                    components=components,
                )
            )
    if not subset_specs:
        raise ValueError(
            "The selected feature subset removed all node definitions. Please choose a valid key."
        )
    return subset_specs
