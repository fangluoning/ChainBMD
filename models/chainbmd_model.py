"""ChainBMD model definition.

This module combines:
1. Graph Convolutional layers that operate on a predefined 7-node chain graph.
2. A temporal Transformer encoder that captures sequence dependencies.
3. An MLP classification head that predicts technical proficiency levels.
4. A lightweight Layer-wise Relevance Propagation (LRP) style analyser that
   attributes the final decision back to each node.

The current implementation uses placeholder/random data helpers so the module can
be smoke-tested before the real dataset is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lrp import (
    linear_lrp,
    classifier_lrp,
    transformer_lrp,
)


# Predefined 7 chain nodes, ordered from feet to hand.
NODE_NAMES: List[str] = [
    "left_foot",
    "right_foot",
    "left_leg",
    "right_leg",
    "trunk_shoulder",
    "upper_arm",
    "forearm_wrist",
]


def build_chain_adjacency(num_nodes: int) -> torch.Tensor:
    """Build adjacency for an undirected chain graph."""
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for idx in range(num_nodes - 1):
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0
    return adjacency


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization for stable GCN training."""
    adjacency = adjacency + torch.eye(adjacency.size(0))
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_mat = torch.diag(degree_inv_sqrt)
    return degree_mat @ adjacency @ degree_mat


class GraphConvolution(nn.Module):
    """Simple dense-graph GCN layer with a single linear projection."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, nodes, in_features)
            adjacency: Normalized adjacency of shape (nodes, nodes)
        """
        support = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.linear(support)


class SinePositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding with learnable scale."""

    def __init__(self, dim: int, max_len: int = 500) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.scale * self.pe[:, :seq_len]


@dataclass
class NodeFeatureSpec:
    name: str
    start: int
    end: int
    components: Optional[List[Tuple[str, int, int]]] = None

    @property
    def dim(self) -> int:
        return self.end - self.start

    def component_ranges(self) -> List[Tuple[str, int, int]]:
        if self.components:
            return [
                (comp_name, self.start + rel_start, self.start + rel_end)
                for comp_name, rel_start, rel_end in self.components
            ]
        ranges = []
        for idx in range(self.dim):
            ranges.append(
                (f"{self.name}_dim{idx}", self.start + idx, self.start + idx + 1)
            )
        return ranges


DEFAULT_NODE_SPECS: List[NodeFeatureSpec] = [
    NodeFeatureSpec(
        "left_foot",
        0,
        3,
        components=[
            ("left_total_force", 0, 1),
            ("left_forefoot", 1, 2),
            ("left_heel", 2, 3),
        ],
    ),
    NodeFeatureSpec(
        "right_foot",
        3,
        6,
        components=[
            ("right_total_force", 0, 1),
            ("right_forefoot", 1, 2),
            ("right_heel", 2, 3),
        ],
    ),
    NodeFeatureSpec(
        "left_leg_emg",
        6,
        10,
        components=[(f"left_leg_emg_c{i+1}", i, i + 1) for i in range(4)],
    ),
    NodeFeatureSpec(
        "core_spine",
        10,
        19,
        components=[
            ("hip_roll", 0, 1),
            ("hip_pitch", 1, 2),
            ("hip_yaw", 2, 3),
            ("spine_roll", 3, 4),
            ("spine_pitch", 4, 5),
            ("spine_yaw", 5, 6),
            ("shoulder_roll", 6, 7),
            ("shoulder_pitch", 7, 8),
            ("shoulder_yaw", 8, 9),
        ],
    ),
    NodeFeatureSpec(
        "upper_arm_emg",
        19,
        27,
        components=[(f"upper_arm_emg_c{i+1}", i, i + 1) for i in range(8)],
    ),
    NodeFeatureSpec(
        "lower_arm_emg",
        27,
        35,
        components=[(f"lower_arm_emg_c{i+1}", i, i + 1) for i in range(8)],
    ),
    NodeFeatureSpec(
        "right_hand_euler",
        35,
        38,
        components=[
            ("hand_roll", 0, 1),
            ("hand_pitch", 1, 2),
            ("hand_yaw", 2, 3),
        ],
    ),
]


@dataclass
class ChainBMDConfig:
    """Main hyperparameters for reuse in config.py."""
    in_features: int = 16
    gcn_hidden: int = 64
    gcn_layers: int = 2
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    mlp_hidden: int = 64
    num_classes: int = 3
    dropout: float = 0.1
    raw_feature_dim: Optional[int] = None
    node_feature_specs: List[NodeFeatureSpec] = None
    use_gcn: bool = True
    use_transformer: bool = True
    use_positional_encoding: bool = True
    use_cls_token: bool = True
    use_lrp: bool = True


class ChainBMDModel(nn.Module):
    """ChainBMD core: GCN + Transformer + classifier + LRP explanation."""

    def __init__(self, config: ChainBMDConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.node_feature_specs is None:
            self.config.node_feature_specs = DEFAULT_NODE_SPECS
        self.node_specs = self.config.node_feature_specs
        self.num_nodes = len(self.node_specs)
        self.node_names = [spec.name for spec in self.node_specs]
        self.use_gcn = bool(self.config.use_gcn)
        self.use_transformer = bool(self.config.use_transformer)
        self.use_positional_encoding = bool(self.config.use_positional_encoding)
        self.use_cls_token = bool(self.config.use_cls_token)
        self.use_lrp = bool(self.config.use_lrp)

        # Build and cache the chain graph; no updates during training.
        adjacency = normalize_adjacency(build_chain_adjacency(self.num_nodes))
        self.register_buffer("adjacency", adjacency)

        # Project per-node features to a shared in_features dimension.
        self.node_tokenizers = nn.ModuleList(
            nn.Linear(spec.dim, config.in_features) for spec in self.node_specs
        )

        # Stack GCN layers to extract per-frame spatial features.
        self.gcn_layers = nn.ModuleList()
        gcn_output_dim = config.in_features
        if self.use_gcn and config.gcn_layers > 0:
            gcn_dims = [config.in_features] + [config.gcn_hidden] * config.gcn_layers
            self.gcn_layers = nn.ModuleList(
                GraphConvolution(gcn_dims[i], gcn_dims[i + 1]) for i in range(len(gcn_dims) - 1)
            )
            gcn_output_dim = gcn_dims[-1]
        self.gcn_output_dim = gcn_output_dim

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        # Project GCN outputs to Transformer dimension.
        self.transformer_input_proj = nn.Linear(self.gcn_output_dim, config.transformer_dim)
        if self.use_transformer:
            if config.transformer_layers < 1:
                raise ValueError("use_transformer=True requires transformer_layers >= 1.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.transformer_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        else:
            self.transformer = None
        self.positional_encoding = (
            SinePositionalEncoding(config.transformer_dim) if self.use_transformer and self.use_positional_encoding else None
        )
        # BERT-style CLS token to aggregate global motion (kept for easy toggling).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.transformer_dim))
        self.norm = nn.LayerNorm(config.transformer_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.num_classes),
        )

    def forward(
        self, node_sequence: torch.Tensor, return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_sequence: Tensor
                either (batch, seq_len, num_nodes, in_features)
                or (batch, seq_len, raw_feature_dim) if feature_tokenizer is set.
            return_intermediate: Whether to return node-level features for LRP.
        """
        node_sequence = self._prepare_node_sequence(node_sequence)
        batch_size, seq_len, num_nodes, _ = node_sequence.shape
        x = node_sequence.view(batch_size * seq_len, num_nodes, -1)
        # Flatten time dimension and run per-frame GCN.
        if self.use_gcn and len(self.gcn_layers) > 0:
            for layer in self.gcn_layers:
                x = self.activation(layer(x, self.adjacency))
                x = self.dropout(x)
        node_embeddings = x.view(batch_size, seq_len, num_nodes, -1)

        # Pool node features to frame tokens for temporal modeling.
        temporal_tokens = node_embeddings.mean(dim=2)  # (batch, seq_len, hidden)
        temporal_tokens = self.transformer_input_proj(temporal_tokens)
        transformer_output = None
        if self.use_transformer and self.transformer is not None:
            transformer_input = temporal_tokens
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                transformer_input = torch.cat([cls_tokens, transformer_input], dim=1)
            if self.use_positional_encoding and self.positional_encoding is not None:
                transformer_input = self.positional_encoding(transformer_input)
            transformer_output = self.transformer(transformer_input)
            if self.use_cls_token:
                cls_output = transformer_output[:, 0]
            else:
                cls_output = transformer_output.mean(dim=1)
        else:
            cls_output = temporal_tokens.mean(dim=1)
        z = self.norm(cls_output)
        logits = self.classifier(z)

        if return_intermediate:
            return logits, z, (node_embeddings, temporal_tokens, transformer_output)
        transformer_snapshot = transformer_output.detach() if transformer_output is not None else None
        return logits, z, (
            node_embeddings.detach(),
            temporal_tokens.detach(),
            transformer_snapshot,
        )

    def explain(
        self, node_sequence: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """Generate node/time relevance via LRP (fallback to grad*input)."""
        self.eval()
        raw_sequence = None
        if node_sequence.dim() == 3:
            raw_sequence = node_sequence.detach()
            raw_sequence.requires_grad_(True)
            prepared = self._prepare_node_sequence(raw_sequence)
        else:
            prepared = self._prepare_node_sequence(node_sequence)
        logits, _, (node_embeds, temporal_tokens, transformer_output) = self.forward(
            prepared, return_intermediate=True
        )
        if target_class is None:
            target_idx = logits.argmax(dim=-1)
        else:
            target_idx = torch.full(
                (logits.size(0),), target_class, dtype=torch.long, device=logits.device
            )
        class_selected = logits.gather(1, target_idx.unsqueeze(1)).sum()
        can_use_lrp = (
            self.use_lrp
            and self.use_transformer
            and self.use_cls_token
            and transformer_output is not None
        )
        if can_use_lrp:
            class_relevance = classifier_lrp(logits, target_idx)
            cls_token_relevance = class_relevance.unsqueeze(1)
            transformer_relevance = transformer_lrp(transformer_output, cls_token_relevance)
            token_relevance = transformer_relevance[:, 1:, :]  # remove CLS
            temporal_relevance = torch.einsum(
                "btd,df->btf", token_relevance, self.transformer_input_proj.weight
            ).detach()
            node_time_raw = torch.einsum("bsnf,bsf->bsn", node_embeds.detach(), temporal_relevance)
            node_time_abs = node_time_raw.abs()
        else:
            node_time_abs = self._grad_node_time(class_selected, node_embeds)
        node_contrib = F.softmax(node_time_abs.sum(dim=1), dim=-1)
        time_importance = F.softmax(node_time_abs.sum(dim=2), dim=-1)
        raw_grad = self._compute_raw_grad(raw_sequence, class_selected)
        node_series = {}
        for idx, name in enumerate(self.node_names):
            series = node_time_abs[:, :, idx]
            min_val = series.min(dim=1, keepdim=True).values
            max_val = series.max(dim=1, keepdim=True).values
            norm = (series - min_val) / (max_val - min_val + 1e-6)
            node_series[name] = norm.detach().cpu().tolist()
        node_components = {}
        if raw_grad is not None:
            for spec in self.node_specs:
                comp_dict = {}
                for comp_name, start, end in spec.component_ranges():
                    comp_values = raw_grad[:, :, start:end].sum(dim=-1)
                    comp_dict[comp_name] = comp_values.detach().cpu().tolist()
                node_components[spec.name] = comp_dict

        explanations: List[Dict[str, Any]] = []
        batch_size = prepared.shape[0]
        for batch_idx in range(batch_size):
            explanations.append(
                {
                    "pred_class": target_idx[batch_idx].item(),
                    "node_contributions": {
                        name: node_contrib[batch_idx, node_idx].item()
                        for node_idx, name in enumerate(self.node_names)
                    },
                    "time_importance": time_importance[batch_idx].detach().cpu().tolist(),
                    "node_time_series": {
                        name: node_series[name][batch_idx] for name in self.node_names
                    },
                    "node_components": {
                        name: {
                            comp: node_components.get(name, {}).get(
                                comp, [0.0] * time_importance.size(1)
                            )
                            for comp in (node_components.get(name, {}) or {})
                        }
                        for name in node_components
                    },
                }
            )
        return explanations, target_idx

    def _grad_node_time(self, class_selected: torch.Tensor, node_embeds: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(
            class_selected,
            node_embeds,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grads is None:
            grads = torch.zeros_like(node_embeds)
        node_time = (grads * node_embeds).abs().sum(dim=-1)
        return node_time.detach()

    def _compute_raw_grad(
        self, raw_sequence: Optional[torch.Tensor], class_selected: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if raw_sequence is None:
            return None
        self.zero_grad(set_to_none=True)
        class_selected.backward(retain_graph=False)
        grad = raw_sequence.grad
        if grad is None:
            return None
        return (grad * raw_sequence).detach()

    def _prepare_node_sequence(self, node_sequence: torch.Tensor) -> torch.Tensor:
        if node_sequence.dim() == 3:
            batch, seq_len, raw_dim = node_sequence.shape
            if self.config.raw_feature_dim is not None and raw_dim != self.config.raw_feature_dim:
                raise ValueError(f"Expected raw feature dim {self.config.raw_feature_dim}, got {raw_dim}")
            node_embeddings = []
            for spec, tokenizer in zip(self.node_specs, self.node_tokenizers):
                node_slice = node_sequence[..., spec.start:spec.end]
                node_embed = tokenizer(node_slice)
                node_embeddings.append(node_embed.unsqueeze(2))
            return torch.cat(node_embeddings, dim=2)
        if node_sequence.dim() == 4:
            if node_sequence.size(2) != self.num_nodes or node_sequence.size(3) != self.config.in_features:
                raise ValueError("Node sequence shape does not match config.")
            return node_sequence
        raise ValueError("node_sequence must be rank-3 or rank-4 tensor.")


def generate_dummy_sequence(
    batch_size: int = 2, seq_len: int = 60, num_nodes: int = len(NODE_NAMES), feature_dim: int = 16
) -> torch.Tensor:
    """Creates placeholder data for quick experiments."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, num_nodes, feature_dim)


if __name__ == "__main__":
    cfg = ChainBMDConfig()
    model = ChainBMDModel(cfg)
    dummy_sequence = generate_dummy_sequence(feature_dim=cfg.in_features)
    logits, z, _ = model(dummy_sequence)
    probs = logits.softmax(dim=-1)
    print("Predicted probabilities:", probs)
    relevances, predicted = model.explain(dummy_sequence)
    print("Predicted classes:", predicted)
    print("Node contributions for batch[0]:", relevances[0])
