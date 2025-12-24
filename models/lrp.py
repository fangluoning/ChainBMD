from typing import Tuple

import torch


def _stabilize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values + eps * torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))


def linear_lrp(
    activations: torch.Tensor,
    weights: torch.Tensor,
    relevance: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    z = torch.einsum("bni,io->bno", activations, weights)
    z = _stabilize(z, eps)
    s = relevance / z
    c = torch.einsum("bno,bni->bni", s, activations)
    return c


def classifier_lrp(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    relevance = torch.zeros_like(logits)
    relevance[batch_indices, target_idx] = logits[batch_indices, target_idx]
    return relevance


def transformer_lrp(
    transformer_output: torch.Tensor,
    cls_relevance: torch.Tensor,
) -> torch.Tensor:
    total = transformer_output.sum(dim=-1, keepdim=True)
    total = _stabilize(total)
    coeffs = cls_relevance.sum(dim=-1, keepdim=True) / total
    return coeffs * transformer_output
