"""Losses for unlikelihood training."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def build_unwanted_token_id_lookup(
    unwanted_lookup: Dict[int, List[str]],
    tokenizer,
) -> Dict[int, List[int]]:
    """Map unwanted amino-acid letters to tokenizer ids per 1-indexed CDR position."""
    token_ids_by_position: Dict[int, List[int]] = {}
    for position, aas in unwanted_lookup.items():
        ids: List[int] = []
        for aa in aas:
            token_id = tokenizer.convert_tokens_to_ids(str(aa))
            if token_id is None or int(token_id) < 0:
                continue
            ids.append(int(token_id))
        # Deduplicate while keeping stable ordering.
        deduped = list(dict.fromkeys(ids))
        if deduped:
            token_ids_by_position[int(position)] = deduped
    return token_ids_by_position


def unlikelihood_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    cdr_positions: torch.Tensor,
    unwanted_token_ids_by_position: Dict[int, List[int]],
    alpha: float,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute MLM + unlikelihood loss at masked positions.

    For each masked token i:
      L_i = -log p(x_i) - alpha * sum_{c in C_i} log(1 - p(c))
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, L, V], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits batch/sequence dimensions.")
    if cdr_positions.shape != labels.shape:
        raise ValueError("cdr_positions must match labels shape.")

    masked = labels.ne(-100)
    if not masked.any():
        zero = logits.new_tensor(0.0)
        return {
            "loss": zero,
            "mlm_loss": zero,
            "unlikelihood_loss": zero,
            "unwanted_probability": zero,
            "num_masked": zero,
            "num_masked_with_unwanted": zero,
        }

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()

    labels_masked = labels[masked]
    cdr_pos_masked = cdr_positions[masked]
    row_idx, col_idx = torch.where(masked)

    mlm_per_token = -log_probs[row_idx, col_idx, labels_masked]
    unlikelihood_per_token = torch.zeros_like(mlm_per_token)
    unwanted_prob_per_token = torch.zeros_like(mlm_per_token)
    has_unwanted = torch.zeros_like(mlm_per_token, dtype=torch.bool)

    for position, token_ids in unwanted_token_ids_by_position.items():
        if not token_ids:
            continue
        pos_mask = cdr_pos_masked.eq(int(position))
        if not pos_mask.any():
            continue

        idx = torch.where(pos_mask)[0]
        token_id_tensor = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        selected_probs = probs[row_idx[idx], col_idx[idx]][:, token_id_tensor]
        selected_probs = selected_probs.clamp(min=0.0, max=1.0 - float(eps))

        unlikelihood_per_token[idx] = -torch.log1p(-selected_probs).sum(dim=-1)
        unwanted_prob_per_token[idx] = selected_probs.sum(dim=-1)
        has_unwanted[idx] = True

    mlm_loss = mlm_per_token.mean()
    unlikelihood_loss = unlikelihood_per_token.mean()
    total_loss = mlm_loss + float(alpha) * unlikelihood_loss

    if has_unwanted.any():
        unwanted_probability = unwanted_prob_per_token[has_unwanted].mean()
        num_masked_with_unwanted = has_unwanted.sum().to(dtype=logits.dtype)
    else:
        unwanted_probability = logits.new_tensor(0.0)
        num_masked_with_unwanted = logits.new_tensor(0.0)

    return {
        "loss": total_loss,
        "mlm_loss": mlm_loss,
        "unlikelihood_loss": unlikelihood_loss,
        "unwanted_probability": unwanted_probability,
        "num_masked": masked.sum().to(dtype=logits.dtype),
        "num_masked_with_unwanted": num_masked_with_unwanted,
    }
