"""Perplexity evaluation utilities for evotuned ESM2."""

import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Compute masked-language-model perplexity over a dataloader.

    Args:
        model: EvotuningModel (or any model returning MaskedLMOutput).
        dataloader: Validation dataloader with MLM masking.
        device: Device to run inference on.
        max_batches: Cap evaluation at this many batches for speed.

    Returns:
        Tuple of (perplexity, avg_loss) where perplexity = exp(avg_loss).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", total=min(max_batches, len(dataloader))):
            if n_batches >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss)
    logger.info("Perplexity: %.2f (avg loss: %.4f, %d batches)", ppl, avg_loss, n_batches)
    return ppl, avg_loss
