"""Perplexity evaluation utilities for evotuned ESM2."""

import logging
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

C05_VH = (
    "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEG"
    "RFTISRDNSKETLYLQMTNLRVEDTGVYYCAKHMSMQQVVSAGWERADLVGDAFDVWGQGTMVTVSS"
)
C05_CDRH3 = "HMSMQQVVSAGWERADLVGDAFDV"
C05_CDRH3_START = C05_VH.index(C05_CDRH3)  # 103
C05_CDRH3_END = C05_CDRH3_START + len(C05_CDRH3)  # 127


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


def compute_cdr_pseudo_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    full_sequence: str = C05_VH,
    cdr_start: int = C05_CDRH3_START,
    cdr_end: int = C05_CDRH3_END,
) -> float:
    """Compute pseudo-perplexity over CDR-H3 positions using full-sequence context.

    Masks one CDR position at a time, feeds the full VH sequence as context,
    and averages the log-probabilities of the correct amino acid.

    Returns:
        Pseudo-perplexity = exp(-mean(log P(aa_i | full context))).
    """
    from protein_design.scoring import compute_masked_log_probs_batch

    cdr_len = cdr_end - cdr_start
    sequences = [full_sequence] * cdr_len
    mask_positions = list(range(cdr_start, cdr_end))
    target_aas = list(full_sequence[cdr_start:cdr_end])

    log_probs = compute_masked_log_probs_batch(
        model, tokenizer, sequences, mask_positions, target_aas, device,
        batch_size=cdr_len,
    )
    ppl = math.exp(-float(np.mean(log_probs)))
    logger.info("CDR-H3 pseudo-perplexity: %.2f (mean log P: %.4f)", ppl, np.mean(log_probs))
    return ppl
