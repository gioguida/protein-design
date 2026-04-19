"""Shared evaluation: MLM + PLL perplexity, CDR pseudo-perplexity, Spearman scoring."""

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from protein_design.model import ESM2Model
from protein_design.constants import (
    C05_CDRH3,
    C05_CDRH3_END,
    C05_CDRH3_START,
    C05_VH,
)

logger = logging.getLogger(__name__)

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ---------------------------------------------------------------------------
# MLM perplexity (evotuning)
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Compute masked-language-model perplexity over a dataloader.

    Returns:
        (perplexity, avg_loss) where perplexity = exp(avg_loss).
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


# ---------------------------------------------------------------------------
# PLL perplexity (DPO)
# ---------------------------------------------------------------------------


def sequence_perplexity(
    sequences: Sequence[str],
    scorer: ESM2Model,
    cdr_only: bool = True,
) -> torch.Tensor:
    """Compute per-sequence perplexity based on Pseudo-Log-Likelihood (PLL).

    Returns a tensor of shape [batch_size] containing perplexity scores.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")

    seq_lengths = {len(seq) for seq in sequences}
    if len(seq_lengths) != 1:
        raise ValueError("All sequences in a batch must have the same length.")

    pll_scores = scorer.pseudo_log_likelihood(
        sequences, cdr_only=cdr_only, use_grad=False
    )

    if cdr_only:
        N = float(next(iter(seq_lengths)))
    else:
        N = float(scorer.tokenize_sequences([sequences[0]]).shape[1])

    if N <= 0:
        raise ValueError("Number of scored positions must be positive.")

    return torch.exp(-pll_scores / N)


def corpus_perplexity(
    sequences: Sequence[str],
    scorer: ESM2Model,
    cdr_only: bool = True,
) -> float:
    """Compute corpus-level perplexity as exp(total_nll / total_scored_tokens).

    Unlike mean per-sequence perplexity, this matches the common MLM-style
    evaluation where loss is averaged before exponentiation.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")

    total_pll = 0.0
    total_scored_tokens = 0

    if cdr_only:
        by_len: Dict[int, List[str]] = {}
        for seq in sequences:
            by_len.setdefault(len(seq), []).append(seq)

        for cdr_len, seq_group in by_len.items():
            if cdr_len <= 0:
                continue
            pll_scores = scorer.pseudo_log_likelihood(
                seq_group, cdr_only=True, use_grad=False
            )
            total_pll += float(pll_scores.sum().item())
            total_scored_tokens += int(cdr_len) * len(seq_group)
    else:
        seq_lengths = {len(seq) for seq in sequences}
        if len(seq_lengths) != 1:
            raise ValueError(
                "All sequences must have the same length when cdr_only=False."
            )
        pll_scores = scorer.pseudo_log_likelihood(
            sequences, cdr_only=False, use_grad=False
        )
        tokens_per_sequence = int(
            scorer.tokenize_sequences([sequences[0]]).shape[1]
        )
        total_pll += float(pll_scores.sum().item())
        total_scored_tokens += tokens_per_sequence * len(sequences)

    if total_scored_tokens <= 0:
        raise ValueError("No valid tokens were evaluated for perplexity.")

    avg_nll = -total_pll / float(total_scored_tokens)
    return float(math.exp(avg_nll))


# ---------------------------------------------------------------------------
# Mutation parsing + masked log-prob scoring
# ---------------------------------------------------------------------------


def parse_mutations(mut_str: str, wt: str) -> list[tuple[int, str, str]]:
    """Parse mutation string into list of (0-indexed position, wt_aa, mut_aa)."""
    mutations = []
    for token in mut_str.split(";"):
        m = _MUT_RE.match(token.strip())
        if m is None:
            raise ValueError(f"Cannot parse mutation token: {token!r}")
        wt_aa, pos_str, mut_aa = m.groups()
        pos = int(pos_str) - 1
        if wt[pos] != wt_aa:
            raise ValueError(
                f"WT mismatch at position {pos}: expected {wt[pos]!r}, got {wt_aa!r}"
            )
        mutations.append((pos, wt_aa, mut_aa))
    return mutations


def compute_masked_log_probs_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    mask_positions: list[int],
    target_aas: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute log P(target_aa | sequence with mask_pos masked) for each sequence."""
    n = len(sequences)
    log_probs = np.empty(n, dtype=np.float32)

    target_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in target_aas]
    mask_token_id = tokenizer.mask_token_id

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]

            encoding = tokenizer(
                batch_seqs, return_tensors="pt", padding=False, truncation=False
            )
            input_ids = encoding["input_ids"].clone()
            attention_mask = encoding["attention_mask"]

            for i in range(end - start):
                token_pos = mask_positions[start + i] + 1
                input_ids[i, token_pos] = mask_token_id

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            log_softmax = F.log_softmax(logits.float(), dim=-1)

            for i in range(end - start):
                token_pos = mask_positions[start + i] + 1
                tid = target_ids[start + i]
                log_probs[start + i] = log_softmax[i, token_pos, tid].item()

    return log_probs


def score_double_mutants(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    wt: str,
    df: pd.DataFrame,
    device: torch.device,
    strategy: str = "average",
    batch_size: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Score double-mutant sequences using mutational path scoring."""
    n = len(df)

    seqs_a1, pos_a1, tgt_a1 = [], [], []
    seqs_a2, pos_a2, tgt_a2 = [], [], []
    seqs_b1, pos_b1, tgt_b1 = [], [], []
    seqs_b2, pos_b2, tgt_b2 = [], [], []

    wt_list = list(wt)

    for mut_str in df["mut"]:
        muts = parse_mutations(mut_str, wt)
        if len(muts) != 2:
            raise ValueError(f"Expected exactly 2 mutations, got {len(muts)}: {mut_str}")
        (pos_i, _, mut_i_aa), (pos_j, _, mut_j_aa) = muts

        seqs_a1.append(wt)
        pos_a1.append(pos_i)
        tgt_a1.append(mut_i_aa)

        seq_with_i = wt_list.copy()
        seq_with_i[pos_i] = mut_i_aa
        seqs_a2.append("".join(seq_with_i))
        pos_a2.append(pos_j)
        tgt_a2.append(mut_j_aa)

        seqs_b1.append(wt)
        pos_b1.append(pos_j)
        tgt_b1.append(mut_j_aa)

        seq_with_j = wt_list.copy()
        seq_with_j[pos_j] = mut_j_aa
        seqs_b2.append("".join(seq_with_j))
        pos_b2.append(pos_i)
        tgt_b2.append(mut_i_aa)

    logger.info("Scoring %d sequences (4 × %d forward-pass batches)", n, -(-n // batch_size))

    logp_a1 = compute_masked_log_probs_batch(model, tokenizer, seqs_a1, pos_a1, tgt_a1, device, batch_size)
    logp_a2 = compute_masked_log_probs_batch(model, tokenizer, seqs_a2, pos_a2, tgt_a2, device, batch_size)
    logp_b1 = compute_masked_log_probs_batch(model, tokenizer, seqs_b1, pos_b1, tgt_b1, device, batch_size)
    logp_b2 = compute_masked_log_probs_batch(model, tokenizer, seqs_b2, pos_b2, tgt_b2, device, batch_size)

    score_a = logp_a1 + logp_a2
    score_b = logp_b1 + logp_b2

    if strategy == "average":
        return (score_a + score_b) / 2.0
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        pick_a = rng.randint(0, 2, size=n).astype(bool)
        return np.where(pick_a, score_a, score_b)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# CDR pseudo-perplexity (evotuning)
# ---------------------------------------------------------------------------


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
    """
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


# ---------------------------------------------------------------------------
# Spearman evaluation
# ---------------------------------------------------------------------------


def evaluate_spearman(
    scores: np.ndarray, enrichment: np.ndarray
) -> tuple[float, float]:
    """Compute Spearman rank correlation, filtering NaN/inf values."""
    mask = np.isfinite(scores) & np.isfinite(enrichment)
    n_valid = mask.sum()
    if n_valid < 3:
        logger.warning("Too few valid values for Spearman: %d", n_valid)
        return float("nan"), float("nan")
    rho, pval = spearmanr(scores[mask], enrichment[mask])
    return float(rho), float(pval)


def load_scoring_data(
    data_path: str,
    n_samples: int,
    enrichment_col: str,
    seed: int,
) -> pd.DataFrame:
    """Load ED2 data, filter to exactly 2 mutations, and subsample."""
    df = pd.read_csv(data_path)
    df = df[df["num_mut"] == 2].copy()
    logger.info("Loaded %d double-mutant sequences from %s", len(df), data_path)

    df = df[np.isfinite(df[enrichment_col])].copy()

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
        logger.info("Subsampled to %d sequences", n_samples)

    df = df.reset_index(drop=True)
    return df


def run_scoring_evaluation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    enrichment_col: str,
    device: torch.device,
    batch_size: int = 512,
    seed: int = 42,
) -> dict:
    """Run full scoring evaluation with both ordering strategies."""
    enrichment = df[enrichment_col].values

    scores_avg = score_double_mutants(
        model, tokenizer, C05_CDRH3, df, device,
        strategy="average", batch_size=batch_size, seed=seed,
    )
    rho_avg, pval_avg = evaluate_spearman(scores_avg, enrichment)
    logger.info("Spearman (average ordering): rho=%.4f, p=%.2e", rho_avg, pval_avg)

    scores_rnd = score_double_mutants(
        model, tokenizer, C05_CDRH3, df, device,
        strategy="random", batch_size=batch_size, seed=seed,
    )
    rho_rnd, pval_rnd = evaluate_spearman(scores_rnd, enrichment)
    logger.info("Spearman (random ordering):  rho=%.4f, p=%.2e", rho_rnd, pval_rnd)

    return {
        "spearman_avg": rho_avg,
        "spearman_avg_pval": pval_avg,
        "spearman_random": rho_rnd,
        "spearman_random_pval": pval_rnd,
        "scores_avg": scores_avg,
        "scores_random": scores_rnd,
    }


def load_scoring_datasets(
    datasets_config: list[dict],
    n_samples: int,
    seed: int,
) -> list[tuple[str, pd.DataFrame, str]]:
    """Load multiple scoring datasets from config."""
    datasets = []
    for ds in datasets_config:
        name = ds["name"]
        df = load_scoring_data(ds["path"], n_samples, ds["enrichment_col"], seed)
        logger.info("Scoring dataset %r: %d sequences", name, len(df))
        datasets.append((name, df, ds["enrichment_col"]))
    return datasets


def run_multi_scoring_evaluation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    datasets: list[tuple[str, pd.DataFrame, str]],
    device: torch.device,
    batch_size: int = 512,
    seed: int = 42,
    scores_csv_dir: str | None = None,
) -> dict:
    """Run scoring evaluation across multiple datasets.

    If `scores_csv_dir` is set, write `<dataset>_scores.csv` with per-pair
    scores (both strategies) alongside the metrics.
    """
    results = {}
    csv_dir = Path(scores_csv_dir) if scores_csv_dir else None
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    for name, df, enrichment_col in datasets:
        logger.info("Scoring dataset: %s", name)
        ds_results = run_scoring_evaluation(
            model, tokenizer, df, enrichment_col, device, batch_size, seed,
        )
        results[f"spearman_avg_{name}"] = ds_results["spearman_avg"]
        results[f"spearman_avg_pval_{name}"] = ds_results["spearman_avg_pval"]
        results[f"spearman_random_{name}"] = ds_results["spearman_random"]
        results[f"spearman_random_pval_{name}"] = ds_results["spearman_random_pval"]

        if csv_dir is not None:
            cols = [c for c in ("aa", "mut", enrichment_col) if c in df.columns]
            out = df[cols].copy()
            out["score_avg"] = ds_results["scores_avg"]
            out["score_random"] = ds_results["scores_random"]
            out_path = csv_dir / f"{name}_scores.csv"
            out.to_csv(out_path, index=False)
            logger.info("Wrote per-pair scores: %s", out_path)

    return results
