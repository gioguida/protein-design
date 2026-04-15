"""Mutational path scoring and Spearman evaluation for evotuned ESM2."""

import logging
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

C05_CDRH3 = "HMSMQQVVSAGWERADLVGDAFDV"  # 24 aa wild-type CDRH3

_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutations(mut_str: str, wt: str) -> list[tuple[int, str, str]]:
    """Parse mutation string into list of (0-indexed position, wt_aa, mut_aa).

    Args:
        mut_str: Semicolon-separated mutations, e.g. "H1A;M2C".
        wt: Wild-type sequence.

    Returns:
        List of (pos, wt_aa, mut_aa) tuples with 0-indexed positions.
    """
    mutations = []
    for token in mut_str.split(";"):
        m = _MUT_RE.match(token.strip())
        if m is None:
            raise ValueError(f"Cannot parse mutation token: {token!r}")
        wt_aa, pos_str, mut_aa = m.groups()
        pos = int(pos_str) - 1  # convert to 0-indexed
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
    """Compute log P(target_aa | sequence with mask_pos masked) for each sequence.

    Args:
        model: ESM2 model (EvotuningModel or EsmForMaskedLM wrapper).
        tokenizer: ESM2 tokenizer.
        sequences: List of amino acid strings (all same length).
        mask_positions: List of 0-indexed positions to mask in each sequence.
        target_aas: List of single amino acid characters to score at each masked position.
        device: Torch device.
        batch_size: Batch size for forward passes.

    Returns:
        Array of log-probabilities, shape (len(sequences),).
    """
    n = len(sequences)
    log_probs = np.empty(n, dtype=np.float32)

    # Pre-compute target token IDs
    target_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in target_aas]
    mask_token_id = tokenizer.mask_token_id

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = sequences[start:end]

            # Tokenize batch (all same length → no padding needed)
            encoding = tokenizer(
                batch_seqs, return_tensors="pt", padding=False, truncation=False
            )
            input_ids = encoding["input_ids"].clone()
            attention_mask = encoding["attention_mask"]

            # Mask the target position for each sequence
            # Token offset: ESM2 prepends <cls>, so string pos i → token index i+1
            for i in range(end - start):
                token_pos = mask_positions[start + i] + 1
                input_ids[i, token_pos] = mask_token_id

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits  # (batch, seq_len, vocab_size)
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
    """Score double-mutant sequences using mutational path scoring.

    For each sequence with mutations at positions i, j:
      Order A: log P(mut_i | WT) + log P(mut_j | WT + mut_i)
      Order B: log P(mut_j | WT) + log P(mut_i | WT + mut_j)

    Args:
        model: ESM2 model.
        tokenizer: ESM2 tokenizer.
        wt: Wild-type CDRH3 sequence.
        df: DataFrame with 'mut' column containing mutation strings.
        device: Torch device.
        strategy: "average" (mean of both orderings) or "random" (pick one per sequence).
        batch_size: Batch size for forward passes.
        seed: Random seed for "random" strategy.

    Returns:
        Array of scores, shape (len(df),).
    """
    n = len(df)

    # Build the 4 arrays: (sequence, mask_pos, target_aa) for each step
    seqs_a1, pos_a1, tgt_a1 = [], [], []  # Order A step 1: WT, mask i
    seqs_a2, pos_a2, tgt_a2 = [], [], []  # Order A step 2: WT+mut_i, mask j
    seqs_b1, pos_b1, tgt_b1 = [], [], []  # Order B step 1: WT, mask j
    seqs_b2, pos_b2, tgt_b2 = [], [], []  # Order B step 2: WT+mut_j, mask i

    wt_list = list(wt)

    for mut_str in df["mut"]:
        muts = parse_mutations(mut_str, wt)
        if len(muts) != 2:
            raise ValueError(f"Expected exactly 2 mutations, got {len(muts)}: {mut_str}")
        (pos_i, _, mut_i_aa), (pos_j, _, mut_j_aa) = muts

        # Order A step 1: mask pos_i in WT
        seqs_a1.append(wt)
        pos_a1.append(pos_i)
        tgt_a1.append(mut_i_aa)

        # Order A step 2: apply mut_i to WT, mask pos_j
        seq_with_i = wt_list.copy()
        seq_with_i[pos_i] = mut_i_aa
        seqs_a2.append("".join(seq_with_i))
        pos_a2.append(pos_j)
        tgt_a2.append(mut_j_aa)

        # Order B step 1: mask pos_j in WT
        seqs_b1.append(wt)
        pos_b1.append(pos_j)
        tgt_b1.append(mut_j_aa)

        # Order B step 2: apply mut_j to WT, mask pos_i
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


def evaluate_spearman(
    scores: np.ndarray, enrichment: np.ndarray
) -> tuple[float, float]:
    """Compute Spearman rank correlation, filtering NaN/inf values.

    Returns:
        (rho, p_value) tuple.
    """
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
    """Load ED2 data, filter to exactly 2 mutations, and subsample.

    Args:
        data_path: Path to CSV file (e.g. data/processed/D2.csv).
        n_samples: Number of sequences to subsample.
        enrichment_col: Column name for enrichment values.
        seed: Random seed for subsampling.

    Returns:
        Subsampled DataFrame with 'mut' and enrichment_col columns.
    """
    df = pd.read_csv(data_path)
    df = df[df["num_mut"] == 2].copy()
    logger.info("Loaded %d double-mutant sequences from %s", len(df), data_path)

    # Filter out rows with NaN/inf enrichment
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
    """Run full scoring evaluation with both ordering strategies.

    Args:
        model: ESM2 model (will be set to eval mode, caller should restore train mode).
        tokenizer: ESM2 tokenizer.
        df: DataFrame from load_scoring_data.
        enrichment_col: Column name for enrichment values.
        device: Torch device.
        batch_size: Batch size for forward passes.
        seed: Random seed.

    Returns:
        Dict with keys: spearman_avg, spearman_avg_pval, spearman_random, spearman_random_pval.
    """
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
    }


def load_scoring_datasets(
    datasets_config: list[dict],
    n_samples: int,
    seed: int,
) -> list[tuple[str, pd.DataFrame, str]]:
    """Load multiple scoring datasets from config.

    Args:
        datasets_config: List of dicts with keys 'name', 'path', 'enrichment_col'.
        n_samples: Number of sequences to subsample per dataset.
        seed: Random seed for subsampling.

    Returns:
        List of (name, DataFrame, enrichment_col) tuples.
    """
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
) -> dict:
    """Run scoring evaluation across multiple datasets.

    Args:
        model: ESM2 model.
        tokenizer: ESM2 tokenizer.
        datasets: List of (name, df, enrichment_col) from load_scoring_datasets.
        device: Torch device.
        batch_size: Batch size for forward passes.
        seed: Random seed.

    Returns:
        Flat dict with keys like 'spearman_avg_M22', 'spearman_random_SI06', etc.
    """
    results = {}
    for name, df, enrichment_col in datasets:
        logger.info("Scoring dataset: %s", name)
        ds_results = run_scoring_evaluation(
            model, tokenizer, df, enrichment_col, device, batch_size, seed,
        )
        results[f"spearman_avg_{name}"] = ds_results["spearman_avg"]
        results[f"spearman_avg_pval_{name}"] = ds_results["spearman_avg_pval"]
        results[f"spearman_random_{name}"] = ds_results["spearman_random"]
        results[f"spearman_random_pval_{name}"] = ds_results["spearman_random_pval"]
    return results
