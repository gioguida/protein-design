"""Shared evaluation: MLM + PLL perplexity and Spearman scoring."""

import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    LEFT_CONTEXT,
    RIGHT_CONTEXT,
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
    model: Optional[torch.nn.Module],
    tokenizer: Optional[AutoTokenizer],
    sequences: list[str],
    mask_positions: list[int],
    target_aas: list[str],
    device: Optional[torch.device],
    batch_size: int = 512,
    scorer: Optional[ESM2Model] = None,
) -> np.ndarray:
    """Compute log P(target_aa | sequence with mask_pos masked) for each sequence."""
    n = len(sequences)
    log_probs = np.empty(n, dtype=np.float32)

    if scorer is not None:
        target_ids = [scorer.tokenizer.convert_tokens_to_ids(aa) for aa in target_aas]
        mask_token_id = scorer.mask_token_idx
        scorer_device = scorer.device
        scorer.model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_seqs = sequences[start:end]
                batch_positions = mask_positions[start:end]
                token_positions = scorer.cdr_to_token_positions(batch_positions)

                input_ids = scorer.tokenize_sequences(batch_seqs).clone()
                for i, token_pos in enumerate(token_positions):
                    input_ids[i, token_pos] = mask_token_id

                with torch.amp.autocast("cuda", enabled=scorer_device.type == "cuda"):
                    logits = scorer.forward_logits(input_ids)

                log_softmax = F.log_softmax(logits.float(), dim=-1)
                for i, token_pos in enumerate(token_positions):
                    tid = target_ids[start + i]
                    log_probs[start + i] = log_softmax[i, token_pos, tid].item()
        return log_probs

    if model is None or tokenizer is None or device is None:
        raise ValueError(
            "model, tokenizer, and device are required when scorer is not provided."
        )

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
    model: Optional[torch.nn.Module],
    tokenizer: Optional[AutoTokenizer],
    wt: str,
    df: pd.DataFrame,
    device: Optional[torch.device],
    strategy: str = "average",
    batch_size: int = 512,
    seed: int = 42,
    scorer: Optional[ESM2Model] = None,
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

    logp_a1 = compute_masked_log_probs_batch(
        model, tokenizer, seqs_a1, pos_a1, tgt_a1, device, batch_size, scorer=scorer
    )
    logp_a2 = compute_masked_log_probs_batch(
        model, tokenizer, seqs_a2, pos_a2, tgt_a2, device, batch_size, scorer=scorer
    )
    logp_b1 = compute_masked_log_probs_batch(
        model, tokenizer, seqs_b1, pos_b1, tgt_b1, device, batch_size, scorer=scorer
    )
    logp_b2 = compute_masked_log_probs_batch(
        model, tokenizer, seqs_b2, pos_b2, tgt_b2, device, batch_size, scorer=scorer
    )

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


def score_sequences_cdr_pll(
    scorer: ESM2Model,
    sequences: Sequence[str],
    batch_size: int = 512,
) -> np.ndarray:
    """Score sequences by CDR-only PLL (all CDR positions, context-aware if enabled)."""
    clean_sequences = [str(seq).strip() for seq in sequences]
    if not clean_sequences:
        return np.empty(0, dtype=np.float32)

    scores = np.empty(len(clean_sequences), dtype=np.float32)
    by_len: Dict[int, List[tuple[int, str]]] = {}
    for idx, seq in enumerate(clean_sequences):
        by_len.setdefault(len(seq), []).append((idx, seq))

    with torch.no_grad():
        for _, seq_pairs in by_len.items():
            idxs = [idx for idx, _ in seq_pairs]
            seqs = [seq for _, seq in seq_pairs]
            for start in range(0, len(seqs), max(1, int(batch_size))):
                end = min(start + max(1, int(batch_size)), len(seqs))
                batch = seqs[start:end]
                batch_idxs = idxs[start:end]
                pll_scores = scorer.pseudo_log_likelihood(
                    batch,
                    cdr_only=True,
                    use_grad=False,
                )
                scores[batch_idxs] = pll_scores.detach().float().cpu().numpy()

    return scores


def compute_m22_cdr_pseudo_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    m22_df: pd.DataFrame,
    batch_size: int = 512,
) -> float:
    """Corpus-level CDR-H3 pseudo-perplexity over M22 variant library.

    For each row in `m22_df`, splice the 24-AA `aa` variant into the C05 VH
    context, mask each CDR-H3 position in turn, and compute log P(target AA).
    Aggregate as exp(-mean(log P)) across all (variant, position) pairs.
    """
    cdr_len = C05_CDRH3_END - C05_CDRH3_START
    right_flank = RIGHT_CONTEXT[: len(C05_VH) - C05_CDRH3_END]

    variants = m22_df["aa"].tolist()
    for v in variants:
        if len(v) != cdr_len:
            raise ValueError(
                f"M22 variant has length {len(v)}, expected {cdr_len}: {v!r}"
            )

    sequences: list[str] = []
    mask_positions: list[int] = []
    target_aas: list[str] = []
    for variant in variants:
        full_vh = LEFT_CONTEXT + variant + right_flank
        for i, aa in enumerate(variant):
            sequences.append(full_vh)
            mask_positions.append(C05_CDRH3_START + i)
            target_aas.append(aa)

    log_probs = compute_masked_log_probs_batch(
        model, tokenizer, sequences, mask_positions, target_aas, device,
        batch_size=batch_size,
    )
    mean_logp = float(np.mean(log_probs))
    ppl = math.exp(-mean_logp)
    logger.info(
        "M22 CDR-H3 pseudo-perplexity: %.2f (mean log P: %.4f, %d variants × %d positions)",
        ppl, mean_logp, len(variants), cdr_len,
    )
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
    """Load scoring data and subsample."""
    df = pd.read_csv(data_path)
    logger.info("Loaded %d scoring sequences from %s", len(df), data_path)

    df = df[np.isfinite(df[enrichment_col])].copy()

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed)
        logger.info("Subsampled to %d sequences", n_samples)

    df = df.reset_index(drop=True)
    return df


def _mutation_positions_per_row(df: pd.DataFrame, wt: str) -> list[set[int]]:
    """For each row, return the set of 0-based CDR positions that were mutated."""
    out = []
    for mut_str in df["mut"]:
        muts = parse_mutations(mut_str, wt)
        out.append({pos for pos, _, _ in muts})
    return out


def run_scoring_evaluation(
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    df: Optional[pd.DataFrame] = None,
    enrichment_col: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
    seed: int = 42,
    flank_ks: Sequence[int] = (),
    scorer: Optional[ESM2Model] = None,
    scoring_mode: str = "mutation_path",
) -> dict:
    """Run full scoring evaluation with both ordering strategies.

    If `flank_ks` is non-empty, also compute Spearman restricted to pairs
    where at least one mutation lies in the LEFT or RIGHT k-wide flank of
    CDRH3, for each k.
    """
    if df is None:
        raise ValueError("df must be provided.")
    if enrichment_col is None:
        raise ValueError("enrichment_col must be provided.")

    enrichment = np.asarray(df[enrichment_col].values)
    mode = str(scoring_mode).strip().lower()

    if mode == "cdr_pll":
        if scorer is None:
            raise ValueError("scoring_mode='cdr_pll' requires scorer.")
        if "aa" not in df.columns:
            raise ValueError("scoring_mode='cdr_pll' requires column 'aa' in df.")
        sequences = df["aa"].astype(str).str.strip().tolist()
        scores_avg = score_sequences_cdr_pll(
            scorer=scorer,
            sequences=sequences,
            batch_size=batch_size,
        )
        rho_avg, pval_avg = evaluate_spearman(scores_avg, enrichment)
        logger.info("Spearman (cdr_pll): rho=%.4f, p=%.2e", rho_avg, pval_avg)
        # Keep result keys stable for downstream logging/storage.
        scores_rnd = scores_avg.copy()
        rho_rnd, pval_rnd = rho_avg, pval_avg
    else:
        scores_avg = score_double_mutants(
            model, tokenizer, C05_CDRH3, df, device,
            strategy="average", batch_size=batch_size, seed=seed,
            scorer=scorer,
        )
        rho_avg, pval_avg = evaluate_spearman(scores_avg, enrichment)
        logger.info("Spearman (average ordering): rho=%.4f, p=%.2e", rho_avg, pval_avg)

        scores_rnd = score_double_mutants(
            model, tokenizer, C05_CDRH3, df, device,
            strategy="random", batch_size=batch_size, seed=seed,
            scorer=scorer,
        )
        rho_rnd, pval_rnd = evaluate_spearman(scores_rnd, enrichment)
        logger.info("Spearman (random ordering):  rho=%.4f, p=%.2e", rho_rnd, pval_rnd)

    results = {
        "spearman_avg": rho_avg,
        "spearman_avg_pval": pval_avg,
        "spearman_random": rho_rnd,
        "spearman_random_pval": pval_rnd,
        "scores_avg": scores_avg,
        "scores_random": scores_rnd,
    }

    if flank_ks:
        cdr_len = len(C05_CDRH3)
        row_muts = _mutation_positions_per_row(df, C05_CDRH3)
        for k in flank_ks:
            left_window = set(range(k))
            right_window = set(range(cdr_len - k, cdr_len))
            left_mask = np.array([bool(m & left_window) for m in row_muts])
            right_mask = np.array([bool(m & right_window) for m in row_muts])
            for side, mask in (("left", left_mask), ("right", right_mask)):
                n_sel = int(mask.sum())
                results[f"n_{side}_{k}"] = n_sel
                rho_a, p_a = evaluate_spearman(scores_avg[mask], enrichment[mask])
                rho_r, p_r = evaluate_spearman(scores_rnd[mask], enrichment[mask])
                results[f"spearman_avg_{side}{k}"] = rho_a
                results[f"spearman_avg_pval_{side}{k}"] = p_a
                results[f"spearman_random_{side}{k}"] = rho_r
                results[f"spearman_random_pval_{side}{k}"] = p_r
                logger.info(
                    "Spearman %s-%d (n=%d): avg rho=%.4f p=%.2e | random rho=%.4f p=%.2e",
                    side, k, n_sel, rho_a, p_a, rho_r, p_r,
                )

    return results


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
    model: Optional[torch.nn.Module],
    tokenizer: Optional[AutoTokenizer],
    datasets: list[tuple[str, pd.DataFrame, str]],
    device: Optional[torch.device],
    batch_size: int = 512,
    seed: int = 42,
    scores_csv_dir: str | None = None,
    flank_ks: Sequence[int] = (),
    scorer: Optional[ESM2Model] = None,
    scoring_mode: str = "mutation_path",
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
            flank_ks=flank_ks,
            scorer=scorer,
            scoring_mode=scoring_mode,
        )
        results[f"spearman_avg_{name}"] = ds_results["spearman_avg"]
        results[f"spearman_avg_pval_{name}"] = ds_results["spearman_avg_pval"]
        results[f"spearman_random_{name}"] = ds_results["spearman_random"]
        results[f"spearman_random_pval_{name}"] = ds_results["spearman_random_pval"]

        for k in flank_ks:
            for side in ("left", "right"):
                for metric in (
                    f"spearman_avg_{side}{k}",
                    f"spearman_avg_pval_{side}{k}",
                    f"spearman_random_{side}{k}",
                    f"spearman_random_pval_{side}{k}",
                ):
                    results[f"{metric}_{name}"] = ds_results[metric]
                results[f"n_{side}_{k}_{name}"] = ds_results[f"n_{side}_{k}"]

        if csv_dir is not None:
            cols = [c for c in ("aa", "mut", enrichment_col) if c in df.columns]
            out = df[cols].copy()
            out["score_avg"] = ds_results["scores_avg"]
            out["score_random"] = ds_results["scores_random"]
            out_path = csv_dir / f"{name}_scores.csv"
            out.to_csv(out_path, index=False)
            logger.info("Wrote per-pair scores: %s", out_path)

    return results
