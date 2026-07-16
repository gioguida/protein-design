"""OAS FASTA dataset and DataLoader utilities for ESM2 evotuning."""

import logging
from typing import Optional

import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from protein_design.evotuning.splits import Split, SplitConfig, split_for

logger = logging.getLogger(__name__)

# Masking modes that use the single-position [MASK] mechanism (PLL-aligned).
SINGLE_MASK_MODES = ("random15", "cdr", "cdr_mix")
HYBRID_MODE = "hybrid"
# Modes that draw positions from the CDR-H3 window (need `cdr_windows`).
CDR_MASK_MODES = ("cdr", "cdr_mix", HYBRID_MODE)

# How the selected position's token is replaced, orthogonal to which mode
# picked the position:
#   "always"         - always substitute [MASK] (current/default behavior).
#   "bert_80_10_10"  - literal BERT-style: 80% [MASK], 10% random amino acid,
#                       10% left as the true residue. Loss is computed against
#                       the true residue in all three cases (labels are set
#                       the same way regardless of the substitution chosen).
MASK_REPLACE_STRATEGIES = ("always", "bert_80_10_10")
_REPLACE_MASK, _REPLACE_RANDOM, _REPLACE_KEEP = 0, 1, 2
_REPLACE_PROBS = (0.8, 0.1, 0.1)
# 20 canonical amino acids (no X) — same alphabet as pssm_baseline.STANDARD_AAS.
STANDARD_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


class OASFastaDataset(Dataset):
    """Torch dataset over a pre-partitioned numpy array of ASCII sequences."""

    def __init__(self, sequences: np.ndarray, tokenizer: AutoTokenizer, max_seq_len: int) -> None:
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sequences[idx].decode("ascii"),
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        return {k: torch.tensor(v) for k, v in encoding.items()}


def _load_fasta_by_split(
    fasta_path: str, split_cfg: SplitConfig
) -> dict[Split, np.ndarray]:
    """Two-pass FASTA load that partitions sequences by hash-based split."""
    counts: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    max_lens: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    total = 0

    logger.info("Scanning %s for split assignment (salt=%r)", fasta_path, split_cfg.salt)
    for record in SeqIO.parse(fasta_path, "fasta"):
        split = split_for(record.id, split_cfg)
        L = len(record.seq)
        counts[split] += 1
        if L > max_lens[split]:
            max_lens[split] = L
        total += 1

    logger.info(
        "Scanned %d sequences; split counts: train=%d val=%d test=%d",
        total, counts["train"], counts["val"], counts["test"],
    )

    arrays: dict[Split, np.ndarray] = {
        split: np.empty(counts[split], dtype=f"S{max(max_lens[split], 1)}")
        for split in ("train", "val", "test")
    }
    cursors: dict[Split, int] = {"train": 0, "val": 0, "test": 0}

    for record in SeqIO.parse(fasta_path, "fasta"):
        split = split_for(record.id, split_cfg)
        arrays[split][cursors[split]] = str(record.seq).encode("ascii")
        cursors[split] += 1

    return arrays


def _load_fasta_seqs_ids_by_split(
    fasta_path: str, split_cfg: SplitConfig
) -> tuple[dict[Split, np.ndarray], dict[Split, np.ndarray]]:
    """Like `_load_fasta_by_split` but also returns the seq_id of each sequence.

    seq_ids are needed by the `cdr` single-mask variant to look up per-sequence
    CDR-H3 windows. Returns (sequences_by_split, ids_by_split).
    """
    counts: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    max_lens: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    max_id_lens: dict[Split, int] = {"train": 0, "val": 0, "test": 0}

    for record in SeqIO.parse(fasta_path, "fasta"):
        split = split_for(record.id, split_cfg)
        counts[split] += 1
        max_lens[split] = max(max_lens[split], len(record.seq))
        max_id_lens[split] = max(max_id_lens[split], len(record.id))

    seqs: dict[Split, np.ndarray] = {
        s: np.empty(counts[s], dtype=f"S{max(max_lens[s], 1)}")
        for s in ("train", "val", "test")
    }
    ids: dict[Split, np.ndarray] = {
        s: np.empty(counts[s], dtype=f"S{max(max_id_lens[s], 1)}")
        for s in ("train", "val", "test")
    }
    cursors: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    for record in SeqIO.parse(fasta_path, "fasta"):
        split = split_for(record.id, split_cfg)
        c = cursors[split]
        seqs[split][c] = str(record.seq).encode("ascii")
        ids[split][c] = record.id.encode("ascii")
        cursors[split] += 1

    logger.info(
        "Scanned %d sequences; split counts: train=%d val=%d test=%d",
        sum(counts.values()), counts["train"], counts["val"], counts["test"],
    )
    return seqs, ids


def load_cdr_windows(parquet_path: str) -> dict[str, tuple[int, int]]:
    """Load the CDR-H3 ± flank windows parquet -> {seq_id: (win_start, win_end)}.

    Only rows with found=True are kept (residue coordinates into the VH).
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    ok = df[df["found"]]
    windows = {
        str(sid): (int(ws), int(we))
        for sid, ws, we in zip(ok["seq_id"], ok["win_start"], ok["win_end"])
    }
    logger.info(
        "Loaded %d CDR windows from %s (%d rows, %d unresolved skipped)",
        len(windows), parquet_path, len(df), len(df) - len(ok),
    )
    return windows


class SingleMaskDataset(Dataset):
    """Single-position MLM dataset: each example masks exactly one residue.

    Every (sequence, position) pair is enumerated explicitly. `labels` is -100
    everywhere except the masked token, which carries the true residue id —
    this is exactly the per-position pseudo-log-likelihood objective used at
    eval time.

    Position selection:
      - "random15": per sequence, positions ~ Bernoulli(`mlm_probability`) over
        the whole VH, resampled each epoch via `set_epoch(epoch_seed)`.
      - "cdr" / "cdr_mix": positions are drawn from range(win_start, win_end)
        in `cdr_windows`. When `resample` is True (training), CDR-window
        positions are kept ~ Bernoulli(`cdr_mask_prob`), resampled each epoch
        via `set_epoch(epoch_seed)`; this breaks the deterministic repetition
        of identical (context, target) pairs that otherwise drives
        memorization. "cdr_mix" additionally keeps framework (non-window)
        positions ~ Bernoulli(`framework_mask_prob`), restoring the
        generalizable framework signal that regularizes "random15". When
        `resample` is False (val/test), only the full CDR window is enumerated
        deterministically (no framework) so `cdr_ppl` is a stable, comparable
        pseudo-perplexity over all CDR positions — identical for "cdr" and
        "cdr_mix" runs.

    Replacement (orthogonal to position selection above): `mask_replace_strategy`
    controls what the selected token is replaced with. "always" (default)
    always substitutes [MASK]. "bert_80_10_10" substitutes [MASK] 80% of the
    time, a random amino acid 10% of the time, and leaves the true residue in
    place 10% of the time — the label is always the true residue regardless.
    The replace-mode draw is seeded the same way position selection is (fixed
    seed for val/test, epoch_seed for train), so it's reproducible.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        seq_ids: np.ndarray,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        masking: str,
        mlm_probability: float = 0.15,
        cdr_windows: Optional[dict[str, tuple[int, int]]] = None,
        epoch_seed: int = 0,
        resample: bool = False,
        cdr_mask_prob: float = 0.5,
        framework_mask_prob: float = 0.05,
        mask_replace_strategy: str = "always",
    ) -> None:
        if masking not in SINGLE_MASK_MODES:
            raise ValueError(f"Unknown single-mask mode: {masking!r}")
        if mask_replace_strategy not in MASK_REPLACE_STRATEGIES:
            raise ValueError(f"Unknown mask_replace_strategy: {mask_replace_strategy!r}")
        self.sequences = sequences
        self.seq_ids = seq_ids
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.masking = masking
        self.mlm_probability = float(mlm_probability)
        self.cdr_mask_prob = float(cdr_mask_prob)
        self.framework_mask_prob = float(framework_mask_prob)
        self.cdr_windows = cdr_windows or {}
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.resample = bool(resample)
        self.mask_replace_strategy = mask_replace_strategy
        self._aa_token_ids = np.asarray(
            [tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA_LETTERS], dtype=np.int64,
        )
        # Max residues that fit alongside BOS/EOS.
        self._max_residues = max_seq_len - 2
        self.index: list[tuple[int, int]] = []
        self.replace_modes = np.zeros(0, dtype=np.int8)
        self.replace_aa_ids = np.zeros(0, dtype=np.int64)
        self.set_epoch(epoch_seed)

    def _assign_replace_modes(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw (replace_mode, random_aa_id) for `n` masked positions.

        replace_mode is 0=[MASK] / 1=random amino acid / 2=keep-true-residue,
        drawn ~Categorical(0.8, 0.1, 0.1). Only meaningful when
        `mask_replace_strategy == "bert_80_10_10"`; a fixed 0-fill (always
        [MASK]) is returned otherwise (cheap, keeps __getitem__ branch-free
        for the common case).
        """
        if self.mask_replace_strategy != "bert_80_10_10" or n == 0:
            return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int64)
        # Offset the seed so this draw doesn't reuse the exact same stream as
        # position selection (which also seeds from epoch_seed).
        rng = np.random.default_rng(int(seed) + 999_999_937)
        modes = rng.choice(3, size=n, p=_REPLACE_PROBS).astype(np.int8)
        random_aa = self._aa_token_ids[rng.integers(0, len(self._aa_token_ids), size=n)]
        return modes, random_aa

    def set_epoch(self, epoch_seed: int) -> None:
        """(Re)build the flattened (seq_idx, residue_pos) index.

        For "random15", and for the CDR modes when `resample` is True, the
        index is resampled each epoch from `epoch_seed`. For non-resampling CDR
        modes (val/test) the index enumerates the full window and is built once.
        """
        if self.masking == "random15":
            self._build_random15(epoch_seed)
        elif self.resample:
            self._build_cdr(epoch_seed)
        elif not self.index:
            self._build_cdr(None)

    def _build_random15(self, epoch_seed: int) -> None:
        rng = np.random.default_rng(int(epoch_seed))
        index: list[tuple[int, int]] = []
        for si in range(len(self.sequences)):
            L = min(len(self.sequences[si]), self._max_residues)
            if L <= 0:
                continue
            picks = np.nonzero(rng.random(L) < self.mlm_probability)[0]
            for p in picks:
                index.append((si, int(p)))
        self.index = index
        self.replace_modes, self.replace_aa_ids = self._assign_replace_modes(len(index), epoch_seed)

    def _build_cdr(self, epoch_seed: Optional[int] = None) -> None:
        """Enumerate CDR-window (and, for "cdr_mix", framework) positions.

        If `epoch_seed` is None, every window position is included and no
        framework positions are added (deterministic full enumeration, for
        val/test). Otherwise (training) each window position is kept
        ~ Bernoulli(`cdr_mask_prob`) with a seeded RNG, and — when masking is
        "cdr_mix" — each framework position is kept
        ~ Bernoulli(`framework_mask_prob`), so the masked positions vary across
        epochs.
        """
        rng = None if epoch_seed is None else np.random.default_rng(int(epoch_seed))
        mix = self.masking == "cdr_mix"
        index: list[tuple[int, int]] = []
        n_skipped = 0
        for si in range(len(self.sequences)):
            sid = self.seq_ids[si].decode("ascii")
            win = self.cdr_windows.get(sid)
            if win is None:
                n_skipped += 1
                continue
            L = min(len(self.sequences[si]), self._max_residues)
            start, end = win
            win_start, win_end = start, min(end, L)
            cdr_positions = list(range(win_start, win_end))
            if rng is None:
                chosen = cdr_positions
            else:
                keep = rng.random(len(cdr_positions)) < self.cdr_mask_prob
                chosen = [p for p, k in zip(cdr_positions, keep) if k]
                if mix:
                    framework = [p for p in range(L) if not (win_start <= p < win_end)]
                    fkeep = rng.random(len(framework)) < self.framework_mask_prob
                    chosen += [p for p, k in zip(framework, fkeep) if k]
            for p in chosen:
                index.append((si, int(p)))
        self.index = index
        # val/test build with epoch_seed=None (deterministic full enumeration);
        # use a fixed seed for the replace-mode draw so it's reproducible too.
        replace_seed = epoch_seed if epoch_seed is not None else 0
        self.replace_modes, self.replace_aa_ids = self._assign_replace_modes(len(index), replace_seed)
        if n_skipped:
            logger.warning(
                "cdr masking: %d/%d sequences had no CDR window and were skipped",
                n_skipped, len(self.sequences),
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        si, pos = self.index[i]
        encoding = self.tokenizer(
            self.sequences[si].decode("ascii"),
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_pos = pos + 1  # BOS offset
        labels = [-100] * len(input_ids)
        labels[token_pos] = input_ids[token_pos]
        mode = int(self.replace_modes[i]) if self.mask_replace_strategy == "bert_80_10_10" else _REPLACE_MASK
        if mode == _REPLACE_MASK:
            input_ids[token_pos] = self.mask_token_id
        elif mode == _REPLACE_RANDOM:
            input_ids[token_pos] = int(self.replace_aa_ids[i])
        # mode == _REPLACE_KEEP: leave input_ids[token_pos] as the true residue.
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SingleMaskCollator:
    """Pad a batch of single-mask examples (pads labels with -100)."""

    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8) -> None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer does not define a pad token id.")
        self.pad_id = int(pad_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        n = len(batch)
        input_ids = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].size(0)
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels[i, :L] = b["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class HybridMaskDataset(Dataset):
    """Hybrid masking: each example independently gets *either* pure
    CDR-window masking *or* pure whole-chain masking, never both.

    Matches Talaei et al.'s Hybrid policy verbatim: "The 80% and 20% ratios
    in the hybrid masking indicate the proportion of training samples
    receiving CDR or WC masking ... within each batch" — i.e. a per-example
    choice (independently for each training sample), not a per-batch-uniform
    choice and not a per-position blend. Contrast with `cdr_mix`
    (SingleMaskDataset), which blends CDR-window and framework masking
    *within* every single example.

    One example per sequence (unlike SingleMaskDataset's flattened
    (seq, pos) index, which is one example per *masked position*): each
    __getitem__ call masks a whole set of positions in a single forward
    pass, literal BERT-style 80/10/10 in both branches (labels are the true
    residue at every masked position, -100 elsewhere).

    The CDR-vs-WC choice per sequence is redrawn each epoch from a seeded
    RNG (`set_epoch`), so it's reproducible. The specific positions masked
    within whichever branch is chosen, and the 80/10/10 replacement draw,
    are resampled fresh (unseeded) on every access — same as the legacy
    whole-chain path (`DataCollatorForLanguageModeling`), which is also not
    epoch-seeded.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        seq_ids: np.ndarray,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        cdr_windows: dict[str, tuple[int, int]],
        hybrid_cdr_sample_prob: float = 0.8,
        cdr_mask_prob: float = 0.5,
        mlm_probability: float = 0.15,
        epoch_seed: int = 0,
    ) -> None:
        self.sequences = sequences
        self.seq_ids = seq_ids
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cdr_windows = cdr_windows or {}
        self.hybrid_cdr_sample_prob = float(hybrid_cdr_sample_prob)
        self.cdr_mask_prob = float(cdr_mask_prob)
        self.mlm_probability = float(mlm_probability)
        self.mask_token_id = int(tokenizer.mask_token_id)
        self._aa_token_ids = np.asarray(
            [tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA_LETTERS], dtype=np.int64,
        )
        # Max residues that fit alongside BOS/EOS.
        self._max_residues = max_seq_len - 2
        self.use_cdr = np.zeros(len(sequences), dtype=bool)
        self.set_epoch(epoch_seed)

    def set_epoch(self, epoch_seed: int) -> None:
        """Redraw the per-example CDR-vs-WC choice ~Bernoulli(hybrid_cdr_sample_prob)."""
        rng = np.random.default_rng(int(epoch_seed))
        self.use_cdr = rng.random(len(self.sequences)) < self.hybrid_cdr_sample_prob
        n_cdr_selected = int(self.use_cdr.sum())
        if n_cdr_selected:
            missing = sum(
                1 for i in np.nonzero(self.use_cdr)[0]
                if self.seq_ids[i].decode("ascii") not in self.cdr_windows
            )
            if missing:
                logger.warning(
                    "hybrid masking: %d/%d CDR-selected sequences had no CDR "
                    "window and fell back to whole-chain masking this epoch",
                    missing, n_cdr_selected,
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def _mask_positions(self, input_ids: list[int], positions: list[int]) -> list[int]:
        """Apply literal BERT 80/10/10 replacement at `positions` (token
        indices, BOS-offset already applied); return the labels array."""
        labels = [-100] * len(input_ids)
        if not positions:
            return labels
        rng = np.random.default_rng()
        modes = rng.choice(3, size=len(positions), p=_REPLACE_PROBS)
        random_aa = self._aa_token_ids[rng.integers(0, len(self._aa_token_ids), size=len(positions))]
        for p, mode, aa in zip(positions, modes, random_aa):
            labels[p] = input_ids[p]
            if mode == _REPLACE_MASK:
                input_ids[p] = self.mask_token_id
            elif mode == _REPLACE_RANDOM:
                input_ids[p] = int(aa)
            # _REPLACE_KEEP: leave input_ids[p] as the true residue.
        return labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx].decode("ascii")
        encoding = self.tokenizer(
            seq, truncation=True, max_length=self.max_seq_len, padding=False, return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        L = min(len(seq), self._max_residues)

        sid = self.seq_ids[idx].decode("ascii")
        win = self.cdr_windows.get(sid) if self.use_cdr[idx] else None
        if win is not None:
            win_start, win_end = win[0], min(win[1], L)
            candidates = list(range(win_start, win_end))
            keep_prob = self.cdr_mask_prob
        else:
            candidates = list(range(L))
            keep_prob = self.mlm_probability

        rng = np.random.default_rng()
        keep = rng.random(len(candidates)) < keep_prob if candidates else np.zeros(0, dtype=bool)
        positions = [c + 1 for c, k in zip(candidates, keep) if k]  # +1 for BOS offset

        labels = self._mask_positions(input_ids, positions)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_train_loader(
    train_dataset: Dataset,
    collator,
    batch_size: int,
    epoch_seed: int,
    skip_samples: int = 0,
) -> DataLoader:
    """Build a train DataLoader whose order is a deterministic, seeded shuffle
    of `train_dataset`, with the first `skip_samples` indices removed.

    Resumption invariant: given the same `epoch_seed` and dataset, the produced
    permutation is bit-identical across processes — so a resumed run that skips
    `samples_seen` covers exactly the unseen samples of the original run.
    """
    # Single-mask datasets resample their flattened (seq, pos) index per epoch.
    if hasattr(train_dataset, "set_epoch"):
        train_dataset.set_epoch(epoch_seed)
    g = torch.Generator()
    g.manual_seed(int(epoch_seed))
    indices = torch.randperm(len(train_dataset), generator=g).tolist()
    if skip_samples > 0:
        indices = indices[skip_samples:]
    subset = Subset(train_dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )


def make_dataloaders(
    fasta_path: str,
    max_seq_len: int,
    mlm_probability: float,
    batch_size: int,
    split_cfg: SplitConfig,
    tokenizer: Optional[AutoTokenizer] = None,
    tokenizer_name: Optional[str] = None,
    skip_samples: int = 0,
    epoch_seed: int = 0,
    masking: Optional[str] = None,
    cdr_windows_cache: Optional[str] = None,
    cdr_mask_prob: float = 0.5,
    framework_mask_prob: float = 0.05,
    mask_replace_strategy: str = "always",
    hybrid_cdr_sample_prob: float = 0.8,
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset, object, int]:
    """Build train / val / test DataLoaders from a single FASTA using
    hash-based split assignment. Splits are stable across runs given the
    same salt and sequence IDs.

    When `masking` is one of `SINGLE_MASK_MODES` ("random15" / "cdr" /
    "cdr_mix"), the single-position [MASK] mechanism (PLL-aligned) is used.
    When `masking == "hybrid"`, each example gets pure CDR-window or pure
    whole-chain masking (see `HybridMaskDataset`). Otherwise the legacy
    80/10/10 `DataCollatorForLanguageModeling` path is used.

    Returns:
        (train_loader, val_loader, test_loader, train_dataset, collator, full_train_len).
        `train_dataset` and `collator` are returned so callers can rebuild the
        train loader cheaply across epochs (via `build_train_loader`) without
        rescanning the FASTA. `full_train_len` is `len(train_dataset)` and is
        invariant under `skip_samples` — used to size the LR scheduler so the
        schedule is unaffected by where we resume.
    """
    if tokenizer is None:
        if tokenizer_name is None:
            raise ValueError("Pass either tokenizer or tokenizer_name to make_dataloaders.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if masking in SINGLE_MASK_MODES:
        seqs, ids = _load_fasta_seqs_ids_by_split(fasta_path, split_cfg)
        cdr_windows = (
            load_cdr_windows(cdr_windows_cache) if masking in CDR_MASK_MODES else None
        )
        if masking in CDR_MASK_MODES and not cdr_windows_cache:
            raise ValueError(f"masking={masking!r} requires cdr_windows_cache (parquet path).")
        common = dict(
            mlm_probability=mlm_probability, cdr_windows=cdr_windows,
            cdr_mask_prob=cdr_mask_prob, framework_mask_prob=framework_mask_prob,
            mask_replace_strategy=mask_replace_strategy,
        )
        # Train resamples per epoch; val/test use a fixed seed for comparability.
        datasets = {
            "train": SingleMaskDataset(
                seqs["train"], ids["train"], tokenizer, max_seq_len, masking,
                epoch_seed=epoch_seed, resample=True, **common,
            ),
            "val": SingleMaskDataset(
                seqs["val"], ids["val"], tokenizer, max_seq_len, masking,
                epoch_seed=0, **common,
            ),
            "test": SingleMaskDataset(
                seqs["test"], ids["test"], tokenizer, max_seq_len, masking,
                epoch_seed=0, **common,
            ),
        }
        collator = SingleMaskCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    elif masking == HYBRID_MODE:
        if not cdr_windows_cache:
            raise ValueError("masking='hybrid' requires cdr_windows_cache (parquet path).")
        seqs, ids = _load_fasta_seqs_ids_by_split(fasta_path, split_cfg)
        cdr_windows = load_cdr_windows(cdr_windows_cache)
        common = dict(
            cdr_windows=cdr_windows, hybrid_cdr_sample_prob=hybrid_cdr_sample_prob,
            cdr_mask_prob=cdr_mask_prob, mlm_probability=mlm_probability,
        )
        # Train resamples the CDR-vs-WC choice per epoch; val/test use a fixed
        # seed so which examples get which policy stays comparable across evals.
        datasets = {
            "train": HybridMaskDataset(
                seqs["train"], ids["train"], tokenizer, max_seq_len,
                epoch_seed=epoch_seed, **common,
            ),
            "val": HybridMaskDataset(
                seqs["val"], ids["val"], tokenizer, max_seq_len, epoch_seed=0, **common,
            ),
            "test": HybridMaskDataset(
                seqs["test"], ids["test"], tokenizer, max_seq_len, epoch_seed=0, **common,
            ),
        }
        collator = SingleMaskCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    else:
        arrays = _load_fasta_by_split(fasta_path, split_cfg)
        datasets = {
            split: OASFastaDataset(arrays[split], tokenizer, max_seq_len)
            for split in ("train", "val", "test")
        }
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8,
        )

    full_train_len = len(datasets["train"])
    train_loader = build_train_loader(
        train_dataset=datasets["train"],
        collator=collator,
        batch_size=batch_size,
        epoch_seed=epoch_seed,
        skip_samples=skip_samples,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )
    return train_loader, val_loader, test_loader, datasets["train"], collator, full_train_len
