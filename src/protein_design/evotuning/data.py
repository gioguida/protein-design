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
# Modes that draw positions from the CDR-H3 window (need `cdr_windows`).
CDR_MASK_MODES = ("cdr", "cdr_mix")


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
    ) -> None:
        if masking not in SINGLE_MASK_MODES:
            raise ValueError(f"Unknown single-mask mode: {masking!r}")
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
        # Max residues that fit alongside BOS/EOS.
        self._max_residues = max_seq_len - 2
        self.index: list[tuple[int, int]] = []
        self.set_epoch(epoch_seed)

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
        input_ids[token_pos] = self.mask_token_id
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
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset, object, int]:
    """Build train / val / test DataLoaders from a single FASTA using
    hash-based split assignment. Splits are stable across runs given the
    same salt and sequence IDs.

    When `masking` is one of `SINGLE_MASK_MODES` ("random15" / "cdr"), the
    single-position [MASK] mechanism (PLL-aligned) is used; otherwise the
    legacy 80/10/10 `DataCollatorForLanguageModeling` path is used.

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
