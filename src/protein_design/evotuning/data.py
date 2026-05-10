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


def build_train_loader(
    train_dataset: Dataset,
    collator: DataCollatorForLanguageModeling,
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
) -> tuple[DataLoader, DataLoader, DataLoader, Dataset, DataCollatorForLanguageModeling, int]:
    """Build train / val / test DataLoaders from a single FASTA using
    hash-based split assignment. Splits are stable across runs given the
    same salt and sequence IDs.

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
