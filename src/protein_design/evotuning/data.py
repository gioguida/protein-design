"""OAS FASTA dataset and DataLoader utilities for ESM2 evotuning."""

import logging
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


class OASFastaDataset(Dataset):
    """Torch dataset that reads antibody sequences from a FASTA file."""

    def __init__(self, fasta_path: str, tokenizer: AutoTokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        logger.info("Loading sequences from %s", fasta_path)

        # Pass 1: count sequences and find max length for tight numpy dtype
        n_seqs = 0
        max_len = 0
        with open(fasta_path) as f:
            cur_len = 0
            for line in f:
                if line.startswith(">"):
                    if n_seqs > 0:
                        max_len = max(max_len, cur_len)
                    n_seqs += 1
                    cur_len = 0
                else:
                    cur_len += len(line.rstrip())
            if n_seqs > 0:
                max_len = max(max_len, cur_len)

        # Pass 2: fill pre-allocated numpy array (CoW-safe across DataLoader workers)
        self.sequences = np.empty(n_seqs, dtype=f"S{max_len}")
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            self.sequences[i] = str(record.seq).encode("ascii")
        logger.info("Loaded %d sequences (max_len=%d, %.1f GB)",
                    n_seqs, max_len, self.sequences.nbytes / 1e9)

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


def make_dataloaders(
    fasta_path: str,
    tokenizer_name: str,
    max_seq_len: int,
    mlm_probability: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train (95%) and validation (5%) DataLoaders from a FASTA file."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = OASFastaDataset(fasta_path, tokenizer, max_seq_len)

    n_val = int(len(dataset) * 0.05)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Split: %d train, %d val", n_train, n_val)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=8,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )
    return train_loader, val_loader
