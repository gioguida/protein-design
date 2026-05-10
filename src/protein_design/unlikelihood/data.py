"""Data utilities for unlikelihood training on ED2 enrichment variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from protein_design.dms_splitting import DEFAULT_CONFIG_PATH, dataset_spec, project_root, resolve_dataset_split


class EnrichmentSequenceDataset(Dataset):
    """Dataset of amino-acid sequences tokenized for ESM MLM training."""

    def __init__(
        self,
        sequences: List[str],
        tokenizer: AutoTokenizer,
        max_seq_len: int,
    ) -> None:
        self.sequences = [str(seq) for seq in sequences]
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        return {k: torch.tensor(v) for k, v in encoding.items()}


class CDRMaskingCollator:
    """Collator that masks only CDR residues for MLM + unlikelihood loss."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        mask_fraction: float,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_fraction = float(mask_fraction)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("Tokenizer does not define a mask token id.")
        self.mask_token_id = int(mask_id)

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        labels = torch.full_like(input_ids, fill_value=-100)
        cdr_positions = torch.full_like(input_ids, fill_value=-1)

        # Build special-token masks once per sequence.
        special_masks = []
        for row in input_ids.tolist():
            special_masks.append(
                self.tokenizer.get_special_tokens_mask(
                    row,
                    already_has_special_tokens=True,
                )
            )
        special_tokens_mask = torch.tensor(special_masks, dtype=torch.bool)

        batch_size = int(input_ids.shape[0])
        for row_idx in range(batch_size):
            valid = attention_mask[row_idx].bool() & (~special_tokens_mask[row_idx])
            candidate_indices = torch.where(valid)[0]
            if candidate_indices.numel() == 0:
                continue

            # CDR positions are 1-indexed.
            cdr_positions[row_idx, candidate_indices] = torch.arange(
                1,
                int(candidate_indices.numel()) + 1,
                dtype=cdr_positions.dtype,
            )

            n_to_mask = max(1, int(round(self.mask_fraction * int(candidate_indices.numel()))))
            n_to_mask = min(int(candidate_indices.numel()), n_to_mask)
            perm = torch.randperm(int(candidate_indices.numel()))
            masked_indices = candidate_indices[perm[:n_to_mask]]

            labels[row_idx, masked_indices] = input_ids[row_idx, masked_indices]
            input_ids[row_idx, masked_indices] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "cdr_positions": cdr_positions,
        }


def _resolve_dms_config_path(cfg: object) -> Path:
    raw = getattr(cfg.data, "dms_config", None)
    path = Path(str(raw)) if raw is not None else project_root() / DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = project_root() / path
    return path


def build_good_split_sequences(
    cfg: object,
    to_absolute_path,
) -> Dict[str, List[str]]:
    """Build train/val/test sequence lists above enrichment threshold."""
    threshold = float(getattr(cfg.data, "enrichment_threshold", 5.19))
    dms_config_path = _resolve_dms_config_path(cfg)
    dataset_key = str(getattr(cfg.data, "dpo_dataset_key", "ed2_m22"))
    spec = dataset_spec(dataset_key, dms_config_path)
    out: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        path = resolve_dataset_split(
            dataset_key,
            split,
            dms_config_path,
            force=bool(getattr(cfg.data, "force_rebuild", False)),
        )
        working = pd.read_csv(path)
        enrichment_col = spec.key_metric_col
        if enrichment_col not in working.columns:
            raise ValueError(f"Missing required column {enrichment_col!r} in {path}.")
        working[enrichment_col] = pd.to_numeric(working[enrichment_col], errors="coerce").astype(float)
        working[spec.sequence_col] = working[spec.sequence_col].astype(str).str.strip()
        working = working.dropna(subset=[enrichment_col]).copy()
        working = working[working[spec.sequence_col] != ""].copy()
        working = working[working[enrichment_col] > threshold].copy()
        out[split] = working[spec.sequence_col].astype(str).tolist()
    return out


def make_unlikelihood_dataloaders(
    cfg: object,
    tokenizer: AutoTokenizer,
    to_absolute_path,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]]]:
    """Build unlikelihood train/val/test DataLoaders."""
    sequences_by_split = build_good_split_sequences(cfg, to_absolute_path)
    max_seq_len = int(getattr(cfg.data, "max_seq_len", 256))
    mask_fraction = float(getattr(cfg.data, "mask_fraction", 0.15))
    collator = CDRMaskingCollator(
        tokenizer=tokenizer,
        mask_fraction=mask_fraction,
        pad_to_multiple_of=8,
    )

    datasets = {
        split: EnrichmentSequenceDataset(
            sequences=sequences_by_split[split],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        for split in ("train", "val", "test")
    }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collator,
    )
    return train_loader, val_loader, test_loader, sequences_by_split


def load_unwanted_lookup_json(path: Path) -> Dict[int, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    out: Dict[int, List[str]] = {}
    for position, aas in payload.items():
        out[int(position)] = [str(aa) for aa in aas]
    return out
