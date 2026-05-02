"""Data utilities for unlikelihood training on ED2 enrichment variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from protein_design.dpo.data_processing import build_processed_views
from protein_design.dpo.dataset import default_data_paths
from protein_design.dpo.splitting import (
    build_or_load_cluster_split_membership,
    split_membership_keys,
)


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


def _resolve_raw_and_processed_paths(cfg: object, to_absolute_path) -> Tuple[Path, Path]:
    defaults = default_data_paths()
    raw_csv = (
        defaults["raw_m22"]
        if getattr(cfg.data, "raw_csv", None) is None
        else Path(to_absolute_path(str(cfg.data.raw_csv)))
    )
    processed_dir = (
        defaults["processed_dir"]
        if getattr(cfg.data, "processed_dir", None) is None
        else Path(to_absolute_path(str(cfg.data.processed_dir)))
    )
    return raw_csv, processed_dir


def _prepare_base_dataframe(
    cfg: object,
    to_absolute_path,
) -> Tuple[pd.DataFrame, Path, Path]:
    raw_csv_path, processed_dir = _resolve_raw_and_processed_paths(cfg, to_absolute_path)
    paths = build_processed_views(
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        force=bool(getattr(cfg.data, "force_rebuild", False)),
        verbose=False,
    )
    base_csv_path = Path(paths["ed2_all"])
    base_df = pd.read_csv(base_csv_path)
    return base_df, base_csv_path, processed_dir


def build_good_split_sequences(
    cfg: object,
    to_absolute_path,
) -> Dict[str, List[str]]:
    """Build train/val/test sequence lists above enrichment threshold."""
    base_df, base_csv_path, processed_dir = _prepare_base_dataframe(cfg, to_absolute_path)
    split_cfg = getattr(cfg.data, "split", None)

    split_membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=base_csv_path,
        processed_dir=processed_dir,
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
        seed=int(cfg.seed),
        force_rebuild=bool(getattr(cfg.data, "force_rebuild", False)),
        positive_threshold=0.0,
        stratify_bins=int(getattr(split_cfg, "stratify_bins", 10)),
        hamming_distance=int(getattr(split_cfg, "hamming_distance", 1)),
    )

    split_map = dict(
        zip(
            split_membership["split_key"].astype(str),
            split_membership["split"].astype(str),
        )
    )
    row_keys = split_membership_keys(base_df).astype(str)
    working = base_df.copy()
    working["split"] = row_keys.map(split_map)

    enrichment_col = "M22_binding_enrichment_adj"
    if enrichment_col not in working.columns:
        raise ValueError(f"Missing required column {enrichment_col!r} in ED2 base dataframe.")

    working[enrichment_col] = pd.to_numeric(working[enrichment_col], errors="coerce").astype(float)
    working["aa"] = working["aa"].astype(str).str.strip()
    working = working.dropna(subset=[enrichment_col, "split"]).copy()
    working = working[working["aa"] != ""].copy()

    threshold = float(getattr(cfg.data, "enrichment_threshold", 5.19))
    working = working[working[enrichment_col] > threshold].copy()

    return {
        split: working.loc[working["split"] == split, "aa"].astype(str).tolist()
        for split in ("train", "val", "test")
    }


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
