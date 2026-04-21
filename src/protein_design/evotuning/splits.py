"""Hash-based split assignment for FASTA corpora.

Each sequence is assigned to train/val/test deterministically by hashing
its identifier. The assignment is:
  - order-invariant (unlike torch.random_split)
  - stable across Python/library versions (SHA-256)
  - stable under corpus growth (new sequences don't change existing buckets)
  - reproducible across processes (no RNG state)

Bump `salt` to intentionally create a different split — old runs stay
reproducible as long as they keep their old salt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitConfig:
    salt: str = "oas-v1"
    train_pct: int = 90
    val_pct: int = 5
    test_pct: int = 5

    def __post_init__(self) -> None:
        total = self.train_pct + self.val_pct + self.test_pct
        if total != 100:
            raise ValueError(
                f"SplitConfig ratios must sum to 100, got {total} "
                f"({self.train_pct}/{self.val_pct}/{self.test_pct})"
            )
        for name, v in (
            ("train_pct", self.train_pct),
            ("val_pct", self.val_pct),
            ("test_pct", self.test_pct),
        ):
            if v < 0 or v > 100:
                raise ValueError(f"{name} out of range: {v}")


def split_bucket(seq_id: str, salt: str) -> int:
    """Deterministic 0..99 bucket from a stable hash of (salt, seq_id)."""
    h = hashlib.sha256(f"{salt}:{seq_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % 100


def split_for(seq_id: str, cfg: SplitConfig) -> Split:
    b = split_bucket(seq_id, cfg.salt)
    if b < cfg.train_pct:
        return "train"
    if b < cfg.train_pct + cfg.val_pct:
        return "val"
    return "test"
