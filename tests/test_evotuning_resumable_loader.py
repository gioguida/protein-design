"""Tests for the seeded-shuffle / mid-epoch resumption invariants in evotuning."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from protein_design.evotuning.data import build_train_loader


class _IntDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def _passthrough_collator(batch: list[int]) -> list[int]:
    return list(batch)


def _loader_indices(loader) -> list[int]:
    """Extract the underlying Subset's indices in iteration order."""
    return list(loader.dataset.indices)


def test_same_seed_same_order() -> None:
    ds = _IntDataset(1000)
    a = build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=42)
    b = build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=42)
    assert _loader_indices(a) == _loader_indices(b)


def test_different_seeds_different_orders() -> None:
    ds = _IntDataset(1000)
    a = build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=1)
    b = build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=2)
    assert _loader_indices(a) != _loader_indices(b)


def test_skip_samples_truncates_correctly() -> None:
    """Resumption invariant: skip=K must yield the same tail as the un-skipped run."""
    ds = _IntDataset(1000)
    full = _loader_indices(build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=42))
    for skip in (0, 1, 100, 333, 999):
        skipped = _loader_indices(
            build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=42, skip_samples=skip)
        )
        assert skipped == full[skip:], f"skip={skip} broke the resumption invariant"


def test_skip_equal_to_len_yields_empty() -> None:
    ds = _IntDataset(64)
    loader = build_train_loader(ds, _passthrough_collator, batch_size=8, epoch_seed=7, skip_samples=64)
    assert _loader_indices(loader) == []


def test_loader_uses_no_external_rng() -> None:
    """The seed must come exclusively from epoch_seed; mutating global RNG must not change output."""
    ds = _IntDataset(200)
    torch.manual_seed(1)
    a = _loader_indices(build_train_loader(ds, _passthrough_collator, batch_size=4, epoch_seed=42))
    torch.manual_seed(99999)
    _ = torch.randn(1000)  # disturb global RNG
    b = _loader_indices(build_train_loader(ds, _passthrough_collator, batch_size=4, epoch_seed=42))
    assert a == b


def test_load_resume_checkpoint_round_trip() -> None:
    """Verify _load_resume_checkpoint correctly extracts saved fields."""
    from protein_design.evotuning.train import _load_resume_checkpoint
    import logging

    # Minimal stand-ins that satisfy the .load_state_dict / .state_dict contract.
    class _StateHolder:
        def __init__(self) -> None:
            self._sd: dict[str, Any] = {}

        def state_dict(self) -> dict[str, Any]:
            return dict(self._sd)

        def load_state_dict(self, sd: dict[str, Any]) -> None:
            self._sd = dict(sd)

    model = _StateHolder()
    optimizer = _StateHolder()
    scheduler = _StateHolder()
    scaler = _StateHolder()

    fake_ckpt = {
        "epoch": 3,
        "global_step": 12345,
        "samples_seen": 6789,
        "epoch_seed": 100 + 3,
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"step": 999},
        "scheduler_state_dict": {"last_epoch": 50},
        "scaler_state_dict": {"scale": 1024.0},
        "val_perplexity": 4.5,
        "best_val_ppl": 4.2,
    }

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        torch.save(fake_ckpt, path)
        epoch, gstep, samples, best_ppl = _load_resume_checkpoint(
            str(path), model, optimizer, scheduler, scaler, logging.getLogger("test"),
        )

    assert epoch == 3
    assert gstep == 12345
    assert samples == 6789
    assert best_ppl == 4.2
    assert model.state_dict() == {"weight": torch.tensor([1.0, 2.0])}.keys() or "weight" in model.state_dict()
    assert optimizer.state_dict() == {"step": 999}
    assert scheduler.state_dict() == {"last_epoch": 50}
    assert scaler.state_dict() == {"scale": 1024.0}


def test_load_resume_checkpoint_back_compat_no_samples_seen() -> None:
    """Older checkpoints lack 'samples_seen' / 'best_val_ppl' / 'epoch_seed' — must not crash."""
    from protein_design.evotuning.train import _load_resume_checkpoint
    import logging

    class _StateHolder:
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass

    legacy_ckpt = {
        "epoch": 1,
        "global_step": 5000,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "scaler_state_dict": {},
        "val_perplexity": 5.0,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        torch.save(legacy_ckpt, path)
        epoch, gstep, samples, best_ppl = _load_resume_checkpoint(
            str(path), _StateHolder(), _StateHolder(), _StateHolder(), _StateHolder(),
            logging.getLogger("test"),
        )
    assert epoch == 1
    assert gstep == 5000
    assert samples == 0  # missing → 0 (start of epoch)
    assert best_ppl == 5.0  # falls back to val_perplexity
