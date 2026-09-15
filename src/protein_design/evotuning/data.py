"""Corpus loading and masking policies for ESM2 evotuning.

Sequences come from a packed corpus built by `scripts/data_prep/pack_corpus.py`,
read through memory maps so the loader's memory stays flat regardless of corpus
size. The pack also fixes which sequences exist: anything whose CDR-H3 could
not be located was dropped at packing time, so every masking policy here trains
on the same sequences in the same order.

Three policies cover the training schedule:

    wc15        whole-chain masking, 15% of residues
    cdr50       50% of CDR-H3 residues, no flank
    hybrid      per batch, 80% of samples get cdr50 and 20% get wc15
    single_pool one residue per sequence, drawn from the CDR-H3, its flanks,
                and a resampled share of the framework

The first three mask every selected position in the same forward pass, with the
80/10/10 mask/random/keep split. `single_pool` masks exactly one position and
leaves the rest of the sequence visible, which is how a point mutation is
scored at evaluation time.

Non-standard residues survive filtering as X. They are never selected for
masking under any policy, since predicting X teaches nothing and inflates
recovery accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from protein_design.evotuning.splits import Split, SplitConfig

logger = logging.getLogger(__name__)

POLICIES = ("wc15", "cdr50", "hybrid", "single_pool")
# Policies that mask a set of positions in one forward pass.
BLOCK_POLICIES = ("wc15", "cdr50", "hybrid")

_REPLACE_MASK, _REPLACE_RANDOM, _REPLACE_KEEP = 0, 1, 2
_REPLACE_PROBS = (0.8, 0.1, 0.1)
STANDARD_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
_X_BYTE = ord("X")

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}

# Step 4's pool: every CDR-H3 position and its flanks, plus this share of the
# framework, redrawn for each occurrence of the sequence.
FRAMEWORK_POOL_FRACTION = 0.3


class PackedCorpus:
    """Read-only view over a packed corpus directory."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        meta_path = self.path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No packed corpus at {self.path}. Build it with "
                f"scripts/data_prep/pack_corpus.py."
            )
        self.meta = json.loads(meta_path.read_text())
        n = int(self.meta["n_sequences"])
        self.n = n
        self.flank = int(self.meta["flank"])
        self.max_residues = int(self.meta["max_residues"])

        self.seqs = np.memmap(self.path / "seqs.u8", dtype=np.uint8, mode="r")
        self.offsets = np.fromfile(self.path / "offsets.i64", dtype=np.int64)
        self.cdr3 = np.fromfile(self.path / "cdr3.i32", dtype=np.int32).reshape(n, 2)
        self.win = np.fromfile(self.path / "win.i32", dtype=np.int32).reshape(n, 2)
        self.split = np.fromfile(self.path / "split.i8", dtype=np.int8)

        self._ids: Optional[np.memmap] = None
        self._id_offsets: Optional[np.ndarray] = None

        logger.info(
            "Packed corpus %s: %d sequences (train=%d val=%d test=%d), flank=%d, "
            "dropped at pack time: %d (%.4f%%)",
            self.path.name, n,
            self.meta["counts"]["train"], self.meta["counts"]["val"],
            self.meta["counts"]["test"], self.flank,
            self.meta["dropped"]["total"], 100.0 * self.meta["dropped"]["rate"],
        )

    def residues(self, i: int) -> np.ndarray:
        return np.asarray(self.seqs[self.offsets[i] : self.offsets[i + 1]])

    def seq_id(self, i: int) -> str:
        if self._ids is None:
            self._ids = np.memmap(self.path / "ids.u8", dtype=np.uint8, mode="r")
            self._id_offsets = np.fromfile(self.path / "id_offsets.i64", dtype=np.int64)
        lo, hi = self._id_offsets[i], self._id_offsets[i + 1]
        return bytes(self._ids[lo:hi]).decode("ascii")

    def indices_for(self, split: Split) -> np.ndarray:
        return np.nonzero(self.split == SPLIT_CODE[split])[0].astype(np.int64)

    def check_split(self, split_cfg: SplitConfig) -> None:
        """Refuse to train on a pack built under different split settings.

        The split is a hash of the sequence id, so a mismatched salt or ratio
        silently produces a different partition than every other artifact in
        the project rather than failing.
        """
        got = self.meta["split"]
        want = {
            "salt": split_cfg.salt, "train_pct": split_cfg.train_pct,
            "val_pct": split_cfg.val_pct, "test_pct": split_cfg.test_pct,
        }
        if got != want:
            raise ValueError(
                f"Packed corpus {self.path} was built with split {got}, but this "
                f"run asks for {want}. Rebuild the pack or fix the config."
            )


def build_token_lut(tokenizer: AutoTokenizer) -> np.ndarray:
    """ASCII byte -> ESM token id, so tokenizing is an array lookup.

    Verified against the tokenizer itself at build time rather than assumed.
    """
    unk = tokenizer.unk_token_id
    lut = np.full(256, unk if unk is not None else 0, dtype=np.int64)
    for letter in STANDARD_AA_LETTERS + "X":
        tid = tokenizer.convert_tokens_to_ids(letter)
        if tid is None or tid == unk:
            raise ValueError(f"Tokenizer has no id for residue {letter!r}.")
        lut[ord(letter)] = tid
    probe = "MKVLAWXCY"
    expected = tokenizer(probe, add_special_tokens=True)["input_ids"]
    got = (
        [tokenizer.cls_token_id]
        + [int(lut[ord(c)]) for c in probe]
        + [tokenizer.eos_token_id]
    )
    if expected != got:
        raise ValueError(
            f"Token lookup table disagrees with the tokenizer: {got} != {expected}"
        )
    return lut


class MaskedCorpusDataset(Dataset):
    """One training example per sequence, masked according to `policy`.

    Indexed by position in the epoch rather than by corpus index. `set_epoch`
    fixes the order and, for the hybrid policy, which slots take CDR masking;
    both are derived from the epoch seed so a resumed run reproduces them.

    The hybrid split is exact within every batch, not an independent draw per
    sequence: the paper specifies a proportion of the samples in each batch, so
    every batch carries the same mixture rather than the same mixture on
    average.
    """

    def __init__(
        self,
        corpus: PackedCorpus,
        base_indices: np.ndarray,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        policy: str,
        *,
        batch_size: int,
        mlm_probability: float = 0.15,
        cdr_mask_prob: float = 0.5,
        hybrid_cdr_frac: float = 0.8,
        framework_pool_fraction: float = FRAMEWORK_POOL_FRACTION,
        epoch_seed: int = 0,
        shuffle: bool = False,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"Unknown masking policy {policy!r}; expected one of {POLICIES}.")
        self.corpus = corpus
        self.base_indices = np.asarray(base_indices, dtype=np.int64)
        self.tokenizer = tokenizer
        self.policy = policy
        self.batch_size = int(batch_size)
        self.mlm_probability = float(mlm_probability)
        self.cdr_mask_prob = float(cdr_mask_prob)
        self.hybrid_cdr_frac = float(hybrid_cdr_frac)
        self.framework_pool_fraction = float(framework_pool_fraction)

        self.max_residues = min(max_seq_len - 2, corpus.max_residues)
        self.lut = build_token_lut(tokenizer)
        self.cls_id = int(tokenizer.cls_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        self._aa_ids = np.asarray(
            [tokenizer.convert_tokens_to_ids(a) for a in STANDARD_AA_LETTERS], dtype=np.int64
        )

        self.epoch_seed = epoch_seed
        self.order = self.base_indices
        self.use_cdr = np.zeros(0, dtype=bool)
        # Where this view starts within the epoch. Non-zero after a resume, so
        # that a sequence keeps the masking draw it would have had in the
        # original run rather than the one belonging to its new slot.
        self.pos_offset = 0
        self.set_epoch(epoch_seed, shuffle=shuffle)

    def set_epoch(self, epoch_seed: int, skip_samples: int = 0, shuffle: bool = True) -> None:
        """Fix this epoch's order and hybrid assignment.

        `skip_samples` drops the leading samples of the epoch on resume. It must
        be a whole number of batches, otherwise the batch boundaries move and
        the hybrid mixture no longer matches the original run.
        """
        if skip_samples % self.batch_size:
            raise ValueError(
                f"skip_samples={skip_samples} is not a multiple of batch_size="
                f"{self.batch_size}; batch boundaries would shift on resume."
            )
        self.epoch_seed = int(epoch_seed)
        if shuffle:
            rng = np.random.default_rng(self.epoch_seed)
            order = self.base_indices[rng.permutation(len(self.base_indices))]
        else:
            order = self.base_indices
        if self.policy == "hybrid":
            self.use_cdr = self._hybrid_assignment(len(order))
        else:
            self.use_cdr = np.zeros(len(order), dtype=bool)
        self.pos_offset = int(skip_samples)
        if skip_samples:
            order = order[skip_samples:]
            self.use_cdr = self.use_cdr[skip_samples:]
        self.order = order

    def _hybrid_assignment(self, n: int) -> np.ndarray:
        """Exactly `round(frac * batch_size)` CDR slots in every full batch.

        The order is already a random permutation, so taking the leading slots
        of each batch is a uniform choice among that batch's sequences.
        """
        b = self.batch_size
        k = int(round(self.hybrid_cdr_frac * b))
        pattern = np.zeros(b, dtype=bool)
        pattern[:k] = True
        reps = int(np.ceil(n / b))
        return np.tile(pattern, reps)[:n]

    def __len__(self) -> int:
        return len(self.order)

    # -- position selection -------------------------------------------------

    def _maskable(self, residues: np.ndarray) -> np.ndarray:
        """Positions eligible for masking: in context, and not an X."""
        return residues[: self.max_residues] != _X_BYTE

    def _block_positions(self, i: int, residues: np.ndarray, use_cdr: bool, rng) -> np.ndarray:
        eligible = self._maskable(residues)
        L = len(eligible)
        if self.policy == "cdr50" or (self.policy == "hybrid" and use_cdr):
            start, end = int(self.corpus.cdr3[i, 0]), min(int(self.corpus.cdr3[i, 1]), L)
            candidates = np.arange(start, end)
            keep_prob = self.cdr_mask_prob
        else:
            candidates = np.arange(L)
            keep_prob = self.mlm_probability
        candidates = candidates[eligible[candidates]]
        if len(candidates) == 0:
            return candidates
        return candidates[rng.random(len(candidates)) < keep_prob]

    def _pool_position(self, i: int, residues: np.ndarray, rng) -> np.ndarray:
        """Step 4's pool, then one position drawn uniformly from it.

        The pool is the CDR-H3 with its fixed flanks, plus a share of the
        framework resampled for this occurrence of the sequence.
        """
        eligible = self._maskable(residues)
        L = len(eligible)
        win_start, win_end = int(self.corpus.win[i, 0]), min(int(self.corpus.win[i, 1]), L)
        window = np.arange(win_start, win_end)
        framework = np.concatenate([np.arange(0, win_start), np.arange(win_end, L)])
        window = window[eligible[window]]
        framework = framework[eligible[framework]]
        n_fw = int(round(self.framework_pool_fraction * len(framework)))
        if n_fw > 0:
            framework = rng.choice(framework, size=min(n_fw, len(framework)), replace=False)
        else:
            framework = framework[:0]
        pool = np.concatenate([window, framework])
        if len(pool) == 0:
            return pool
        return pool[rng.integers(0, len(pool), size=1)]

    # -- example construction -----------------------------------------------

    def __getitem__(self, pos: int) -> dict[str, torch.Tensor]:
        i = int(self.order[pos])
        residues = self.corpus.residues(i)[: self.max_residues]
        # Redrawn for each occurrence, but seeded by (epoch, slot) so a rerun
        # of the same epoch produces the same masking.
        rng = np.random.default_rng([self.epoch_seed, pos + self.pos_offset])

        if self.policy == "single_pool":
            positions = self._pool_position(i, residues, rng)
        else:
            use_cdr = bool(self.use_cdr[pos]) if len(self.use_cdr) else False
            positions = self._block_positions(i, residues, use_cdr, rng)

        n = len(residues)
        input_ids = np.empty(n + 2, dtype=np.int64)
        input_ids[0] = self.cls_id
        input_ids[1 : n + 1] = self.lut[residues]
        input_ids[n + 1] = self.eos_id
        labels = np.full(n + 2, -100, dtype=np.int64)

        if len(positions):
            tok = positions + 1  # shift past the BOS token
            labels[tok] = input_ids[tok]
            modes = rng.choice(3, size=len(tok), p=_REPLACE_PROBS)
            input_ids[tok[modes == _REPLACE_MASK]] = self.mask_id
            rand = tok[modes == _REPLACE_RANDOM]
            if len(rand):
                input_ids[rand] = self._aa_ids[rng.integers(0, len(self._aa_ids), size=len(rand))]
            # _REPLACE_KEEP leaves the true residue in place.

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.ones(n + 2, dtype=torch.long),
            "labels": torch.from_numpy(labels),
        }


class PadCollator:
    """Pad a batch to a common length, padding labels with -100."""

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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_train_loader(
    train_dataset: MaskedCorpusDataset,
    collator: PadCollator,
    batch_size: int,
    epoch_seed: int,
    skip_samples: int = 0,
) -> DataLoader:
    """Build the train loader for one epoch.

    Given the same epoch seed and corpus the order is identical across
    processes, so a resumed run that skips the samples it already saw covers
    exactly the rest of the epoch.
    """
    train_dataset.set_epoch(epoch_seed, skip_samples=skip_samples, shuffle=True)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )


def make_dataloaders(
    pack_path: str,
    max_seq_len: int,
    mlm_probability: float,
    batch_size: int,
    split_cfg: SplitConfig,
    policy: str,
    tokenizer: Optional[AutoTokenizer] = None,
    tokenizer_name: Optional[str] = None,
    skip_samples: int = 0,
    epoch_seed: int = 0,
    cdr_mask_prob: float = 0.5,
    hybrid_cdr_frac: float = 0.8,
    subsample_n: Optional[int] = None,
    subsample_seed: int = 0,
    val_max_sequences: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, MaskedCorpusDataset, PadCollator, int, PackedCorpus]:
    """Build train/val/test loaders over a packed corpus.

    `subsample_n` shrinks the training split only, deterministically, for the
    learning-rate sweep. `val_max_sequences` caps the validation split so the
    in-loop metrics cost the same on any corpus.

    Returns the loaders, plus the train dataset and collator so the train
    loader can be rebuilt cheaply per epoch, the full training length (which
    ignores `skip_samples`, so the learning-rate schedule is unaffected by
    where a run resumes), and the corpus itself.
    """
    if tokenizer is None:
        if tokenizer_name is None:
            raise ValueError("Pass either tokenizer or tokenizer_name to make_dataloaders.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    corpus = PackedCorpus(pack_path)
    corpus.check_split(split_cfg)

    train_idx = corpus.indices_for("train")
    val_idx = corpus.indices_for("val")
    test_idx = corpus.indices_for("test")

    if subsample_n is not None and subsample_n < len(train_idx):
        rng = np.random.default_rng(subsample_seed)
        train_idx = np.sort(rng.choice(train_idx, size=subsample_n, replace=False))
        logger.info(
            "Training split subsampled to %d sequences (seed=%d)", subsample_n, subsample_seed
        )
    if val_max_sequences is not None and val_max_sequences < len(val_idx):
        rng = np.random.default_rng(subsample_seed)
        val_idx = np.sort(rng.choice(val_idx, size=val_max_sequences, replace=False))
        logger.info("Validation split capped at %d sequences", val_max_sequences)

    common = dict(
        tokenizer=tokenizer, max_seq_len=max_seq_len, policy=policy,
        batch_size=batch_size, mlm_probability=mlm_probability,
        cdr_mask_prob=cdr_mask_prob, hybrid_cdr_frac=hybrid_cdr_frac,
    )
    train_dataset = MaskedCorpusDataset(
        corpus, train_idx, epoch_seed=epoch_seed, shuffle=True, **common
    )
    val_dataset = MaskedCorpusDataset(corpus, val_idx, epoch_seed=0, shuffle=False, **common)
    test_dataset = MaskedCorpusDataset(corpus, test_idx, epoch_seed=0, shuffle=False, **common)

    collator = PadCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    full_train_len = len(train_idx)

    train_loader = build_train_loader(
        train_dataset, collator, batch_size, epoch_seed, skip_samples=skip_samples
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=collator,
    )
    logger.info(
        "Policy %r: train=%d val=%d test=%d sequences",
        policy, full_train_len, len(val_idx), len(test_idx),
    )
    return train_loader, val_loader, test_loader, train_dataset, collator, full_train_len, corpus
