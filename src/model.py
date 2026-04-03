"""ESM2 scoring utilities for DPO training and Gibbs generation.

This module provides:
1) model loading
2) sequence tokenization with optional antibody context
3) pseudo-log-likelihood (PLL) scoring for masked language modeling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from esme import ESM2
from esme.alphabet import Alphabet, tokenize


LEFT_CONTEXT = (
	"EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
)
RIGHT_CONTEXT = (
	"WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"
)


def add_context(cdr: str) -> str:
	"""Add fixed heavy-chain context around a CDR sequence."""
	return LEFT_CONTEXT + cdr + RIGHT_CONTEXT


def get_mask_token_idx(alphabet: Alphabet) -> int:
	"""Resolve mask token index across possible Alphabet API variants."""
	for attr in ("mask_idx", "mask_index", "mask_token_id"):
		if hasattr(alphabet, attr):
			return int(getattr(alphabet, attr))

	for attr in ("tok_to_idx", "token_to_idx", "stoi"):
		mapping = getattr(alphabet, attr, None)
		if isinstance(mapping, dict) and "<mask>" in mapping:
			return int(mapping["<mask>"])

	raise AttributeError("Could not find mask token index in Alphabet.")


@dataclass
class ESM2Config:
	esm_model_path: str
	device: str = "cuda"
	use_context: bool = True


class ESM2PLLScorer:
	"""Thin wrapper around ESM2 for sequence-level PLL scoring."""

	def __init__(self, config: ESM2Config):
		self.config = config
		self.device = config.device
		self.alphabet = Alphabet()
		self.mask_token_idx = get_mask_token_idx(self.alphabet)

		self.model = ESM2.from_pretrained(config.esm_model_path, device=self.device)
		self.model.eval()

	def tokenize_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
		"""Tokenize a list of sequences, adding context if configured."""
		if self.config.use_context:
			sequences = [add_context(seq) for seq in sequences]
		return tokenize(list(sequences), alphabet=self.alphabet).to(self.device)

	def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Forward pass returning token logits with shape [B, L, V]."""
		logits = self.model(tokens)
		if not isinstance(logits, torch.Tensor):
			raise TypeError(f"Expected tensor logits from model(tokens), got {type(logits)}")
		return logits

	def _cdr_positions(self, cdr_length: int) -> List[int]:
		"""Return token indices corresponding to CDR residues.

		Based on validated probe behavior where CDR starts at index 104 when context is used.
		"""
		if not self.config.use_context:
			raise ValueError("cdr_only=True requires use_context=True.")
		start = len(LEFT_CONTEXT)
		return list(range(start, start + cdr_length))

	def pseudo_log_likelihood(
		self,
		sequences: Sequence[str],
		cdr_only: bool = True,
		use_grad: bool = False,
	) -> torch.Tensor:
		"""Compute PLL for each input sequence.

		Args:
			sequences: list of CDR sequences (without context when use_context=True).
			cdr_only: if True, score only CDR positions; otherwise score all token positions.
			use_grad: set True for policy model scoring during DPO training.

		Returns:
			Tensor of shape [batch] with PLL scores.
		"""
		if len(sequences) == 0:
			raise ValueError("sequences must not be empty")

		cdr_lengths = {len(seq) for seq in sequences}
		if len(cdr_lengths) != 1:
			raise ValueError("All sequences in a batch must have the same CDR length.")
		cdr_length = next(iter(cdr_lengths))

		tokens = self.tokenize_sequences(sequences)
		batch_size, seq_len = tokens.shape

		if cdr_only:
			positions = self._cdr_positions(cdr_length)
		else:
			positions = list(range(seq_len))

		pll = torch.zeros(batch_size, device=tokens.device, dtype=torch.float32)

		grad_context = torch.enable_grad() if use_grad else torch.no_grad()
		with grad_context:
			for pos in positions:
				masked_tokens = tokens.clone()
				true_token_ids = tokens[:, pos].clone()
				masked_tokens[:, pos] = self.mask_token_idx

				logits = self.forward_logits(masked_tokens)
				log_probs = torch.log_softmax(logits[:, pos, :].float(), dim=-1)
				true_log_probs = log_probs.gather(
					1, true_token_ids.unsqueeze(1)
				).squeeze(1)

				pll = pll + true_log_probs

		return pll

	def masked_pseudo_log_likelihood(
		self,
		sequences: Sequence[str],
		mask_positions: Sequence[int],
		use_grad: bool = False,
	) -> torch.Tensor:
		"""Compute PLL for specified masked positions in each input sequence.

		Args:
			sequences: list of CDR sequences (without context when use_context=True).
			mask_positions: list of token indices to mask and score.
			use_grad: set True for policy model scoring during DPO training.

		Returns:
			Tensor of shape [batch] with PLL scores.
		"""
		if len(sequences) == 0:
			raise ValueError("sequences must not be empty")

		tokens = self.tokenize_sequences(sequences)
		batch_size, seq_len = tokens.shape

		if any(pos < 0 or pos >= seq_len for pos in mask_positions):
			raise ValueError(
				"mask_positions must be valid token indices within the sequence length."
			)

		pll = torch.zeros(batch_size, device=tokens.device, dtype=torch.float32)

		grad_context = torch.enable_grad() if use_grad else torch.no_grad()
		with grad_context:
			for pos in mask_positions:
				masked_tokens = tokens.clone()
				true_token_ids = tokens[:, pos].clone()
				masked_tokens[:, pos] = self.mask_token_idx

				logits = self.forward_logits(masked_tokens)
				log_probs = torch.log_softmax(logits[:, pos, :].float(), dim=-1)
				true_log_probs = log_probs.gather(
					1, true_token_ids.unsqueeze(1)
				).squeeze(1)

				pll = pll + true_log_probs

		return pll
        
    
    

