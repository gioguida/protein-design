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

from .utils import (
	LEFT_CONTEXT, 
	RIGHT_CONTEXT, 
	add_context, 
	get_mask_token_idx, 
	ModelConfig
	)


class ESM2PLLScorer:
	"""Thin wrapper around ESM2 for sequence-level PLL scoring."""

	def __init__(self, config: ModelConfig):
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

	def forward_log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Forward pass returning token log-probabilities with shape [B, L, V]."""
		if hasattr(self.model, "predict_log_prob"):
			log_probs = self.model.predict_log_prob(tokens)
			if not isinstance(log_probs, torch.Tensor):
				raise TypeError(
					f"Expected tensor log_probs from model.predict_log_prob(tokens), got {type(log_probs)}"
				)
			return log_probs.float()

		logits = self.forward_logits(tokens)
		return torch.log_softmax(logits.float(), dim=-1)

	def cdr_to_token_positions(self, cdr_positions: Sequence[int]) -> List[int]:
		"""Map 0-based CDR residue indices to tokenizer token indices."""
		offset = 1 + (len(LEFT_CONTEXT) if self.config.use_context else 0)
		token_positions: List[int] = []
		for pos in cdr_positions:
			if int(pos) < 0:
				raise ValueError("cdr_positions must contain only non-negative indices.")
			token_positions.append(int(pos) + offset)
		return token_positions

	def _cdr_positions(self, cdr_length: int) -> List[int]:
		"""Return token indices corresponding to CDR residues.

		Tokenized sequences include a leading <cls> token.
		"""
		if not self.config.use_context:
			raise ValueError("cdr_only=True requires use_context=True.")
		start = 1 + len(LEFT_CONTEXT)
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

		# Validate that all sequences have the same CDR length, which is required for cdr_only=True
		cdr_lengths = {len(seq) for seq in sequences}
		if len(cdr_lengths) != 1:
			raise ValueError("All sequences in a batch must have the same CDR length.")
		# If all sequences have same length, can take the length of the first (and only) element in the set
		cdr_length = next(iter(cdr_lengths))

		# tokenize all sequences
		tokens = self.tokenize_sequences(sequences)
		batch_size, seq_len = tokens.shape

		# determine which positions to score based on cdr_only flag
		if cdr_only:
			positions = self._cdr_positions(cdr_length)
		else:
			positions = list(range(seq_len))

		pll = torch.zeros(batch_size, device=tokens.device, dtype=torch.float32)

		with torch.enable_grad() if use_grad else torch.no_grad():
			num_pos = len(positions)	# get number of positions to mask and score
			# create flat lists of batch indices and corresponding token positions to mask across the batch
			flat_batch_idx = [b for b in range(batch_size) for _ in range(num_pos)]
			num_masks = len(flat_batch_idx)
			# for each sequence in the batch, repeat the positions to mask (either CDR or all) and flatten into a single list
			flat_pos = [positions[p_idx] for _ in range(batch_size) for p_idx in range(num_pos)]
			

			if num_masks > 0:
				idx_tensor = torch.tensor(flat_batch_idx, device=tokens.device, dtype=torch.long)
				pos_tensor = torch.tensor(flat_pos, device=tokens.device, dtype=torch.long)
				
				masked_tokens = tokens[idx_tensor].clone()
				true_token_ids = masked_tokens[torch.arange(num_masks), pos_tensor].clone()
				masked_tokens[torch.arange(num_masks), pos_tensor] = self.mask_token_idx
				
				max_batch_size = 64
				all_true_log_probs = []
				for i in range(0, num_masks, max_batch_size):
					chunk_tokens = masked_tokens[i:i+max_batch_size]
					chunk_pos = pos_tensor[i:i+max_batch_size]
					chunk_true = true_token_ids[i:i+max_batch_size]

					log_probs = self.forward_log_probs(chunk_tokens)
					
					chunk_batch_idx = torch.arange(chunk_tokens.shape[0], device=tokens.device)
					all_true_log_probs.append(log_probs[chunk_batch_idx, chunk_pos, chunk_true])
					
				if all_true_log_probs:
					all_true_log_probs = torch.cat(all_true_log_probs)
					accumulated_pll = []
					for b in range(batch_size):
						b_mask = (idx_tensor == b)
						accumulated_pll.append(all_true_log_probs[b_mask].sum())
					pll = torch.stack(accumulated_pll)

		return pll

	def masked_pseudo_log_likelihood(
		self,
		sequences: Sequence[str],
		mask_positions: Sequence[int],
		use_grad: bool = False,
		positions_are_cdr: bool = False,
	) -> torch.Tensor:
		"""Compute PLL only for specified masked positions in each input sequence.

		Args:
			sequences: list of CDR sequences (without context when use_context=True).
			mask_positions: list of token indices to mask and score.
			use_grad: set True for policy model scoring during DPO training.
			positions_are_cdr: if True, mask_positions are interpreted as 0-based
				CDR residue indices and mapped to token indices.

		Returns:
			Tensor of shape [batch] with PLL scores.
		"""
		if len(sequences) == 0:
			raise ValueError("sequences must not be empty")

		mask_positions = list(mask_positions)
		if positions_are_cdr:
			mask_positions = self.cdr_to_token_positions(mask_positions)

		tokens = self.tokenize_sequences(sequences)
		batch_size, seq_len = tokens.shape

		if any(pos < 0 or pos >= seq_len for pos in mask_positions):
			raise ValueError(
				"mask_positions must be valid token indices within the sequence length."
			)

		pll = torch.zeros(batch_size, device=tokens.device, dtype=torch.float32)

		with torch.enable_grad() if use_grad else torch.no_grad():
			num_pos = len(mask_positions)
			flat_batch_idx = [b for b in range(batch_size) for _ in range(num_pos)]
			flat_pos = [mask_positions[p_idx] for _ in range(batch_size) for p_idx in range(num_pos)]
			num_masks = len(flat_batch_idx)

			if num_masks > 0:
				idx_tensor = torch.tensor(flat_batch_idx, device=tokens.device, dtype=torch.long)
				pos_tensor = torch.tensor(flat_pos, device=tokens.device, dtype=torch.long)

				masked_tokens = tokens[idx_tensor].clone()
				true_token_ids = masked_tokens[torch.arange(num_masks), pos_tensor].clone()
				masked_tokens[torch.arange(num_masks), pos_tensor] = self.mask_token_idx

				max_batch_size = 64
				all_true_log_probs = []
				for i in range(0, num_masks, max_batch_size):
					chunk_tokens = masked_tokens[i:i+max_batch_size]
					chunk_pos = pos_tensor[i:i+max_batch_size]
					chunk_true = true_token_ids[i:i+max_batch_size]

					log_probs = self.forward_log_probs(chunk_tokens)

					chunk_batch_idx = torch.arange(chunk_tokens.shape[0], device=tokens.device)
					all_true_log_probs.append(log_probs[chunk_batch_idx, chunk_pos, chunk_true])

				if all_true_log_probs:
					all_true_log_probs = torch.cat(all_true_log_probs)
					accumulated_pll = []
					for b in range(batch_size):
						b_mask = (idx_tensor == b)
						accumulated_pll.append(all_true_log_probs[b_mask].sum())
					pll = torch.stack(accumulated_pll)

		return pll
        
    
    

