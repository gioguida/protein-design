"""Unified ESM2 wrapper: trainable masked LM + antibody-context PLL scoring."""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from protein_design.config import ModelConfig
from protein_design.constants import LEFT_CONTEXT, add_context

logger = logging.getLogger(__name__)


class ESM2Model(nn.Module):
    """ESM2 wrapper supporting both MLM fine-tuning and PLL-based scoring.

    Training path: standard HF `EsmForMaskedLM.forward(input_ids, attention_mask, labels)`
    with optional layer freezing, used by evotuning.

    Scoring path: sequence-level pseudo-log-likelihood over (optionally context-wrapped)
    antibody sequences, used by DPO and evaluation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.esm_model_path)
        mask_token_idx = self.tokenizer.mask_token_id
        if mask_token_idx is None:
            raise ValueError("Tokenizer does not define a mask token id.")
        self.mask_token_idx = int(mask_token_idx)

        self.model = EsmForMaskedLM.from_pretrained(config.esm_model_path)

        chunk_size = int(getattr(config, "pll_mask_chunk_size", 64))
        if chunk_size < 1:
            raise ValueError("pll_mask_chunk_size must be >= 1")
        self.pll_mask_chunk_size = chunk_size

        self._freeze_layers()
        self.is_peft: bool = False

        # If LoRA is requested at construction time, attach it now. Callers
        # who need to load a non-LoRA checkpoint first should construct with
        # config.lora=None and call attach_lora() afterwards.
        if getattr(config, "lora", None) is not None:
            self.attach_lora(config.lora)

        if getattr(config, "freeze_lm_head", False):
            self.freeze_lm_head()

    def attach_lora(self, spec) -> None:
        """Wrap the underlying EsmForMaskedLM with peft.LoraConfig."""
        if self.is_peft:
            raise RuntimeError("LoRA already attached to this ESM2Model.")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(spec.r),
            lora_alpha=int(spec.alpha),
            lora_dropout=float(spec.dropout),
            target_modules=list(spec.target_modules),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.is_peft = True
        logger.info(
            "Attached LoRA (r=%d, alpha=%d, target=%s)",
            spec.r, spec.alpha, list(spec.target_modules),
        )

    def freeze_lm_head(self) -> None:
        for param in self._lm_head_module().parameters():
            param.requires_grad = False
        logger.info("Froze MLM (lm) head")

    def _lm_head_module(self) -> nn.Module:
        """Return the underlying EsmForMaskedLM lm_head module, unwrapping
        PEFT if present."""
        m = self.model
        if self.is_peft:
            # peft.PeftModel -> base_model (LoraModel) -> model (EsmForMaskedLM)
            m = m.base_model.model
        return m.lm_head

    # ------------------------------------------------------------------ training

    def _freeze_layers(self) -> None:
        """Freeze embedding layer and/or first N transformer layers."""
        if getattr(self.config, "freeze_embeddings", False):
            for param in self.model.esm.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze embedding layer")

        n_freeze = int(getattr(self.config, "freeze_first_n_layers", 0))
        if n_freeze > 0:
            for layer in self.model.esm.encoder.layer[:n_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("Froze first %d transformer layers", n_freeze)

    def param_summary(self) -> dict[str, int]:
        """Return counts of trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ------------------------------------------------------------------ checkpoint I/O

    def save_state(self, path: str | Path, extra: Optional[dict] = None) -> None:
        """Save model state. With LoRA active, writes only the adapter state
        dict (small); otherwise writes the full state dict. Always tags the
        payload with `format` so load_state can dispatch."""
        if self.is_peft:
            from peft import get_peft_model_state_dict

            state = {
                "format": "peft_adapter",
                "model_state_dict": get_peft_model_state_dict(self.model),
            }
        else:
            state = {"format": "full", "model_state_dict": self.model.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, str(path))

    def load_state(self, path: str | Path) -> dict:
        """Load model state written by save_state. Returns the full ckpt dict
        (useful for reading `step`, etc.)."""
        ckpt = torch.load(str(path), map_location="cpu")
        fmt = ckpt.get("format")
        sd = ckpt["model_state_dict"]
        if fmt == "peft_adapter":
            if not self.is_peft:
                raise ValueError(
                    "Checkpoint is a PEFT adapter but this ESM2Model has no LoRA."
                )
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.model, sd)
        else:
            # Backward-compat path: existing evotuning checkpoints don't tag a
            # format and were saved as the wrapper's full state dict.
            self.load_state_dict(sd)
        return ckpt

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:
        """Forward pass — delegates to EsmForMaskedLM which computes MLM loss internally."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------ PLL path

    def tokenize_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        """Tokenize a list of sequences, adding antibody context if configured."""
        normalized = ["".join(seq.split()) for seq in sequences]
        if self.config.use_context:
            normalized = [add_context(seq) for seq in normalized]
        spaced = [" ".join(list(seq)) for seq in normalized]

        encoding = self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        return encoding["input_ids"].to(self.device)

    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning token logits with shape [B, L, V]."""
        outputs = self.model(input_ids=tokens)
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"Expected tensor logits from model, got {type(logits)}"
            )
        return logits

    def forward_log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass returning token log-probabilities with shape [B, L, V]."""
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
        """Return token indices corresponding to CDR residues."""
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
        """Compute PLL for each input sequence."""
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

        with torch.enable_grad() if use_grad else torch.no_grad():
            num_pos = len(positions)
            flat_batch_idx = [b for b in range(batch_size) for _ in range(num_pos)]
            num_masks = len(flat_batch_idx)
            flat_pos = [
                positions[p_idx]
                for _ in range(batch_size)
                for p_idx in range(num_pos)
            ]

            if num_masks > 0:
                idx_tensor = torch.tensor(
                    flat_batch_idx, device=tokens.device, dtype=torch.long
                )
                pos_tensor = torch.tensor(
                    flat_pos, device=tokens.device, dtype=torch.long
                )

                masked_tokens = tokens[idx_tensor].clone()
                true_token_ids = masked_tokens[
                    torch.arange(num_masks), pos_tensor
                ].clone()
                masked_tokens[torch.arange(num_masks), pos_tensor] = (
                    self.mask_token_idx
                )

                max_batch_size = self.pll_mask_chunk_size
                all_true_log_probs = []
                for i in range(0, num_masks, max_batch_size):
                    chunk_tokens = masked_tokens[i : i + max_batch_size]
                    chunk_pos = pos_tensor[i : i + max_batch_size]
                    chunk_true = true_token_ids[i : i + max_batch_size]

                    log_probs = self.forward_log_probs(chunk_tokens)

                    chunk_batch_idx = torch.arange(
                        chunk_tokens.shape[0], device=tokens.device
                    )
                    all_true_log_probs.append(
                        log_probs[chunk_batch_idx, chunk_pos, chunk_true]
                    )

                if all_true_log_probs:
                    all_true_log_probs = torch.cat(all_true_log_probs)
                    accumulated_pll = []
                    for b in range(batch_size):
                        b_mask = idx_tensor == b
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
        """Compute PLL only for specified masked positions in each input sequence."""
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
            flat_pos = [
                mask_positions[p_idx]
                for _ in range(batch_size)
                for p_idx in range(num_pos)
            ]
            num_masks = len(flat_batch_idx)

            if num_masks > 0:
                idx_tensor = torch.tensor(
                    flat_batch_idx, device=tokens.device, dtype=torch.long
                )
                pos_tensor = torch.tensor(
                    flat_pos, device=tokens.device, dtype=torch.long
                )

                masked_tokens = tokens[idx_tensor].clone()
                true_token_ids = masked_tokens[
                    torch.arange(num_masks), pos_tensor
                ].clone()
                masked_tokens[torch.arange(num_masks), pos_tensor] = (
                    self.mask_token_idx
                )

                max_batch_size = self.pll_mask_chunk_size
                all_true_log_probs = []
                for i in range(0, num_masks, max_batch_size):
                    chunk_tokens = masked_tokens[i : i + max_batch_size]
                    chunk_pos = pos_tensor[i : i + max_batch_size]
                    chunk_true = true_token_ids[i : i + max_batch_size]

                    log_probs = self.forward_log_probs(chunk_tokens)

                    chunk_batch_idx = torch.arange(
                        chunk_tokens.shape[0], device=tokens.device
                    )
                    all_true_log_probs.append(
                        log_probs[chunk_batch_idx, chunk_pos, chunk_true]
                    )

                if all_true_log_probs:
                    all_true_log_probs = torch.cat(all_true_log_probs)
                    accumulated_pll = []
                    for b in range(batch_size):
                        b_mask = idx_tensor == b
                        accumulated_pll.append(all_true_log_probs[b_mask].sum())
                    pll = torch.stack(accumulated_pll)

        return pll
