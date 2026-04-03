"""ESM2 model wrapper for evotuning (continued MLM pretraining)."""

import logging
from typing import Optional

import torch.nn as nn
from transformers import EsmForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

logger = logging.getLogger(__name__)


class EvotuningModel(nn.Module):
    """Wraps ESM2 for masked language model fine-tuning on antibody sequences."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = EsmForMaskedLM.from_pretrained(config["model_name"])
        self._freeze_layers()

    def _freeze_layers(self) -> None:
        """Freeze embedding layer and/or first N transformer layers."""
        if self.config.get("freeze_embeddings", False):
            for param in self.model.esm.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze embedding layer")

        n_freeze = self.config.get("freeze_first_n_layers", 0)
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

    def forward(
        self,
        input_ids: Optional["torch.Tensor"] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
    ) -> MaskedLMOutput:
        """Forward pass — delegates to EsmForMaskedLM which computes MLM loss internally."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
