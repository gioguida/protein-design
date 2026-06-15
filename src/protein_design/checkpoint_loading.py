from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import EsmForMaskedLM, EsmModel

from protein_design.config import ModelConfig
from protein_design.model import ESM2Model

logger = logging.getLogger(__name__)

DEFAULT_ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
_HIDDEN_SIZE_TO_MODEL_ID = {
    320: "facebook/esm2_t6_8M_UR50D",
    480: "facebook/esm2_t12_35M_UR50D",
    640: "facebook/esm2_t30_150M_UR50D",
    1280: "facebook/esm2_t33_650M_UR50D",
}
_LORA_MODULE_RE = re.compile(r"\.([^.]+)\.lora_(?:A|B|embedding_A|embedding_B)\.")


def _is_local_path_like(checkpoint: str) -> bool:
    return (
        checkpoint.startswith(("/", "./", "../", "~"))
        or checkpoint.endswith(".pt")
        or "\\" in checkpoint
        or checkpoint.count("/") > 1
    )


def _strip_model_prefix(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(raw)}")
    for key in ("policy_state_dict", "model_state_dict", "adapter_state_dict"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _is_lora_checkpoint(raw: Any, state: dict[str, torch.Tensor]) -> bool:
    if isinstance(raw, dict) and isinstance(raw.get("adapter_state_dict"), dict):
        return True
    return any("lora_" in key for key in state)


def _normalize_checkpoint_ref(checkpoint: Optional[str]) -> Optional[str]:
    if checkpoint is None:
        return None
    raw = os.path.expandvars(os.path.expanduser(str(checkpoint).strip()))
    return raw or None


def _resolve_pt_path(checkpoint: str) -> Optional[Path]:
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return p
    if p.is_dir():
        return next((p / name for name in ("best.pt", "final.pt") if (p / name).exists()), None)
    return None


def _infer_base_model_id(raw: Any, state: dict[str, torch.Tensor]) -> str:
    if isinstance(raw, dict):
        for key in ("base_model_name", "base_model_id", "esm_model_path", "model_name"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in ("base_model_name", "base_model_id", "esm_model_path", "model_name"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    candidate_dims: set[int] = set()
    for value in state.values():
        if not isinstance(value, torch.Tensor):
            continue
        for dim in value.shape:
            dim_int = int(dim)
            if dim_int in _HIDDEN_SIZE_TO_MODEL_ID:
                candidate_dims.add(dim_int)

    if not candidate_dims:
        logger.warning(
            "Could not infer base ESM2 size from checkpoint tensors; falling back to %s",
            DEFAULT_ESM2_MODEL_ID,
        )
        return DEFAULT_ESM2_MODEL_ID

    hidden_size = max(candidate_dims)
    return _HIDDEN_SIZE_TO_MODEL_ID[hidden_size]


def _infer_lora_config(raw: Any, adapter_state: dict[str, torch.Tensor]) -> LoraConfig:
    base_model_name = _infer_base_model_id(raw, adapter_state)

    lora_cfg_raw = raw.get("lora_config") if isinstance(raw, dict) else None
    if isinstance(lora_cfg_raw, dict):
        rank = int(lora_cfg_raw.get("r", 8))
        alpha = int(lora_cfg_raw.get("lora_alpha", lora_cfg_raw.get("alpha", max(1, 2 * rank))))
        target_modules = list(lora_cfg_raw.get("target_modules", []))
        dropout = float(lora_cfg_raw.get("lora_dropout", lora_cfg_raw.get("dropout", 0.0)))
        bias = str(lora_cfg_raw.get("bias", "none"))
    else:
        rank = 0
        target_modules: list[str] = []
        for key, value in adapter_state.items():
            match = _LORA_MODULE_RE.search(key)
            if match is not None:
                target_modules.append(match.group(1))
            if ".lora_A." in key and isinstance(value, torch.Tensor) and value.ndim == 2:
                rank = max(rank, int(value.shape[0]))
            if ".lora_embedding_A." in key and isinstance(value, torch.Tensor) and value.ndim >= 1:
                rank = max(rank, int(value.shape[0]))
        if rank <= 0:
            raise RuntimeError("Could not infer LoRA rank from adapter checkpoint.")
        target_modules = sorted(set(target_modules))
        if not target_modules:
            raise RuntimeError("Could not infer LoRA target modules from adapter checkpoint.")
        alpha = max(1, 2 * rank)
        dropout = 0.0
        bias = "none"
        logger.warning(
            "LoRA checkpoint is missing lora_config metadata; inferred target_modules=%s, r=%d, "
            "and assumed lora_alpha=%d.",
            target_modules,
            rank,
            alpha,
        )

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
    )
    setattr(config, "base_model_name", base_model_name)
    return config


def _load_lora_mlm(raw: Any, adapter_state: dict[str, torch.Tensor]) -> tuple[EsmForMaskedLM, str]:
    lora_config = _infer_lora_config(raw, adapter_state)
    base_model_name = str(getattr(lora_config, "base_model_name"))
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)
    peft_model = get_peft_model(base_model, lora_config)

    merged_state = peft_model.state_dict()
    normalized_adapter_state = {
        _strip_model_prefix(key): value for key, value in adapter_state.items()
    }
    merged_state.update(normalized_adapter_state)
    missing, unexpected = peft_model.load_state_dict(merged_state, strict=False)
    non_optional_missing = [
        key for key in missing if "lora_" in key or "modules_to_save" in key
    ]
    if non_optional_missing:
        raise RuntimeError(f"LoRA checkpoint missing required adapter keys: {non_optional_missing[:5]}")
    if unexpected:
        logger.warning("Ignored %d unexpected LoRA keys (e.g. %s)", len(unexpected), unexpected[:3])

    merged_model = peft_model.merge_and_unload()
    if not isinstance(merged_model, EsmForMaskedLM):
        raise TypeError(f"Expected merged LoRA model to be EsmForMaskedLM, got {type(merged_model)}")
    return merged_model, base_model_name


def _load_full_mlm(raw: Any, state: dict[str, torch.Tensor]) -> tuple[EsmForMaskedLM, str]:
    base_model_name = _infer_base_model_id(raw, state)
    model = EsmForMaskedLM.from_pretrained(base_model_name)
    normalized_state = {
        _strip_model_prefix(key): value for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    non_optional_missing = [
        key for key in missing if not key.startswith("esm.contact_head.") and "position_ids" not in key
    ]
    if non_optional_missing:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional_missing[:5]}")
    if unexpected:
        logger.warning("Ignored %d unexpected keys from checkpoint (e.g. %s)", len(unexpected), unexpected[:3])
    return model, base_model_name


def load_mlm_from_checkpoint(checkpoint: Optional[str]) -> tuple[EsmForMaskedLM, str]:
    normalized = _normalize_checkpoint_ref(checkpoint)
    if not normalized:
        return EsmForMaskedLM.from_pretrained(DEFAULT_ESM2_MODEL_ID), DEFAULT_ESM2_MODEL_ID

    p = Path(normalized)
    if p.is_dir() and ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
        return EsmForMaskedLM.from_pretrained(str(p)), str(p)

    pt_path = _resolve_pt_path(normalized)
    if pt_path is not None:
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(raw)
        if _is_lora_checkpoint(raw, state):
            return _load_lora_mlm(raw, state)
        return _load_full_mlm(raw, state)

    if _is_local_path_like(normalized):
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {normalized}")
    return EsmForMaskedLM.from_pretrained(normalized), normalized


def load_encoder_from_checkpoint(checkpoint: Optional[str]) -> tuple[EsmModel, str]:
    model, tokenizer_ref = load_mlm_from_checkpoint(checkpoint)
    return model.esm, tokenizer_ref


def load_scorer_from_checkpoint(
    checkpoint: Optional[str],
    *,
    device: str,
    use_context: bool = True,
    pll_mask_chunk_size: int = 64,
) -> ESM2Model:
    model, base_model_name = load_mlm_from_checkpoint(checkpoint)
    scorer = ESM2Model(
        ModelConfig(
            esm_model_path=base_model_name,
            device=device,
            use_context=use_context,
            pll_mask_chunk_size=pll_mask_chunk_size,
        )
    )
    scorer.model = model
    scorer.to(scorer.device)
    scorer.eval()
    return scorer
