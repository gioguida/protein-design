"""Stochastic beam search on the C05 CDR-H3 loop using ESM2 masked-LM conditionals.

Inputs:
    --model-variant   Label for an ESM2 checkpoint (e.g. ``vanilla``, ``evotuned``).
    --checkpoint-path Optional explicit path to an HF-compatible checkpoint dir;
                      otherwise the variant label is resolved via DEFAULT_HF_IDS or
                      treated as an HF model ID.
    --wt-cdrh3        Starting 24-aa CDR-H3 (default: ``C05_CDRH3``).
    --beam-size, --n-steps, --snapshot-every, --temperature, --seed, --device,
    --output-path     See argparse below.

Outputs:
    {output_path}            CSV: one row per snapshot per beam member with columns
                             chain_id, gibbs_step, sequence, cdrh3, n_mutations,
                             model_variant.
    {output_path}.meta.json  Run hyperparameters.

Key design decision
-------------------
ESM2 always sees the full VH (LEFT_CONTEXT + CDR-H3 + RIGHT_CONTEXT, built via
``add_context``). We only score/mutate CDR-H3 residues. For each beam sequence
at each step, we run one masked forward pass per mutable position, cache
temperature-scaled log-probabilities over the 20 amino acids, score all
single-substitution neighbors analytically, then select the next beam with
Gumbel-top-B.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import C05_CDRH3, C05_CDRH3_END, C05_CDRH3_START, add_context

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
DEFAULT_HF_IDS = {"vanilla": ESM2_MODEL_ID}
CDRH3_LEN = C05_CDRH3_END - C05_CDRH3_START


def load_dms_seed_pool(m22_path: Path, si06_path: Path, max_n: int, seed: int) -> List[str]:
    """Load CDR-H3 sequences from D2_M22 U D2_SI06 for use as starts."""
    m22 = pd.read_csv(m22_path)[["aa"]]
    si06 = pd.read_csv(si06_path)[["aa"]]
    merged = pd.concat([m22, si06], ignore_index=True).drop_duplicates(subset=["aa"])
    merged = merged[merged["aa"].astype(str).str.len() == CDRH3_LEN]
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=seed).reset_index(drop=True)
    return merged["aa"].astype(str).tolist()


def _extract_state_dict(raw) -> dict:
    """Pull model weights out of known checkpoint wrappers."""
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: str, device: torch.device) -> EsmForMaskedLM:
    """Load a .pt state dict produced by this repo's training pipeline."""
    print(f"[model] loading state-dict checkpoint {pt_path}")
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state = {}
    for k, v in state.items():
        new_state[k[len("model."):] if k.startswith("model.") else k] = v
    model = EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_optional = [m for m in missing
                    if not m.startswith("esm.contact_head.") and "position_ids" not in m]
    if non_optional:
        raise RuntimeError(f"checkpoint missing keys: {non_optional[:5]}")
    if unexpected:
        print(f"[model] ignored {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")
    return model


def load_model_and_tokenizer(checkpoint: str, device: torch.device):
    """Resolve ``checkpoint`` to (tokenizer, model) across HF and .pt layouts."""
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
        model = _load_pt_into_mlm(str(p), device)
        return tokenizer, model
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(p))
            model = EsmForMaskedLM.from_pretrained(str(p))
            return tokenizer, model
        for name in ("best.pt", "final.pt"):
            if (p / name).exists():
                tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
                model = _load_pt_into_mlm(str(p / name), device)
                return tokenizer, model
        raise FileNotFoundError(
            f"No HF weights, best.pt, or final.pt found at {checkpoint}"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = EsmForMaskedLM.from_pretrained(checkpoint)
    return tokenizer, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-variant", required=True,
                   help="Label for the model (e.g. 'vanilla', 'evotuned').")
    p.add_argument("--checkpoint-path", default=None,
                   help="Explicit checkpoint path; if omitted, resolve via "
                        "DEFAULT_HF_IDS[variant] or treat variant as HF model ID.")
    p.add_argument("--wt-cdrh3", default=C05_CDRH3,
                   help="Reference WT CDR-H3 (used for the n_mutations column "
                        "and as the start when --start-mode=wt).")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-path", required=True,
                   help="Path to output CSV; companion .meta.json is written alongside.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-mode", choices=["wt", "dms"], default="wt",
                   help="wt: start from --wt-cdrh3. "
                        "dms: start from a random CDR-H3 sampled from "
                        "D2_M22 U D2_SI06 (deterministic given --seed).")
    p.add_argument("--dms-m22-path", type=Path, default=None,
                   help="Path to D2_M22.csv (required when --start-mode=dms).")
    p.add_argument("--dms-si06-path", type=Path, default=None,
                   help="Path to D2_SI06.csv (required when --start-mode=dms).")
    p.add_argument("--max-dms-seeds", type=int, default=500,
                   help="Cap on the DMS pool size before selecting a start sequence.")
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> str:
    raw = args.checkpoint_path or DEFAULT_HF_IDS.get(args.model_variant, args.model_variant)
    return os.path.expandvars(os.path.expanduser(raw))


def tokenize_full_vh(tokenizer: AutoTokenizer, vh: str) -> torch.Tensor:
    """HF ESM tokenization following the project convention (space-separated)."""
    spaced = " ".join(list(vh))
    encoding = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    return encoding["input_ids"][0]


def decode_full_vh(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> str:
    """Decode a single-tokenized VH with BOS/EOS into residue string."""
    ids = token_ids.detach().cpu().tolist()
    aa_tokens = tokenizer.convert_ids_to_tokens(ids[1:-1])
    return "".join(aa_tokens)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def make_record(
    chain_id: int, step: int, full_vh: str, wt_cdrh3: str, model_variant: str
) -> dict:
    cdrh3 = full_vh[C05_CDRH3_START:C05_CDRH3_END]
    return {
        "chain_id": chain_id,
        "gibbs_step": step,
        "sequence": full_vh,
        "cdrh3": cdrh3,
        "n_mutations": hamming(cdrh3, wt_cdrh3),
        "model_variant": model_variant,
    }


def _position_log_probs(
    seq_tokens: torch.Tensor,
    token_pos: int,
    *,
    model: EsmForMaskedLM,
    aa_token_ids: torch.Tensor,
    mask_id: int,
    temperature: float,
) -> torch.Tensor:
    masked = seq_tokens.clone()
    masked[token_pos] = mask_id
    with torch.no_grad():
        logits = model(input_ids=masked.unsqueeze(0)).logits[0, token_pos, aa_token_ids].float()
    return torch.log_softmax(logits / temperature, dim=-1)


def _score_template(
    seq_tokens: torch.Tensor,
    mutable_token_positions: List[int],
    *,
    model: EsmForMaskedLM,
    aa_token_ids: torch.Tensor,
    aa_index_by_token: Dict[int, int],
    mask_id: int,
    temperature: float,
) -> Tuple[Dict[int, torch.Tensor], float]:
    scores: Dict[int, torch.Tensor] = {}
    pll_template = 0.0
    for token_pos in mutable_token_positions:
        log_probs = _position_log_probs(
            seq_tokens, token_pos,
            model=model,
            aa_token_ids=aa_token_ids,
            mask_id=mask_id,
            temperature=temperature,
        )
        scores[token_pos] = log_probs
        current_token = int(seq_tokens[token_pos].item())
        if current_token not in aa_index_by_token:
            raise ValueError(
                f"Token id {current_token} at mutable position {token_pos} is not a standard AA token."
            )
        pll_template += float(log_probs[aa_index_by_token[current_token]].item())
    return scores, pll_template


def stochastic_beam_search(
    seed_tokens: torch.Tensor,
    mutable_token_positions: List[int],
    *,
    model: EsmForMaskedLM,
    aa_token_ids: torch.Tensor,
    aa_index_by_token: Dict[int, int],
    mask_id: int,
    temperature: float,
    beam_size: int,
    n_steps: int,
) -> Tuple[List[List[Tuple[torch.Tensor, float]]], Dict[Tuple[int, ...], float]]:
    """Run SBS and return beam snapshots-by-step and all seen sequences with PLL."""
    init_scores, init_pll = _score_template(
        seed_tokens,
        mutable_token_positions,
        model=model,
        aa_token_ids=aa_token_ids,
        aa_index_by_token=aa_index_by_token,
        mask_id=mask_id,
        temperature=temperature,
    )
    del init_scores  # kept only to compute the seed PLL consistently

    beam: List[Tuple[torch.Tensor, float]] = [(seed_tokens.clone(), init_pll)]
    beam_history: List[List[Tuple[torch.Tensor, float]]] = [[(seed_tokens.clone(), init_pll)]]

    all_seen: Dict[Tuple[int, ...], float] = {
        tuple(seed_tokens.detach().cpu().tolist()): init_pll
    }
    aa_ids = aa_token_ids.detach().cpu().tolist()

    for _ in range(n_steps):
        candidates: Dict[Tuple[int, ...], float] = {}
        candidate_tensors: Dict[Tuple[int, ...], torch.Tensor] = {}

        for seq_tokens, _ in beam:
            position_scores, pll_template = _score_template(
                seq_tokens,
                mutable_token_positions,
                model=model,
                aa_token_ids=aa_token_ids,
                aa_index_by_token=aa_index_by_token,
                mask_id=mask_id,
                temperature=temperature,
            )
            for token_pos in mutable_token_positions:
                log_probs = position_scores[token_pos]
                current_token = int(seq_tokens[token_pos].item())
                old_idx = aa_index_by_token[current_token]
                old_term = float(log_probs[old_idx].item())
                for aa_idx, new_token in enumerate(aa_ids):
                    if aa_idx == old_idx:
                        continue
                    child = seq_tokens.clone()
                    child[token_pos] = int(new_token)
                    child_key = tuple(child.detach().cpu().tolist())
                    pll_child = pll_template - old_term + float(log_probs[aa_idx].item())
                    if child_key not in candidates or pll_child > candidates[child_key]:
                        candidates[child_key] = pll_child
                        candidate_tensors[child_key] = child

        if not candidates:
            break

        keys = list(candidates.keys())
        base_scores = torch.tensor([candidates[k] for k in keys], device=seed_tokens.device, dtype=torch.float32)
        u = torch.rand(base_scores.shape[0], device=seed_tokens.device, dtype=torch.float32)
        eps = torch.finfo(u.dtype).eps
        u = u.clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        noisy = base_scores + gumbel

        k = min(beam_size, len(keys))
        top_idx = torch.topk(noisy, k=k).indices.detach().cpu().tolist()
        beam = [(candidate_tensors[keys[i]].clone(), float(candidates[keys[i]])) for i in top_idx]
        beam_history.append([(seq.clone(), score) for seq, score in beam])

        for key, score in candidates.items():
            if key not in all_seen or score > all_seen[key]:
                all_seen[key] = score

    return beam_history, all_seen


def main() -> None:
    args = parse_args()
    print(f"[init] args: {vars(args)}")
    print(f"[init] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[init] gpu={torch.cuda.get_device_name(0)}")

    if len(args.wt_cdrh3) != C05_CDRH3_END - C05_CDRH3_START:
        raise ValueError(
            f"--wt-cdrh3 must have length {C05_CDRH3_END - C05_CDRH3_START}, "
            f"got {len(args.wt_cdrh3)}: {args.wt_cdrh3!r}"
        )
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be >= 1")
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be >= 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.start_mode == "dms":
        if args.dms_m22_path is None or args.dms_si06_path is None:
            raise ValueError("--start-mode=dms requires --dms-m22-path and --dms-si06-path")
        pool = load_dms_seed_pool(args.dms_m22_path, args.dms_si06_path,
                                  args.max_dms_seeds, args.seed)
        if not pool:
            raise ValueError(f"DMS seed pool is empty (after length-{CDRH3_LEN} filter)")
        start_cdrh3 = random.choice(pool)
        print(f"[seeds] start_mode=dms pool_size={len(pool)} selected={start_cdrh3}")
    else:
        start_cdrh3 = args.wt_cdrh3
        print(f"[seeds] start_mode=wt start={start_cdrh3}")

    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args)
    print(f"\n[model] variant={args.model_variant}  checkpoint={checkpoint}")
    t0 = time.time()
    tokenizer, model = load_model_and_tokenizer(checkpoint, device)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    aa_token_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(a) for a in STANDARD_AAS],
        device=device, dtype=torch.long,
    )
    aa_index_by_token = {int(tok): idx for idx, tok in enumerate(aa_token_ids.detach().cpu().tolist())}
    mask_id = tokenizer.mask_token_id
    print(f"[vocab] mask token id={mask_id}")
    print(f"[vocab] AA token IDs: {dict(zip(STANDARD_AAS, aa_token_ids.tolist()))}")

    wt_full_vh = add_context(args.wt_cdrh3)
    print(f"\n[wt] cdrh3={args.wt_cdrh3!r}")
    print(f"[wt] full VH length={len(wt_full_vh)}")
    print(f"[wt] full VH start[:20]={wt_full_vh[:20]!r}")
    print(f"[wt] full VH end[-20:] ={wt_full_vh[-20:]!r}")

    wt_tokens = tokenize_full_vh(tokenizer, wt_full_vh)
    print(f"[tokenize] WT full VH tokens shape={[1, wt_tokens.shape[0]]}  "
          f"BOS={wt_tokens[0].item()}  EOS={wt_tokens[-1].item()}")

    mutable_token_positions = list(range(C05_CDRH3_START + 1, C05_CDRH3_END + 1))
    sanity_pos = mutable_token_positions[0]
    sanity_log_probs = _position_log_probs(
        wt_tokens.to(device), sanity_pos,
        model=model,
        aa_token_ids=aa_token_ids,
        mask_id=mask_id,
        temperature=args.temperature,
    )
    sanity_probs = torch.softmax(sanity_log_probs, dim=-1)
    topv, topi = sanity_probs.topk(5)
    top5 = [(STANDARD_AAS[int(i)], float(v)) for v, i in zip(topv, topi)]
    print(f"[sanity] masked forward pass at token pos {sanity_pos}:")
    print(f"[sanity]   WT  CDRH3 pos 0 = {args.wt_cdrh3[0]!r}")
    print(f"[sanity]   top5            = {top5}")

    start_full_vh = add_context(start_cdrh3)
    seed_tokens = tokenize_full_vh(tokenizer, start_full_vh).to(device)

    print(f"\n[run] beam_size={args.beam_size}  n_steps={args.n_steps}  "
          f"snapshot_every={args.snapshot_every}  temperature={args.temperature}")
    t_run = time.time()
    beam_history, all_seen = stochastic_beam_search(
        seed_tokens,
        mutable_token_positions,
        model=model,
        aa_token_ids=aa_token_ids,
        aa_index_by_token=aa_index_by_token,
        mask_id=mask_id,
        temperature=args.temperature,
        beam_size=args.beam_size,
        n_steps=args.n_steps,
    )
    print(f"[run] completed in {time.time() - t_run:.1f}s  unique_seen={len(all_seen)}")

    snapshots: List[dict] = []
    for step, beam_members in enumerate(beam_history):
        if step == 0:
            should_snapshot = True
        else:
            should_snapshot = (step % args.snapshot_every == 0) or (step == args.n_steps)
        if not should_snapshot:
            continue
        for chain_id, (seq_tokens, _) in enumerate(beam_members):
            seq = decode_full_vh(tokenizer, seq_tokens)
            snapshots.append(
                make_record(chain_id, step, seq, args.wt_cdrh3, args.model_variant)
            )
        print(f"[snapshot] step={step} beam_members={len(beam_members)}")

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(snapshots)[
        ["chain_id", "gibbs_step", "sequence", "cdrh3", "n_mutations", "model_variant"]
    ]
    df.to_csv(args.output_path, index=False)
    print(f"\n[done] wrote {len(df)} snapshot rows to {args.output_path}")

    meta = {
        "model_variant": args.model_variant,
        "checkpoint_path": checkpoint,
        "wt_cdrh3": args.wt_cdrh3,
        "beam_size": args.beam_size,
        "n_steps": args.n_steps,
        "snapshot_every": args.snapshot_every,
        "temperature": args.temperature,
        "seed": args.seed,
        "start_mode": args.start_mode,
        "chain_starts": [start_cdrh3],
    }
    meta_path = f"{args.output_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
