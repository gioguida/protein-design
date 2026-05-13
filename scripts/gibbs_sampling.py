"""Gibbs sampling on the C05 CDR-H3 loop using ESM2 masked-LM conditionals.

Inputs:
    --model-variant   Label for an ESM2 checkpoint (e.g. ``vanilla``, ``evotuned``).
    --checkpoint-path Optional explicit path to an HF-compatible checkpoint dir;
                      otherwise the variant label is resolved via DEFAULT_HF_IDS or
                      treated as an HF model ID.
    --wt-cdrh3        Starting 24-aa CDR-H3 (default: ``C05_CDRH3``).
    --n-chains, --n-steps, --snapshot-every, --temperature, --seed, --device,
    --output-path     See argparse below.

Outputs:
    {output_path}            CSV: one row per snapshot per chain with columns
                             chain_id, gibbs_step, sequence, cdrh3, n_mutations,
                             model_variant.
    {output_path}.meta.json  Run hyperparameters.

Key design decision
-------------------
ESM2 always sees the full VH (LEFT_CONTEXT + CDR-H3 + RIGHT_CONTEXT, built via
``add_context``). At each step only the target CDR-H3 position is replaced with
``<mask>``; the framework residues stay unmasked so they condition the
prediction. Logits at the masked position are restricted to the 20 standard
amino acids, divided by ``--temperature``, softmaxed, and sampled from.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import C05_CDRH3, C05_CDRH3_END, C05_CDRH3_START, add_context

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
DEFAULT_HF_IDS = {"vanilla": ESM2_MODEL_ID}
CDRH3_LEN = C05_CDRH3_END - C05_CDRH3_START


def load_dms_seed_pool(m22_path: Path, si06_path: Path, max_n: int, seed: int) -> List[str]:
    """Load CDR-H3 sequences from D2_M22 ∪ D2_SI06 for use as chain starts."""
    m22 = pd.read_csv(m22_path)[["aa"]]
    si06 = pd.read_csv(si06_path)[["aa"]]
    merged = pd.concat([m22, si06], ignore_index=True).drop_duplicates(subset=["aa"])
    merged = merged[merged["aa"].astype(str).str.len() == CDRH3_LEN]
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=seed).reset_index(drop=True)
    return merged["aa"].astype(str).tolist()


def _extract_state_dict(raw) -> dict:
    """Pull the model-weights dict out of a .pt file with one of three shapes.

    1. Evotuning / finetuning checkpoints: ``{"model_state_dict": {...}, ...}``
       with keys prefixed ``model.esm.*`` / ``model.lm_head.*`` (the
       ``ESM2Model`` wrapper).
    2. DPO checkpoints: ``{"policy_state_dict": {...}, "optimizer_state_dict":
       ..., ...}`` where ``policy_state_dict`` keys are already in HF format
       (``esm.*`` / ``lm_head.*``).
    3. A bare state dict (no wrapper).
    """
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
    """Resolve ``checkpoint`` to (tokenizer, model) supporting four shapes:
    HF model ID, HF-format dir, dir with ``best.pt``/``final.pt``, or a direct
    ``.pt`` file. Tokenizer always comes from the public ESM2 model ID — the
    .pt state dicts produced by this repo do not bundle tokenizer files.
    """
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
    # Treat as HF model ID.
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
                        "and as the chain start when --start-mode=wt).")
    p.add_argument("--n-chains", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--burn-in", type=int, default=0,
                   help="Number of initial steps to run before recording any snapshots. "
                        "Step numbering in the output CSV starts after burn-in.")
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-path", required=True,
                   help="Path to output CSV; companion .meta.json is written alongside.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-mode", choices=["wt", "dms"], default="wt",
                   help="wt: every chain starts from --wt-cdrh3. "
                        "dms: each chain starts from a random CDR-H3 sampled from "
                        "D2_M22 ∪ D2_SI06 (deterministic given --seed).")
    p.add_argument("--dms-m22-path", type=Path, default=None,
                   help="Path to D2_M22.csv (required when --start-mode=dms).")
    p.add_argument("--dms-si06-path", type=Path, default=None,
                   help="Path to D2_SI06.csv (required when --start-mode=dms).")
    p.add_argument("--max-dms-seeds", type=int, default=500,
                   help="Cap on the DMS pool size before per-chain sampling.")
    p.add_argument("--max-mutations", type=int, default=None,
                   help="If set, reject any Gibbs step that would push edit distance "
                        "from WT beyond this value. The old residue is kept instead.")
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> str:
    raw = args.checkpoint_path or DEFAULT_HF_IDS.get(args.model_variant, args.model_variant)
    return os.path.expandvars(os.path.expanduser(raw))


def tokenize_full_vh(tokenizer: AutoTokenizer, vh: str) -> torch.Tensor:
    """HF ESM tokenization following the project convention (space-separated)."""
    spaced = " ".join(list(vh))
    encoding = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    return encoding["input_ids"]


def gibbs_step(
    full_vh: str,
    tokens: torch.Tensor,
    local_pos: int,
    *,
    model: EsmForMaskedLM,
    aa_token_ids: torch.Tensor,
    mask_id: int,
    temperature: float,
    debug: bool = False,
) -> tuple[str, torch.Tensor, list[tuple[str, float]] | None]:
    """Replace position ``local_pos`` of the CDR-H3 by sampling from ESM2's masked
    conditional given the full VH context.

    ``tokens`` is the current token tensor of shape ``(1, L+2)`` matching ``full_vh``;
    the function masks one token, runs a forward pass, samples a new AA token and
    writes it back. Returns the updated (full_vh, tokens, top5).
    """
    cdr_pos = C05_CDRH3_START + local_pos
    token_pos = cdr_pos + 1

    tokens[0, token_pos] = mask_id

    with torch.no_grad():
        logits = model(input_ids=tokens).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

    pos_log_probs = log_probs[0, token_pos, aa_token_ids]
    scaled = pos_log_probs / temperature
    probs = torch.softmax(scaled, dim=-1)

    sampled_idx = int(torch.multinomial(probs, num_samples=1).item())
    new_aa = STANDARD_AAS[sampled_idx]

    tokens[0, token_pos] = aa_token_ids[sampled_idx]
    new_vh = full_vh[:cdr_pos] + new_aa + full_vh[cdr_pos + 1:]

    top5 = None
    if debug:
        topv, topi = probs.topk(5)
        top5 = [(STANDARD_AAS[int(i)], float(v)) for v, i in zip(topv, topi)]

    return new_vh, tokens, top5


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


def _do_gibbs_step(
    current_vh: str,
    current_tokens: torch.Tensor,
    *,
    args: argparse.Namespace,
    model: EsmForMaskedLM,
    aa_token_ids: torch.Tensor,
    mask_id: int,
) -> tuple[str, torch.Tensor]:
    """Run one Gibbs step with optional max-mutation rejection."""
    local_pos = random.randrange(C05_CDRH3_END - C05_CDRH3_START)
    new_vh, new_tokens, _ = gibbs_step(
        current_vh, current_tokens, local_pos,
        model=model, aa_token_ids=aa_token_ids, mask_id=mask_id,
        temperature=args.temperature, debug=False,
    )
    if args.max_mutations is not None:
        new_cdrh3 = new_vh[C05_CDRH3_START:C05_CDRH3_END]
        if hamming(new_cdrh3, args.wt_cdrh3) > args.max_mutations:
            return current_vh, current_tokens
    return new_vh, new_tokens


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
        if len(pool) >= args.n_chains:
            chain_starts = random.sample(pool, args.n_chains)
        else:
            chain_starts = [random.choice(pool) for _ in range(args.n_chains)]
        print(f"[seeds] start_mode=dms pool_size={len(pool)} chain_starts:")
        for i, cdr in enumerate(chain_starts):
            print(f"[seeds]   chain {i}: {cdr}")
    else:
        chain_starts = [args.wt_cdrh3] * args.n_chains
        print(f"[seeds] start_mode=wt all chains start from {args.wt_cdrh3}")

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
    mask_id = tokenizer.mask_token_id
    print(f"[vocab] mask token id={mask_id}")
    print(f"[vocab] AA token IDs: {dict(zip(STANDARD_AAS, aa_token_ids.tolist()))}")

    full_vh = add_context(args.wt_cdrh3)
    print(f"\n[wt] cdrh3={args.wt_cdrh3!r}")
    print(f"[wt] full VH length={len(full_vh)}")
    print(f"[wt] full VH start[:20]={full_vh[:20]!r}")
    print(f"[wt] full VH end[-20:] ={full_vh[-20:]!r}")

    wt_tokens = tokenize_full_vh(tokenizer, full_vh)
    print(f"[tokenize] WT full VH tokens shape={list(wt_tokens.shape)}  "
          f"BOS={wt_tokens[0, 0].item()}  EOS={wt_tokens[0, -1].item()}")

    expected_token_pos = C05_CDRH3_START + 0 + 1
    assert expected_token_pos == 104, expected_token_pos
    sanity_tokens = wt_tokens.clone().to(device)
    new_vh, _, top5 = gibbs_step(
        full_vh, sanity_tokens, local_pos=0,
        model=model, aa_token_ids=aa_token_ids, mask_id=mask_id,
        temperature=args.temperature, debug=True,
    )
    print(f"[sanity] gibbs_step at local_pos=0 (token pos {expected_token_pos}):")
    print(f"[sanity]   WT  CDRH3 pos 0 = {args.wt_cdrh3[0]!r}")
    print(f"[sanity]   sampled         = {new_vh[C05_CDRH3_START]!r}")
    print(f"[sanity]   top5            = {top5}")

    snapshots: list[dict] = []
    print(f"\n[run] n_chains={args.n_chains}  n_steps={args.n_steps}  "
          f"burn_in={args.burn_in}  snapshot_every={args.snapshot_every}  "
          f"temperature={args.temperature}  max_mutations={args.max_mutations}")

    for chain_id in range(args.n_chains):
        start_cdrh3 = chain_starts[chain_id]
        current_vh = add_context(start_cdrh3)
        current_tokens = tokenize_full_vh(tokenizer, current_vh).to(device)
        t_chain = time.time()
        print(f"\n[chain {chain_id}] start  cdrh3={start_cdrh3}")

        for _ in range(args.burn_in):
            current_vh, current_tokens = _do_gibbs_step(
                current_vh, current_tokens,
                args=args, model=model, aa_token_ids=aa_token_ids, mask_id=mask_id,
            )

        if args.burn_in > 0:
            cdrh3 = current_vh[C05_CDRH3_START:C05_CDRH3_END]
            print(f"[chain {chain_id}] burn-in done ({args.burn_in} steps)  "
                  f"cdrh3={cdrh3}  mut={hamming(cdrh3, args.wt_cdrh3)}")

        snapshots.append(make_record(chain_id, 0, current_vh, args.wt_cdrh3, args.model_variant))

        for step in range(args.n_steps):
            current_vh, current_tokens = _do_gibbs_step(
                current_vh, current_tokens,
                args=args, model=model, aa_token_ids=aa_token_ids, mask_id=mask_id,
            )
            steps_taken = step + 1
            is_snapshot = steps_taken % args.snapshot_every == 0 or step == args.n_steps - 1
            if is_snapshot:
                snapshots.append(make_record(chain_id, steps_taken, current_vh, args.wt_cdrh3, args.model_variant))
                cdrh3 = current_vh[C05_CDRH3_START:C05_CDRH3_END]
                print(f"[chain {chain_id}] step {steps_taken:>{len(str(args.n_steps))}}/{args.n_steps}"
                      f"  cdrh3={cdrh3}  mut={hamming(cdrh3, args.wt_cdrh3)}"
                      f"  elapsed={time.time() - t_chain:.1f}s")

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
        "n_chains": args.n_chains,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "snapshot_every": args.snapshot_every,
        "max_mutations": args.max_mutations,
        "temperature": args.temperature,
        "seed": args.seed,
        "start_mode": args.start_mode,
        "chain_starts": chain_starts,
    }
    meta_path = f"{args.output_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
