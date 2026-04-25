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

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import C05_CDRH3, C05_CDRH3_END, C05_CDRH3_START, add_context

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_HF_IDS = {"vanilla": "facebook/esm2_t12_35M_UR50D"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-variant", required=True,
                   help="Label for the model (e.g. 'vanilla', 'evotuned').")
    p.add_argument("--checkpoint-path", default=None,
                   help="Explicit checkpoint path; if omitted, resolve via "
                        "DEFAULT_HF_IDS[variant] or treat variant as HF model ID.")
    p.add_argument("--wt-cdrh3", default=C05_CDRH3,
                   help="Starting 24-aa CDR-H3 sequence.")
    p.add_argument("--n-chains", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-path", required=True,
                   help="Path to output CSV; companion .meta.json is written alongside.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> str:
    if args.checkpoint_path:
        return args.checkpoint_path
    return DEFAULT_HF_IDS.get(args.model_variant, args.model_variant)


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

    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args)
    print(f"\n[model] variant={args.model_variant}  checkpoint={checkpoint}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = EsmForMaskedLM.from_pretrained(checkpoint).to(device).eval()
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
          f"snapshot_every={args.snapshot_every}  temperature={args.temperature}")

    for chain_id in range(args.n_chains):
        current_vh = add_context(args.wt_cdrh3)
        current_tokens = tokenize_full_vh(tokenizer, current_vh).to(device)
        snapshots.append(make_record(chain_id, 0, current_vh, args.wt_cdrh3, args.model_variant))
        t_chain = time.time()
        print(f"\n[chain {chain_id}] start  cdrh3={args.wt_cdrh3}")

        for step in range(args.n_steps):
            local_pos = random.randrange(C05_CDRH3_END - C05_CDRH3_START)
            current_vh, current_tokens, _ = gibbs_step(
                current_vh, current_tokens, local_pos,
                model=model, aa_token_ids=aa_token_ids, mask_id=mask_id,
                temperature=args.temperature, debug=False,
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
        "snapshot_every": args.snapshot_every,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    meta_path = f"{args.output_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
