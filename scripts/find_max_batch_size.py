#!/usr/bin/env python
"""Sweep batch sizes to find the GPU VRAM limit for a forward+backward pass.

Run interactively on the target GPU:
    srun --gpus=1 --gres=gpumem:24g --mem=32G --pty \\
        python scripts/find_max_batch_size.py --config configs/evotuning_base.yaml

The script doubles the batch size each step until OOM, then prints a summary
table and a recommended config snippet.
"""

import argparse
import gc
import time

import torch
from transformers import AutoTokenizer

from protein_design.model import EvotuningModel
from protein_design.utils import load_config


def vram_used_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3


def vram_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / 1024**3


def try_batch(
    model: torch.nn.Module,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    use_fp16: bool,
    scaler: "torch.amp.GradScaler",
) -> tuple[bool, float, float]:
    """Attempt a forward+backward pass. Returns (success, throughput_seq_per_s, vram_gb)."""
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # Simulate a realistic MLM batch: random token ids + labels
        input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        # ~15% of tokens are masked (label=-100 means ignored in loss)
        labels = input_ids.clone()
        mask = torch.rand_like(labels, dtype=torch.float) > 0.15
        labels[mask] = -100

        model.train()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5
        ))
        scaler.update()

        elapsed = time.perf_counter() - t0
        throughput = batch_size / elapsed
        used = vram_used_gb()

        # Clean up gradients for next iteration
        for p in model.parameters():
            p.grad = None

        return True, throughput, used

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False, 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max batch size for GPU VRAM")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument(
        "--start", type=int, default=32,
        help="Starting batch size (default: 32)",
    )
    parser.add_argument(
        "--target-util", type=float, default=0.85,
        help="Target fraction of max successful batch size for recommended config (default: 0.85)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run this on a GPU node.")
        return

    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {total_vram:.1f} GB")
    print(f"Model: {config['model_name']}")
    print(f"Seq len: {config['max_seq_len']}, fp16: {config.get('fp16', False)}\n")

    model = EvotuningModel(config)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    vocab_size = tokenizer.vocab_size
    seq_len = config["max_seq_len"]
    use_fp16 = config.get("fp16", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"{'Batch':>8}  {'Throughput':>14}  {'VRAM used':>10}  {'Status':>8}")
    print("-" * 50)

    batch_size = args.start
    last_ok_batch = None
    last_ok_throughput = None

    while True:
        ok, throughput, vram = try_batch(
            model, vocab_size, batch_size, seq_len, device, use_fp16, scaler
        )
        if ok:
            print(f"{batch_size:>8}  {throughput:>12.0f}/s  {vram:>8.2f} GB  {'OK':>8}")
            last_ok_batch = batch_size
            last_ok_throughput = throughput
            batch_size *= 2
        else:
            print(f"{batch_size:>8}  {'':>14}  {'':>10}  {'OOM':>8}")
            break

    print("-" * 50)

    if last_ok_batch is None:
        print(f"\nStarting batch size {args.start} already OOM. Try a smaller --start value.")
        return

    # Recommend a batch size at target utilization (round down to power of 2)
    rec_batch = last_ok_batch
    effective_target = 512  # aim for an effective batch of ~512 sequences
    rec_accum = max(1, effective_target // rec_batch)

    print(f"\nMax batch size that fits:  {last_ok_batch}")
    print(f"Throughput at max:         {last_ok_throughput:.0f} seq/s")
    print(f"\nRecommended config (effective batch = {rec_batch * rec_accum}):")
    print(f"  batch_size: {rec_batch}")
    print(f"  gradient_accumulation_steps: {rec_accum}")
    print(f"\nNote: using the max batch size leaves little headroom.")
    print(f"For a safer run, halve the batch_size and double gradient_accumulation_steps.")


if __name__ == "__main__":
    main()
