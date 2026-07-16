#!/usr/bin/env python
"""Split $SCRATCH_DIR/oas_raw/*.csv.gz into balanced shards for filter_oas.py.

Raw OAS file sizes are extremely skewed (median ~0.9MB, max ~2.2GB in the
2026-07-16 human+heavy Bulk download), and processing order is otherwise
alphabetical, so a handful of huge files can end up clustered together and
processed serially in one process -- both slow and memory-heavy (an
exact-dup hash set that grows across the whole corpus). Splitting into
several balanced shards (greedy largest-first bin-packing by file size) lets
filter_oas.py run each shard in its own process/job with bounded memory, in
parallel.

Usage:
  uv run scripts/data_prep/make_oas_shards.py [--num-shards 32]

Writes manifests to $SCRATCH_DIR/oas_shards/manifest_{i}.txt (one filename
per line, relative to $SCRATCH_DIR/oas_raw/).
"""

import argparse
import glob
import os
import sys

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-shards", type=int, default=32)
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    input_dir = os.path.join(scratch_dir, "oas_raw")
    shards_dir = os.path.join(scratch_dir, "oas_shards")
    os.makedirs(shards_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.csv.gz")))
    if not files:
        print(f"No .csv.gz files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    sized = sorted(((os.path.getsize(f), f) for f in files), reverse=True)

    K = args.num_shards
    bins: list[list[str]] = [[] for _ in range(K)]
    bin_totals = [0] * K
    # Greedy largest-first: always add the next (largest remaining) file to
    # the currently-smallest bin. Standard longest-processing-time heuristic
    # for balanced bin-packing.
    for size, path in sized:
        i = min(range(K), key=lambda j: bin_totals[j])
        bins[i].append(os.path.basename(path))
        bin_totals[i] += size

    for i, names in enumerate(bins):
        with open(os.path.join(shards_dir, f"manifest_{i}.txt"), "w") as f:
            f.write("\n".join(names) + "\n")

    print(f"Wrote {K} manifests to {shards_dir}")
    print(f"Files: {len(files)}, total size: {sum(bin_totals) / 1024**3:.1f} GB")
    print(f"Per-shard totals (GB): min={min(bin_totals)/1024**3:.2f} "
          f"max={max(bin_totals)/1024**3:.2f} mean={sum(bin_totals)/K/1024**3:.2f}")
    print(f"Per-shard file counts: min={min(len(b) for b in bins)} "
          f"max={max(len(b) for b in bins)}")


if __name__ == "__main__":
    main()
