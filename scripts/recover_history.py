"""Recover history.csv from W&B server after a bad resume overwrote the file.

Pulls the complete scalar history for a run from the W&B API (all steps, no
sampling), maps the wandb key names to the history.csv column names used by
RunArtifacts, and writes the result atomically.

Usage:
    uv run python scripts/recover_history.py <run_dir> [--entity E] [--project P]

Defaults:
    entity  = mdenegri-eth-z-rich
    project = protein-design-evotuning

Example:
    uv run python scripts/recover_history.py \\
        /cluster/scratch/mdenegri/protein-design/train/oas_full_evo_v1_20260504_095735
"""

import argparse
from pathlib import Path


ENTITY = "mdenegri-eth-z-rich"
PROJECT = "protein-design-evotuning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Path to the training run directory")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    wandb_id_path = run_dir / "wandb_id.txt"
    if not wandb_id_path.exists():
        raise FileNotFoundError(f"No wandb_id.txt found in {run_dir}")
    run_id = wandb_id_path.read_text().strip()
    print(f"Run dir : {run_dir}")
    print(f"W&B run : {args.entity}/{args.project}/{run_id}")

    import wandb
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{run_id}")
    print(f"Run name: {run.name}  state: {run.state}")

    # scan_history yields every logged row without sampling.
    print("Downloading full history from W&B...")
    rows = list(run.scan_history())
    print(f"  {len(rows)} rows downloaded")

    if not rows:
        raise RuntimeError("W&B returned an empty history — nothing to recover.")

    import pandas as pd

    df = pd.DataFrame(rows)

    # Use _step as the canonical step column (matches global_step in training).
    if "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    elif "step" not in df.columns:
        raise RuntimeError("Neither '_step' nor 'step' found in W&B history columns.")

    # Drop wandb-internal columns that don't belong in history.csv.
    drop_cols = [c for c in df.columns if c.startswith("_") and c != "step"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Sort by step and re-order columns: step first, rest alphabetical.
    df = df.sort_values("step").reset_index(drop=True)
    cols = ["step"] + sorted(c for c in df.columns if c != "step")
    df = df[cols]

    out_path = run_dir / "recovered_history.csv"
    tmp = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)

    print(f"\nRecovered {len(df)} rows, steps {int(df['step'].min())}–{int(df['step'].max())}")
    print(f"Written to {out_path}")

    # Quick sanity: show non-null counts for key columns.
    key_cols = [c for c in ["train/loss", "val/loss", "val/perplexity"] if c in df.columns]
    if key_cols:
        print("\nNon-null counts:")
        for c in key_cols:
            print(f"  {c}: {df[c].notna().sum()}")


if __name__ == "__main__":
    main()
