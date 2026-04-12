"""Filter enrichment datasets to distance-2 mutants and save cleaned CSVs."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path("/cluster/project/infk/krause/ssussex/flu/data")
OUT_DIR = Path(os.environ.get("PROJECT_DIR", ".")) / "datasets" / "scoring"

DATASETS = {
    "D2_M22.csv": "M22_binding_enrichment.csv",
    "D2_SI06.csv": "SI06_binding_enrichment.csv",
    "D2_exp.csv": "exp_enrichment.csv",
}


def clean(src_path: Path) -> pd.DataFrame:
    df = pd.read_csv(src_path)
    # Drop the unnamed index column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)
    # Keep only distance-2 mutants
    df = df[df["num_mut"] == 2]
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, src_name in DATASETS.items():
        df = clean(SRC_DIR / src_name)
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"{out_name}: {len(df)} rows")
