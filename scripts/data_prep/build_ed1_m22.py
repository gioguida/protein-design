"""Build a combined ED1 (single-mutant) M22 DMS dataset from the ED2/ED5/ED811 raw pools.

Each of the ED2_M22 / ED5_M22 / ED811_M22 raw files is nominally targeted at a
different edit-distance bucket, but `num_mut` shows they each contain a spread
of edit distances (including a handful of num_mut==1 rows from off-target
library composition / sequencing). This script pools every num_mut==1 row
across all three M22 raw files into a single ED1 dataset.

ED2_M22 turns out to already be a COMPLETE single-mutant DMS scan of the C05
CDR-H3 (24 positions x 19 substitutions = 456 rows, verified below), and its
456 unique sequences are a strict superset of the num_mut==1 rows found in
ED5_M22 (64) and ED811_M22 (74). Where a sequence appears in more than one
pool, the M22_binding_enrichment_adj values disagree substantially (pairwise
correlation 0.59 for ED2-vs-ED5, -0.24 for ED2-vs-ED811 on the overlap), so
rows are NOT averaged across pools -- each sequence's row is taken from a
single, priority-ordered source pool (ED2_M22 > ED5_M22 > ED811_M22), on the
assumption that the pool targeting the lowest diversity has the most reads
(and hence the least noise) for a near-WT single mutant. A `source_pool`
column records provenance.

Output columns are the subset shared by all three raw schemas (bucket-specific
raw read-count columns like `count_ED2M22pos` are dropped, since their names
and meanings differ per pool and aren't needed downstream -- only the already
-computed `M22_binding_enrichment_adj` metric and its columns are kept).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

RAW_SOURCES = {
    # priority order: first match wins on duplicate 'aa'
    "ED2_M22": "/cluster/project/infk/krause/gguidarini/protein-design/data/raw/ED2_M22_binding_enrichment.csv",
    "ED5_M22": "/cluster/project/infk/krause/gguidarini/protein-design/data/raw/ED5_M22_binding_enrichment.csv",
    "ED811_M22": "/cluster/project/infk/krause/gguidarini/protein-design/data/raw/ED811_M22_enrichment_full.csv",
}

COMMON_COLUMNS = [
    "aa",
    "num_mut",
    "mut",
    "num_out_of_scope_muts",
    "f_pos_r1",
    "f_neg",
    "f_pre",
    "M22_enrichment_PosdivPre_adj",
    "M22_binding_enrichment_adj",
    "M22_binding_enrichment_se",
    "M22_enrichment_PosdivPre_se",
]

OUT_PATH = (
    Path(os.environ.get("PROJECT_DIR", "."))
    / "data"
    / "raw"
    / "ED1_M22_binding_enrichment_combined.csv"
)


def build() -> pd.DataFrame:
    parts = []
    seen_aa: set[str] = set()
    for pool, path in RAW_SOURCES.items():
        df = pd.read_csv(path)
        df1 = df[df["num_mut"] == 1].copy()
        df1 = df1[~df1["aa"].isin(seen_aa)]
        seen_aa.update(df1["aa"])
        df1 = df1[COMMON_COLUMNS].copy()
        df1["source_pool"] = pool
        parts.append(df1)
        print(f"{pool}: {len(df1)} new ED1 rows (cumulative unique: {len(seen_aa)})")

    combined = pd.concat(parts, ignore_index=True)

    wt_len = combined["aa"].str.len().unique()
    assert len(wt_len) == 1, f"Expected a single CDR-H3 length, got {wt_len}"
    n_expected = wt_len[0] * 19
    print(
        f"Combined ED1 dataset: {len(combined)} rows "
        f"(theoretical full single-mutant scan = {wt_len[0]} positions x 19 subs = {n_expected})"
    )
    if len(combined) != n_expected:
        print(
            f"WARNING: combined row count ({len(combined)}) != theoretical full scan "
            f"({n_expected}) -- dataset is not a complete saturation mutagenesis."
        )
    return combined


if __name__ == "__main__":
    combined = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(combined)} rows)")
    print(combined["source_pool"].value_counts())
