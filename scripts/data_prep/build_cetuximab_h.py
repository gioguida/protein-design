"""Build a single-mutant heavy-chain DMS dataset from AbAgym's Cetuximab entry.

AbAgym stores mutations as (chain, site, wildtype, mutation) rows against a
PDB structure, not as full mutant sequences -- there is no "aa" column to plug
directly into the DMS-splitting/DPO pipeline like the C05 M22 datasets have.
This script builds one: for each heavy-chain (H) single-point mutation of
EGFR_2013_Cetuximab, substitute the mutant residue into the WT heavy chain
(extracted fresh from the PDB structure, CA atoms only) to get a full 220-aa
mutant sequence.

Scope: heavy chain only (559 of 1071 non-redundant mutations), matching this
project's single-chain convention (C05 DMS data only varies the heavy chain
CDR-H3; see report/evotuning.md). All heavy-chain mutations here fall within
CDR-H1/H2/H3 (sites 31-35, 50-65, 98-108) -- verified in
data/AbAgym/CETUXIMAB_DPO_GUIDE.md -- so, unlike C05 (CDR-H3 only), this
dataset spans all three heavy CDR loops.

Score sign: AbAgym's DMS_score for this dataset has negative = enhanced
binding affinity (preferred), positive = reduced affinity (dispreferred) --
see CETUXIMAB_DPO_GUIDE.md section 5. This is the OPPOSITE convention from
this project's M22 metric (positive = preferred), so this script negates it
into `neg_DMS_score` to match. The unmutated WT has an implicit DMS_score of
0 (no effect), so `wt_metric_value=0.0` for this dataset either way.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

ABAGYM_DIR = Path(
    "/cluster/project/infk/krause/mdenegri/protein-design/data/AbAgym"
)
PDB_PATH = ABAGYM_DIR / "PDB_files" / "DMS_big_table_PDB_files" / "Cetuximab_1yy9.pdb"
DATA_PATH = ABAGYM_DIR / "AbAgym_data_non-redundant.csv"
DMS_NAME = "EGFR_2013_Cetuximab"

OUT_PATH = (
    Path(os.environ.get("PROJECT_DIR", "."))
    / "data"
    / "raw"
    / "AbAgym_cetuximab_h_mutants.csv"
)

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}


def get_chain_seq(pdb_path: Path, chain_id: str) -> dict[str, str]:
    """Return {site_str: one-letter aa} for a chain, from CA atoms only."""
    seq: dict[str, str] = {}
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA" or line[21] != chain_id:
                continue
            resnum, icode, resname = line[22:26].strip(), line[26].strip(), line[17:20].strip()
            assert icode == "", f"Unexpected insertion code at site {resnum}{icode}"
            seq.setdefault(resnum, AA3TO1.get(resname, "X"))
    return seq


def build() -> pd.DataFrame:
    heavy = get_chain_seq(PDB_PATH, "H")
    wt_sites = sorted(heavy, key=int)
    wt_seq = "".join(heavy[s] for s in wt_sites)
    print(f"WT heavy chain: {len(wt_seq)} aa")

    nr = pd.read_csv(DATA_PATH)
    cet_h = nr[(nr["DMS_name"] == DMS_NAME) & (nr["chains"] == "H")].copy()
    print(f"Cetuximab H-chain single mutants: {len(cet_h)}")

    # Sanity check: CSV wildtype must match the PDB-extracted residue at every site.
    mismatches = [
        (row.site, row.wildtype, heavy.get(str(row.site)))
        for row in cet_h.itertuples()
        if heavy.get(str(row.site)) != row.wildtype
    ]
    assert not mismatches, f"WT mismatches vs PDB: {mismatches[:10]}"

    def mutate(site: int, mut_aa: str) -> str:
        idx = wt_sites.index(str(site))
        return wt_seq[:idx] + mut_aa + wt_seq[idx + 1 :]

    cet_h["aa"] = [mutate(row.site, row.mutation) for row in cet_h.itertuples()]
    cet_h["num_mut"] = 1
    # Alias to "mut" -- required (but not actually parsed) by lora_dpo/train.py's
    # test-Spearman loader (`_load_test_spearman_df`), which gates on this column's
    # presence even though the "cdr_pll" scoring mode it feeds only uses "aa".
    cet_h["mut"] = cet_h["mut_names"]
    cet_h["neg_DMS_score"] = -cet_h["DMS_score"]

    out = cet_h[
        [
            "aa",
            "num_mut",
            "mut",
            "mut_names",
            "chains",
            "site",
            "wildtype",
            "mutation",
            "DMS_score",
            "neg_DMS_score",
        ]
    ].reset_index(drop=True)

    assert out["aa"].str.len().eq(len(wt_seq)).all()
    assert out["aa"].nunique() == len(out), "Expected all mutant sequences to be unique"
    return out, wt_seq


if __name__ == "__main__":
    out, wt_seq = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(out)} rows)")
    print(f"WT heavy chain sequence:\n{wt_seq}")
    print(out["neg_DMS_score"].describe())
