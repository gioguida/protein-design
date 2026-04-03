"""
OAS (Observed Antibody Space) Explorer
---------------------------------------
Explores OAS data units directly from URLs — no full download needed.
Focuses on human IgG heavy chains relevant for C05 antibody evotuning.

Usage:
    pip install pandas requests tqdm
    python explore_oas.py
"""

import json
import pandas as pd
import requests
from io import BytesIO
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — swap these URLs to explore different data units
# Format: http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/<Study>/<file>.csv.gz
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_UNITS = {
    "human_IgG_heavy": (
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
        "Greiff_2017/csv/ERR1759628_Heavy_IGHG.csv.gz"
    ),
    "human_IgM_heavy": (
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
        "Greiff_2017/csv/ERR1759628_Heavy_IGHM.csv.gz"
    ),
}

# Columns we actually care about for evotuning / project analysis
PROJECT_COLS = [
    "sequence_alignment_aa",   # full VH variable domain AA sequence → ESM2 input
    "cdr3_aa",                 # CDRH3 amino acid sequence
    "ANARCI_status",           # quality flags (truncations, indels, unusual residues)
    "Redundancy",              # how many times this sequence was seen in the study
    "productive",              # is this a productive (non-pseudogene) rearrangement?
    "v_call",                  # V germline gene (e.g. IGHV3-30)
    "j_call",                  # J germline gene (e.g. IGHJ4)
    "v_identity",              # % identity to V germline (proxy for somatic hypermutation)
    "j_identity",
    "germline_alignment_aa",   # inferred germline sequence (pre-SHM)
    "junction_aa",             # junction region (CDR3 + anchors), IMGT definition
]

DIVIDER = "=" * 70


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def load_data_unit(url: str) -> tuple[dict, pd.DataFrame]:
    """
    Load an OAS data unit from a URL.
    Returns (metadata_dict, sequences_dataframe).
    OAS stores JSON metadata in the first row (as column names),
    actual sequences start from row index 1.
    """
    print(f"  Fetching: {url.split('unpaired/')[-1]}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    buf = BytesIO(resp.content)

    # Row 0 = metadata encoded as column header JSON
    meta_row = pd.read_csv(buf, nrows=0, compression="gzip")
    metadata = json.loads(",".join(meta_row.columns))

    # Actual sequences start at row 1
    buf.seek(0)
    df = pd.read_csv(buf, header=1, compression="gzip", low_memory=False)

    return metadata, df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD A DATA UNIT
# ─────────────────────────────────────────────────────────────────────────────

section("1. LOADING A DATA UNIT")

url = EXAMPLE_UNITS["human_IgG_heavy"]
metadata, df = load_data_unit(url)

print(f"\n  Loaded {len(df):,} sequences")
print(f"  Total columns: {len(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. METADATA
# ─────────────────────────────────────────────────────────────────────────────

section("2. DATA UNIT METADATA")
print("  Each data unit has a metadata header describing the experiment:\n")
for k, v in metadata.items():
    print(f"    {k:<25} {v}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL COLUMN LIST
# ─────────────────────────────────────────────────────────────────────────────

section("3. ALL AVAILABLE COLUMNS (~100 per sequence)")
print("  OAS stores ~100 columns per sequence. Here's everything available:\n")

cols = df.columns.tolist()
for i, col in enumerate(cols):
    print(f"    [{i:>3}] {col}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROJECT-RELEVANT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

section("4. PROJECT-RELEVANT COLUMNS")
print("  Showing columns most relevant for evotuning and C05 analysis:\n")

available_project_cols = [c for c in PROJECT_COLS if c in df.columns]
missing = [c for c in PROJECT_COLS if c not in df.columns]

print(f"  Available: {available_project_cols}")
if missing:
    print(f"  Missing in this data unit: {missing}")

proj_df = df[available_project_cols].copy()
print(f"\n  First 3 rows:\n")
with pd.option_context("display.max_colwidth", 60, "display.width", 120):
    print(proj_df.head(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 5. A FEW EXAMPLE SEQUENCES (full detail)
# ─────────────────────────────────────────────────────────────────────────────

section("5. EXAMPLE SEQUENCES — FULL DETAIL")
print("  Inspecting 3 individual sequences in full:\n")

for i, (_, row) in enumerate(proj_df.head(3).iterrows()):
    print(f"  ── Sequence {i+1} ──")
    for col in available_project_cols:
        val = row.get(col, "N/A")
        # Truncate long sequences for display
        if isinstance(val, str) and len(val) > 80:
            val = val[:77] + "..."
        print(f"    {col:<30} {val}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUALITY OVERVIEW — ANARCI_STATUS
# ─────────────────────────────────────────────────────────────────────────────

section("6. QUALITY FLAGS — ANARCI_STATUS")
print("  This column tells you which sequences have problems.\n")

if "ANARCI_status" in df.columns:
    status_counts = df["ANARCI_status"].value_counts()
    total = len(df)
    print(f"  {'Status':<50} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*50} {'-'*8}  {'-'*6}")
    for status, count in status_counts.items():
        label = str(status)[:48]
        print(f"  {label:<50} {count:>8,}  {count/total*100:>5.1f}%")

    # Apply the "good sequence" filter
    good = df[~df["ANARCI_status"].str.contains("Shorter|insert|unusual", na=False, case=False)]
    productive_good = good[good.get("productive", pd.Series(["T"]*len(good))) == "T"]
    print(f"\n  After filtering (no truncation/indel/unusual, productive only):")
    print(f"    Kept: {len(productive_good):,} / {total:,}  ({len(productive_good)/total*100:.1f}%)")
else:
    print("  ANARCI_status column not found in this data unit.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CDRH3 LENGTH DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

section("7. CDRH3 LENGTH DISTRIBUTION")
print("  Key for your project: C05's CDRH3 length determines what's comparable.\n")

if "cdr3_aa" in df.columns:
    cdr3_lengths = df["cdr3_aa"].dropna().str.len()
    print(f"  CDRH3 length stats across {len(cdr3_lengths):,} sequences:")
    print(f"    min    : {cdr3_lengths.min()}")
    print(f"    median : {cdr3_lengths.median():.0f}")
    print(f"    mean   : {cdr3_lengths.mean():.1f}")
    print(f"    max    : {cdr3_lengths.max()}")
    print(f"\n  Length distribution (top 15 most common):")
    len_dist = cdr3_lengths.value_counts().sort_index()
    print(f"  {'Length':>8}   {'Count':>8}   {'Bar'}")
    for length, count in len_dist.head(15).items():
        bar = "█" * int(count / len_dist.max() * 30)
        print(f"  {length:>8}   {count:>8,}   {bar}")
else:
    print("  cdr3_aa column not available in this data unit.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. V GENE USAGE
# ─────────────────────────────────────────────────────────────────────────────

section("8. V GENE USAGE (top 10)")
print("  V gene determines framework region. C05 uses IGHV1-69.\n")
print("  Knowing the dominant V genes helps plan targeted evotuning.\n")

if "v_call" in df.columns:
    v_counts = df["v_call"].value_counts().head(10)
    total = len(df)
    print(f"  {'V gene':<25} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*25} {'-'*8}  {'-'*6}")
    for v_gene, count in v_counts.items():
        print(f"  {str(v_gene):<25} {count:>8,}  {count/total*100:>5.1f}%")
else:
    print("  v_call column not available in this data unit.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SOMATIC HYPERMUTATION LEVEL (v_identity)
# ─────────────────────────────────────────────────────────────────────────────

section("9. SOMATIC HYPERMUTATION — v_identity")
print(dedent("""
  v_identity = % identity between observed sequence and V germline.
  C05 is a broadly neutralizing antibody: it's affinity-matured,
  so it has lower v_identity (more mutations from germline).
  For evotuning, you might prefer sequences with low-to-mid v_identity
  (=experienced, affinity-matured repertoire) rather than naive IgM.
"""))

if "v_identity" in df.columns:
    vi = pd.to_numeric(df["v_identity"], errors="coerce").dropna()
    print(f"  V identity stats ({len(vi):,} sequences):")
    print(f"    min    : {vi.min():.3f}")
    print(f"    median : {vi.median():.3f}")
    print(f"    mean   : {vi.mean():.3f}")
    print(f"    max    : {vi.max():.3f}")
    print(f"\n  Bucket distribution:")
    buckets = pd.cut(vi, bins=[0, 0.7, 0.8, 0.9, 0.95, 1.01],
                     labels=["<70%", "70–80%", "80–90%", "90–95%", ">95%"])
    for bucket, count in buckets.value_counts().sort_index().items():
        bar = "█" * int(count / len(vi) * 40)
        print(f"    {str(bucket):<10} {count:>8,}  {bar}")
else:
    print("  v_identity column not available in this data unit.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. REDUNDANCY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

section("10. REDUNDANCY (clonal expansion)")
print(dedent("""
  Redundancy = how many times this exact sequence appears in the study.
  High redundancy = clonal expansion (one B-cell proliferated a lot).
  For evotuning you want to deduplicate, so this tells you how much
  your corpus will shrink after MMseqs2 clustering.
"""))

if "Redundancy" in df.columns:
    red = pd.to_numeric(df["Redundancy"], errors="coerce").dropna()
    print(f"  Redundancy stats ({len(red):,} sequences):")
    print(f"    seen once  : {(red == 1).sum():>8,}  ({(red==1).mean()*100:.1f}%)")
    print(f"    seen 2–5x  : {((red >= 2) & (red <= 5)).sum():>8,}  ({((red>=2)&(red<=5)).mean()*100:.1f}%)")
    print(f"    seen >5x   : {(red > 5).sum():>8,}  ({(red>5).mean()*100:.1f}%)")
    print(f"    max        : {red.max():.0f}")
else:
    print("  Redundancy column not available in this data unit.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FILTER PIPELINE SIMULATION — what your evotuning corpus will look like
# ─────────────────────────────────────────────────────────────────────────────

section("11. EVOTUNING FILTER PIPELINE SIMULATION")
print("  Simulating the filters you'd apply before evotuning:\n")

total = len(df)
print(f"  Step 0 — Raw sequences:               {total:>8,}")

# Step 1: productive only
if "productive" in df.columns:
    df1 = df[df["productive"] == "T"]
else:
    df1 = df  # assume productive if column missing
print(f"  Step 1 — Keep productive only:        {len(df1):>8,}  ({len(df1)/total*100:.1f}%)")

# Step 2: no ANARCI quality issues
if "ANARCI_status" in df1.columns:
    df2 = df1[~df1["ANARCI_status"].str.contains("Shorter|insert|unusual", na=False, case=False)]
else:
    df2 = df1
print(f"  Step 2 — Remove truncated/indel seqs: {len(df2):>8,}  ({len(df2)/total*100:.1f}%)")

# Step 3: has CDRH3
if "cdr3_aa" in df2.columns:
    df3 = df2[df2["cdr3_aa"].notna() & (df2["cdr3_aa"].str.len() > 0)]
else:
    df3 = df2
print(f"  Step 3 — Has CDRH3 sequence:          {len(df3):>8,}  ({len(df3)/total*100:.1f}%)")

# Step 4: has full VH sequence
if "sequence_alignment_aa" in df3.columns:
    df4 = df3[df3["sequence_alignment_aa"].notna()]
else:
    df4 = df3
print(f"  Step 4 — Has full VH sequence:        {len(df4):>8,}  ({len(df4)/total*100:.1f}%)")

print(f"\n  → {len(df4):,} sequences pass all filters from this data unit.")
print(f"  → After MMseqs2 deduplication at 99% identity, expect ~{int(len(df4)*0.4):,}–{int(len(df4)*0.7):,}.")

# Show a few clean sequences ready for evotuning
if len(df4) > 0 and "sequence_alignment_aa" in df4.columns:
    print(f"\n  Sample of 3 sequences ready for evotuning FASTA:\n")
    sample = df4["sequence_alignment_aa"].dropna().head(3)
    for i, (_, seq) in enumerate(sample.items()):
        print(f"  >oas_seq_{i}")
        print(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 12. COMPARE IgG vs IgM (load second data unit)
# ─────────────────────────────────────────────────────────────────────────────

section("12. IgG vs IgM COMPARISON (same study, different isotype)")
print("  Loading IgM data unit from the same study for comparison...\n")

try:
    meta_igm, df_igm = load_data_unit(EXAMPLE_UNITS["human_IgM_heavy"])

    for label, d in [("IgG", df4), ("IgM", df_igm)]:
        vi_col = pd.to_numeric(d.get("v_identity", pd.Series()), errors="coerce").dropna()
        cdr3_col = d.get("cdr3_aa", pd.Series()).dropna()
        print(f"  ── {label} ──")
        print(f"    Total sequences      : {len(d):,}")
        if len(vi_col) > 0:
            print(f"    Median v_identity    : {vi_col.median():.3f}  (lower = more SHM)")
        if len(cdr3_col) > 0:
            print(f"    Median CDRH3 length  : {cdr3_col.str.len().median():.0f} AA")
        print()

    print("  Interpretation:")
    print("  IgG = class-switched, affinity-matured (lower v_identity = more SHM)")
    print("  IgM = mostly naive pre-mutation repertoire")
    print("  → For evotuning C05 (a mature bnAb), IgG is more relevant.\n")
except Exception as e:
    print(f"  Could not load IgM data unit: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY — What to take away")
print(dedent("""
  Key fields for your evotuning corpus:
    sequence_alignment_aa  →  feed this to ESM2 (full VH variable domain)
    cdr3_aa                →  inspect CDRH3 diversity; NOT what evotuning learns
    ANARCI_status          →  filter: reject "Shorter", "insert", "unusual"
    productive             →  filter: keep "T" only
    v_call                 →  optionally restrict to C05's V gene (IGHV1-69)
    v_identity             →  proxy for somatic hypermutation level

  Recommended filters before evotuning:
    1. productive == "T"
    2. ANARCI_status has no truncation/indel/unusual flags
    3. cdr3_aa is present (non-null, non-empty)
    4. sequence_alignment_aa is present
    5. Deduplicate with MMseqs2 easy-linclust at 99% identity

  Next step on Euler:
    - Download a URL list from opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/
      (filter: Human, Heavy, IGHG)
    - Apply these filters across all data units
    - Run MMseqs2 easy-linclust for deduplication
    - Feed resulting FASTA to ESM2 continued pretraining
"""))
