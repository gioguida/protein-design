from __future__ import annotations

import numpy as np
import pandas as pd


def _hamming_distance(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def nearest_neighbor_match(
    query_seqs: list[str],
    dms_df: pd.DataFrame,
    enrichment_cols: list[str],
) -> pd.DataFrame:
    """Nearest-neighbor DMS match by Hamming distance for each query sequence."""
    if "aa" not in dms_df.columns:
        raise ValueError("dms_df must contain an 'aa' column.")
    if not enrichment_cols:
        raise ValueError("enrichment_cols must not be empty.")
    missing = [c for c in enrichment_cols if c not in dms_df.columns]
    if missing:
        raise ValueError(f"dms_df missing enrichment columns: {missing}")

    dms_work = dms_df.copy()
    dms_work["aa"] = dms_work["aa"].astype(str)

    rows: list[dict[str, object]] = []
    for query in query_seqs:
        query = str(query)
        cands = dms_work[dms_work["aa"].str.len() == len(query)]
        if cands.empty:
            row = {"query_seq": query, "nn_seq": None, "hamming_dist": np.nan}
            for col in enrichment_cols:
                row[col] = np.nan
            rows.append(row)
            continue

        best_dist = None
        best_rows: list[pd.Series] = []
        for _, cand in cands.iterrows():
            dist = _hamming_distance(query, str(cand["aa"]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_rows = [cand]
            elif dist == best_dist:
                best_rows.append(cand)

        if len(best_rows) == 1:
            best = best_rows[0]
        else:
            first_col = enrichment_cols[0]
            best = max(
                best_rows,
                key=lambda r: float(r[first_col]) if pd.notna(r[first_col]) else float("-inf"),
            )

        row = {
            "query_seq": query,
            "nn_seq": str(best["aa"]),
            "hamming_dist": int(best_dist),
        }
        for col in enrichment_cols:
            row[col] = best[col]
        rows.append(row)

    return pd.DataFrame(rows)
