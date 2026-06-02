from __future__ import annotations

from pathlib import Path

import pandas as pd

from protein_design.analysis.novelty import (
    ReferenceSource,
    annotate_sequence_membership,
    build_reference_index,
)


def test_build_reference_index_and_annotation(monkeypatch, tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_csv = raw_dir / "raw.csv"

    pd.DataFrame({"aa": ["AAA", "BBB", "AAA"]}).to_csv(dataset_csv, index=False)
    pd.DataFrame({"aa": ["BBB", "CCC"]}).to_csv(raw_csv, index=False)

    monkeypatch.setattr(
        "protein_design.analysis.novelty.iter_reference_sources",
        lambda repo_root: [
            ReferenceSource(label="configured_ds", path=dataset_csv, seq_col="aa"),
            ReferenceSource(label="raw:raw", path=raw_csv, seq_col="aa"),
        ],
    )

    reference_index = build_reference_index(tmp_path)

    assert reference_index["AAA"] == frozenset({"configured_ds"})
    assert reference_index["BBB"] == frozenset({"configured_ds", "raw:raw"})
    assert reference_index["CCC"] == frozenset({"raw:raw"})

    df = pd.DataFrame({"cdrh3": ["AAA", "BBB", "DDD", None]})
    annotated = annotate_sequence_membership(
        df,
        seq_col="cdrh3",
        reference_index=reference_index,
    )

    assert annotated["present_in_existing_dataset"].tolist() == [True, True, False, False]
    assert annotated["n_matching_datasets"].tolist() == [1, 2, 0, 0]
    assert annotated["matching_datasets"].tolist() == [
        "configured_ds",
        "configured_ds;raw:raw",
        "",
        "",
    ]
