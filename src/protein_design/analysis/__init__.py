"""Analysis / paper-figure pipeline.

Two registries drive everything:
  - conf/analysis/models.yaml   — models (key -> checkpoint, base_model, label, ...)
  - conf/analysis/dms_datasets.yaml — DMS datasets (key -> path, seq_col, truth, scorer)

Artifacts are written to a writable per-model tree under $ANALYSIS_DIR:
    $ANALYSIS_DIR/<model_key>/<kind>/<name>   (+ <name>.meta.json sidecar)

`registry` holds the path/IO/provenance helpers; `figures` holds the
notebook-facing plotting functions.
"""
