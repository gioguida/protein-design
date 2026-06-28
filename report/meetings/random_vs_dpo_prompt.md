# Prompt: build `random_vs_dpo_650m_comparison.ipynb`

> Paste everything below the line into a new Claude Code chat started from the
> `protein-design` repo root.

---

## Goal

Create `report/meetings/random_vs_dpo_650m_comparison.ipynb`: a near-exact clone of
the existing `report/meetings/pssm_vs_dpo_650m_comparison.ipynb`, but where the
**naive baseline generator is a "dumb" random library-mutant sampler instead of the
PSSM sampler**. The DPO 650M side stays identical. Everything must be ready to
generate sequences and run them through the same three scorers (M22 esme, the
ModelArcEnsemble uncertainty scorer, and DPO PLL).

Two design decisions are already made:
1. **Sweep the trust radius** (max edits from WT) in place of the PSSM temperature
   sweep. Headline run = trust_radius 11 (matches the DPO beam's `--n-steps 11`).
2. **Port the generator faithfully** from the Lorenz code (WT-centered, uniform
   over observed residues) — see "Generator algorithm" below.

## Repo conventions (do not deviate)
- Every Python invocation goes through `uv run` (never bare `python`/`python3`).
- Data-prep / heavy artifacts (generated CSVs, scores, plots) go under
  `$SCRATCH` (`/cluster/scratch/mdenegri/protein-design/random_vs_dpo_650m`);
  final reusable code + the notebook + vector figures go under the project.
- No AI / Co-Authored-By attribution in any git commit.
- Branch before committing if on `main`.

## Source material to read first
**This repo (the thing to mirror):**
- `report/meetings/pssm_vs_dpo_650m_comparison.ipynb` — the template notebook.
- `src/protein_design/pssm_baseline.py` — the module to mirror (counts → log-freq →
  temperature sampling → `build_output_rows`). Note the output schema:
  `chain_id, gibbs_step(=0), sequence(=add_context(cdrh3)), cdrh3, n_mutations, model_variant`.
- `scripts/analysis/run_pssm_baseline_sweep.py` — the sweep driver to mirror.
- `scripts/pssm_sampling.py` — the single-run CLI to mirror.
- `conf/analysis/pssm_baseline_sweep_dpo_comparison.yaml` — the sweep config to mirror.
- `bash_scripts/utils/run_pssm_baseline_sweep_dpo_comparison.sbatch` — the sbatch to mirror.
- `tests/test_pssm_baseline.py` — write the analogous test.
- Scorer entry points already used by the PSSM notebook (reuse unchanged):
  `bash_scripts/utils/score_generated_with_esme.sbatch`,
  `bash_scripts/utils/score_generated_with_uncertainty.sbatch`,
  `scripts/analysis/score_generated_with_pll.py` / `bash_scripts/utils/score_generated_with_pll.sbatch`.

**The original generator source (Lorenz branch — already checked out & up to date):**
- `~/uncertainty_protein/src/genetic_algorithm/alphabet.py` — port
  `build_position_alphabet`, `_random_mutant`, and `build_seed_pool(mode="random_wt")`.
  (Branch `Lorenz`; confirmed identical to `origin/Lorenz`.)

## Generator algorithm (port faithfully, one intentional deviation)
From `alphabet.py`, WT = `C05_CDRH3` (length 24):
1. **Position alphabet**: for each position, the *set* of residues observed in the
   train data, with the WT residue always included. "Mutable" positions = those
   with > 1 observed residue. (Uniform over the set — NOT frequency-weighted; this
   is what makes it dumber than the PSSM.)
2. **One random mutant** (`_random_mutant`): draw `k ~ Uniform[1, min(trust_radius,
   #mutable)]`, pick `k` mutable positions without replacement, set each to a
   uniformly-chosen observed residue ≠ current. So `trust_radius` caps edits from WT.
3. **Intentional deviation for sample-size parity:** the PSSM produces
   `n_sequences` rows *with* possible duplicates. The Lorenz `build_seed_pool`
   dedups to unique. To keep the libraries the same size (5000, matching the DPO
   filtered set), generate exactly `n_sequences` rows **allowing duplicates** (do
   not dedup-to-unique). Make this a flag if easy, default = allow duplicates.

**Train data for the alphabet (apples-to-apples with PSSM):** build the position
alphabet from the *same* train split the PSSM used — dataset `ed2_m22`, filtered by
`enrichment_threshold = WT_M22_BINDING_ENRICHMENT` — so the ONLY difference between
the two baselines is the sampling scheme. Reuse
`protein_design.pssm_baseline.load_train_dataframe` (or the same resolution logic)
rather than re-implementing split resolution. Use `add_context` for the `sequence`
column and `model_variant="random"`.

## Generator code — ALREADY BUILT (do not rebuild; read & reuse)
The following are already implemented, tested (`uv run --with pytest python -m
pytest tests/test_random_baseline.py` → 6 passed), and smoke-run verified:
- `src/protein_design/random_baseline.py`
- `scripts/random_sampling.py`
- `scripts/analysis/run_random_baseline_sweep.py`
- `conf/analysis/random_baseline_sweep_dpo_comparison.yaml`
- `bash_scripts/utils/run_random_baseline_sweep_dpo_comparison.sbatch`
- `tests/test_random_baseline.py`

Headline run = `radius_11/random_output.csv` under
`/cluster/scratch/mdenegri/protein-design/random_vs_dpo_650m/random_sweep`.
Note: the generator is WT-centered and always makes ≥1 edit, so WT itself never
appears and `n_mutations ∈ [1, trust_radius]`. Your remaining job is the NOTEBOOK.

## (Reference) What the generator code does — mirrors the PSSM infra
1. `src/protein_design/random_baseline.py` — mirrors `pssm_baseline.py`:
   `build_position_alphabet`, `sample_random_mutants(...)`, `build_output_rows(...)`
   (schema identical to PSSM's, `model_variant="random"`). Reuse helpers from
   `pssm_baseline.py` where sensible (train loading, `hamming_distance`, `add_context`).
2. `scripts/random_sampling.py` — single-run CLI (mirror `pssm_sampling.py`): args
   for `--trust-radius`, `--n-sequences`, `--seed`, `--output-csv`.
3. `scripts/analysis/run_random_baseline_sweep.py` — sweep driver (mirror the PSSM
   one) sweeping `trust_radius`, writing per-radius CSVs + the same diagnostic
   plots (entropy heatmap, JSD-vs-knob, edit-distance, pairwise-Hamming, logos,
   summary CSV). Swap "temperature" → "trust_radius" in labels/paths.
4. `conf/analysis/random_baseline_sweep_dpo_comparison.yaml` — mirror the PSSM
   config: `trust_radii: [2, 4, 6, 8, 11]`, `n_sequences: 5000`, `seed: 42`,
   `dataset: ed2_m22`, `enrichment_threshold: WT_M22_BINDING_ENRICHMENT`,
   output base under `.../random_vs_dpo_650m/random_sweep`.
5. `bash_scripts/utils/run_random_baseline_sweep_dpo_comparison.sbatch` — CPU-only
   (pure numpy), mirror the PSSM sbatch.
6. `tests/test_random_baseline.py` — analogous to `test_pssm_baseline.py`
   (determinism by seed, length-24 outputs, n_mutations ≤ trust_radius, alphabet
   built correctly, output schema).

## The notebook `random_vs_dpo_650m_comparison.ipynb`
Clone the PSSM notebook section-by-section; keep prose, figures, and analysis
identical except:
- **Generator name everywhere**: PSSM → "random library-mutant baseline" (`random`).
  Update the "Intuition / Naive baseline" markdown to describe the WT-centered
  uniform-over-observed sampler (contrast with PSSM's frequency-weighted draw).
- **Paths**: new scratch base
  `/cluster/scratch/mdenegri/protein-design/random_vs_dpo_650m`; new figure dir
  `report/figures/random_vs_dpo_650m`. `GEN_SEQ_COL = 'cdrh3'` unchanged.
- **Reuse the DPO side unchanged**: the DPO beam CSV and all DPO scores (M22, UQ,
  PLL) are generator-independent. Point the DPO paths at the EXISTING
  `pssm_vs_dpo_650m` scratch artifacts (do not regenerate the DPO beam or rescore
  DPO). Only the baseline side is new.
- **Temperature → trust radius**: replace the "Why T=0.9" and "Does a lower
  temperature maximise the M22 score?" sections with the trust-radius analog
  ("Why trust_radius=11", "Does a smaller trust radius maximise the M22 score?").
  The headline baseline run = trust_radius 11.
- **Keep**: the `MAX_N_MUTATIONS` fair-comparison cap, novelty analysis, M22
  scorer section + summary/top-15/score-vs-distance, the ModelArcEnsemble
  uncertainty-scorer section (epistemic/aleatoric ribbons), the quality-vs-
  diversity frontier, and the DPO-PLL-vs-scorers section — all identical, just fed
  the random baseline instead of PSSM.
- Preflight cells must print the new `sbatch`/CLI commands for the missing random
  artifacts and raise if absent (same pattern as the PSSM notebook); do NOT submit
  jobs automatically.

## Workflow
1. Read the source files above; confirm the PSSM notebook's exact cell structure.
2. Build the module + CLI + sweep + config + sbatch + test; run the test with
   `uv run pytest tests/test_random_baseline.py`.
3. Do a tiny local smoke run of the generator (e.g. 20 seqs, trust_radius 5) to
   confirm the output schema and n_mutations bounds.
4. Build the notebook. Validate it runs end-to-end ONLY after the artifacts exist;
   until then the preflight cells should cleanly report what's missing.
5. Summarize what was created and the exact sbatch commands to generate + score the
   random baseline.

Ask me before regenerating or rescoring anything on the DPO side — that should be
reused as-is.
