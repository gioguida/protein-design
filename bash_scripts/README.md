# `bash_scripts/` — SLURM job recipes

How to run the SBATCH and helper scripts in this directory, in what order, and
where to look when you want to change a model, dataset, or hyperparameter.

## Conventions

- Every `*.sbatch` writes its log to `bash_scripts/logs/<job_name>_<jobid>.log`.
  Check there first when a job fails.
- `common_setup.sh` is sourced by every `.sbatch`: it `cd`s to the repo root,
  loads `eth_proxy`, exports `.env.local`, and activates `.venv`. If a job
  can't see `wandb` or a project env var, check that `.env.local` exists.
- The `flu` conda env at `/cluster/project/infk/krause/mdenegri/miniconda3/envs/flu`
  is used only for scripts that need `esme` + `flash_attn` (currently
  `score_dms.sbatch`). Flash-attn needs Ampere or newer (`rtx_3090`, `A100`),
  not V100.
- `--time=` defaults are conservative — bump them for new larger datasets.
- Outputs always go under either `$PROJECT_DIR` (caches, big artifacts) or
  `$HOME/protein-design/plots/` (plots, reports). Source data under
  `gguidarini`'s tree is read-only.

---

## Pipelines

### 1. DMS correlation plots (PLL ↔ scorer ↔ truth)

**Goal.** For one ESM2 checkpoint and one DMS dataset, produce four PDFs that
relate the model's pseudo-log-likelihood (PLL) on the CDR-H3, the
M22/SI06 binder scorer's prediction, and the experimental log-enrichment
("truth"):

```
$HOME/protein-design/plots/dms/<model_name>/<dataset>/
  pll_vs_scorer.pdf         (colored scatter + marginal histograms)
  pll_vs_truth.pdf          (hist2d, log color scale)
  scorer_vs_truth.pdf       (hist2d, log color scale)
  pll_scorer_truth_3d.pdf   (3D scatter colored by truth)
```

Each plot shows OLS fit, Pearson `r`, Spearman `ρ`, and `n`. p-values are
omitted because at `n` ≫ 1000 they underflow to 0.

#### Scripts in this pipeline

| Step | Script | Env | Cost |
|---|---|---|---|
| 1 | `utils/compute_pll.sbatch` | venv (`common_setup.sh`) + GPU | ~30 min / 30k seqs |
| 2 | `utils/score_dms.sbatch`   | `flu` conda env + Ampere GPU  | ~15 s / 40k seqs |
| 3 | `utils/plot_dms.sbatch`    | venv, CPU                     | <1 min |

Steps 1 and 2 are independent and can run in parallel. Step 3 only reads
caches and is fast — you can also run it locally with `uv run python …`
instead of `sbatch`.

#### The registry: `conf/analysis/dms_datasets.yaml`

**This file (not the CLI) is where you change datasets, ground-truth columns,
or scorer settings.** It has three sections:

- `scorers:` — checkpoint path + esme-loader settings for each scorer (`m22`,
  `si06`). The defaults reproduce training-time behavior bit-for-bit.
- `datasets:` — for each dataset key: `path`, `split`, `seq_col`,
  `enrichment_col`, `scorer` (or `null` if there is no binder scorer for it).
- `paths:` — `cache_root` for cached PLL/scorer CSVs, `plots_root` for PDFs.

To add a dataset: append an entry under `datasets:`. To change which scorer a
dataset uses (e.g. score `ed2_m22` with the SI06 scorer): change `scorer:`.
To change the truth column: change `enrichment_col:`.

#### Cache layout

```
$cache_root/                          # default: $PROJECT_DIR/cache
├── pll/<model_name>/<dataset>.csv         # cols: <seq_col>, pll
└── scorer_preds/<dataset>/<scorer>.csv    # cols: <seq_col>, score
```

PLL is model-dependent but scorer-independent → keyed by model. Scorer
predictions are scorer/dataset-dependent but model-independent → keyed by
dataset/scorer. The plot script joins them on the sequence column.

Every step is **idempotent**: if its output CSV exists it is reused. Pass
`--force` to recompute.

#### Running it

**1) Compute PLL for a new model.**

```bash
sbatch bash_scripts/utils/compute_pll.sbatch \
  --model-name <human-readable-name> \
  --checkpoint /path/to/checkpoint.pt \
  --dataset all
```

- `--model-name` is the cache key (folder name). Pick something descriptive
  like `evodpo_4ep_step1376` or `just_dpo_4ep_step1376`. Used by `plot_dms`.
- `--checkpoint` can be a `.pt` file, an HF directory, or an HF model ID.
  Omit it to use vanilla `facebook/esm2_t12_35M_UR50D`.
- `--dataset` is a key from the registry, or `all` for every dataset.
- Each PLL run takes ~30 min per 30k unique sequences on a single GPU.

**2) Run the binder scorers on the datasets.** You only need to do this once
per (dataset, scorer) pair — the cache survives across all models.

```bash
sbatch bash_scripts/utils/score_dms.sbatch --dataset all
```

Skips datasets that already have a cached scorer CSV. Requires an Ampere GPU
(`rtx_3090` or `A100`) for flash-attn.

**3) Plot.**

```bash
sbatch bash_scripts/utils/plot_dms.sbatch \
  --model-name <human-readable-name> \
  --dataset all
```

(Or `uv run python scripts/analysis/plot_dms_correlations.py --model-name … --dataset …`
locally — it's CPU-only and fast.) Datasets without both PLL and scorer
caches are skipped with a warning telling you which step is missing.

#### Worked example

Plotting all four PDFs for `just_dpo_4ep_step1376` on every DMS dataset
that has an M22/SI06 scorer:

```bash
# one-time per dataset (skipped if cache exists)
sbatch bash_scripts/utils/score_dms.sbatch --dataset all

# per-model — three parallel jobs is faster than --dataset all sequentially
for ds in ed2_m22 ed5_m22 exp; do
  sbatch --time=2:00:00 bash_scripts/utils/compute_pll.sbatch \
    --model-name just_dpo_4ep_step1376 \
    --checkpoint /cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo_4ep/step_1376.pt \
    --dataset $ds
done

# once the PLL jobs finish (no GPU needed)
uv run python scripts/analysis/plot_dms_correlations.py \
  --model-name just_dpo_4ep_step1376 --dataset all
```

#### Troubleshooting

- **`PLL cache missing` from `plot_dms`** → run `compute_pll.sbatch` for the
  `(model_name, dataset)` it mentions. The error message includes the exact
  command.
- **`FlashAttention only supports Ampere GPUs or newer`** → the scorer landed
  on a V100. Re-submit `score_dms.sbatch` (it already requests `rtx_3090`).
- **Plots produced with the wrong scorer prediction** → if you ever see
  predictions clustered in a narrow range (~`−10..−8`), the scorer was
  loaded with the wrong `use_context` / `aggregate` setting. The defaults
  in `dms_datasets.yaml` (`use_context: false`, `aggregate: mean_pooling`,
  `dropout: 0`) reproduce M22_best.pt training. Don't change them unless
  you're loading a different scorer checkpoint.
- **`p = 0.0e+00` in old plots** → harmless; just means the t-statistic
  overflowed and `scipy.stats.t.sf` underflowed. Newer plots omit p entirely.

---

## TODO: other pipelines to document

- Training (`train.sbatch`, `eval.sbatch`, `ttt.sbatch`)
- OAS data prep (`utils/download_oas.sbatch`, `utils/prepare_oas.sbatch`,
  `utils/filter_oas.sbatch`)
- Beam search / Gibbs sampling (`utils/run_beam_*`, `utils/run_gibbs_*`)
- Embedding / similarity analyses (`utils/plot_*similarity*.sbatch`,
  `utils/mmseqs2.sbatch`)
