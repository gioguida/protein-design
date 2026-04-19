# protein-design — ESM2 training pipeline

Continued masked-language-model pretraining of ESM2 on antibody sequences,
with a unified pipeline that chains evotuning → domain finetune → TTT into
a single declarative config.

Target platform: **ETH Euler cluster**.

---

## Quickstart

```bash
# One sbatch, one pipeline name. Chains every stage in a single allocation.
sbatch bash_scripts/train.sbatch evotuning                 # OAS evotuning only
sbatch bash_scripts/train.sbatch evotune_c05_ttt           # evotune → C05 → TTT
sbatch bash_scripts/train.sbatch c05 \
    pipeline.init_from=$PROJECT_DIR/checkpoints/<evo_run>/best.pt
```

Pipelines live in `conf/pipeline/`. Add a new YAML there to define a new
chain; no new sbatch needed.

---

## One-time setup

```bash
uv sync                                  # install deps
cp .env.template .env.local              # fill in cluster paths (see below)
source .venv/bin/activate && wandb login # on a login node
```

`.env.local`:

```
SCRATCH_DIR=/cluster/scratch/$USER/protein-design
PROJECT_DIR=/cluster/project/infk/krause/$USER/protein-design
TRAIN_DIR=$SCRATCH_DIR/train
WANDB_PROJECT=protein-design-evotuning
WANDB_DIR=$SCRATCH_DIR/wandb
```

---

## Storage layout

| What | Where |
|---|---|
| Code | `~/protein-design/` |
| Raw/intermediate data | `$SCRATCH_DIR` |
| Training runs | `$TRAIN_DIR/<run_name>/` |
| Archived checkpoints | `$PROJECT_DIR/checkpoints/<run_name>/` |
| Datasets | `$PROJECT_DIR/datasets/` |
| Reports & plots | `~/protein-design/plots/`, `~/protein-design/reports/` |
| SLURM logs | `bash_scripts/logs/` |

---

## Training

A **pipeline** is a list of stages. Each stage names a `task` (hyperparams
in `conf/task/`) and a `data` group (`conf/data/`). The output checkpoint
of stage *n* is automatically threaded as the `finetune` input to stage
*n+1*.

Available pipelines (`conf/pipeline/`):

| Pipeline | Stages |
|---|---|
| `evotuning` | evotuning on OAS |
| `c05` | C05-5k finetune (needs `pipeline.init_from=...`) |
| `ttt` | TTT on one CDRH3 (needs `pipeline.init_from=...`) |
| `evotune_c05_ttt` | all three, chained |

Submit with:

```bash
sbatch bash_scripts/train.sbatch <pipeline> [hydra overrides...]
```

Any Hydra override works. To tweak a specific stage, use
`pipeline.stages.<i>.overrides.<key>=<value>`:

```bash
sbatch bash_scripts/train.sbatch evotune_c05_ttt \
    pipeline.stages.1.overrides.training.learning_rate=1e-4
```

Each stage writes to `$TRAIN_DIR/<pipeline_name>__<stage_name>_<timestamp>/`
and archives the handoff checkpoint to
`$PROJECT_DIR/checkpoints/<same_name>/`.

---

## Evaluation

Score an existing checkpoint against the D2 datasets without retraining:

```bash
sbatch bash_scripts/eval.sbatch $PROJECT_DIR/checkpoints/<run>/best.pt scoring=d2
```

Writes `eval_metrics.json` next to the checkpoint.

---

## Data preparation

Everything under `bash_scripts/utils/` is one-off data-prep:

| Script | Purpose |
|---|---|
| `utils/download_oas.sbatch` | Download OAS `.csv.gz` files |
| `utils/filter_oas.sbatch` | Filter sequences → FASTA |
| `utils/mmseqs2.sbatch` | Dedup at 99% identity |
| `utils/prepare_oas.sbatch` | Filter + dedup combined |
| `utils/clean_d2.sbatch` | Filter D2 enrichment data |
| `utils/search_c05.sbatch` | MMseqs2 search for C05-like seqs |
| `utils/plot_cdrh3_lengths.sbatch` | CDRH3 length histogram |

Full OAS setup:

```bash
# Local: export OAS URL list, scp oas_ighg_urls.txt to cluster
sbatch bash_scripts/utils/download_oas.sbatch   # → $SCRATCH_DIR/oas_raw/
sbatch bash_scripts/utils/prepare_oas.sbatch    # filter + dedup in one job
```

---

## Find optimal batch size (before first real run)

```bash
srun --gpus=1 --gres=gpumem:24g --mem-per-cpu=4G --pty \
    uv run python scripts/find_max_batch_size.py
```

The script sweeps batch sizes and prints throughput + VRAM until OOM.
Apply the result: set `training.batch_size` to the max (or half for
headroom) and adjust `gradient_accumulation_steps` to keep effective
batch ≈512. Scale `learning_rate` by √(effective-batch-ratio).

---

## Project layout

```
conf/
  config.yaml              base defaults (task/model/data/scoring)
  pipeline/                stage lists — pick one via +pipeline=<name>
  task/ model/ data/ scoring/
scripts/
  train.py                 pipeline entry point
  eval.py                  standalone scorer (+output_csv_dir for per-pair CSVs)
  find_max_batch_size.py   VRAM sweep
  data_prep/               filter_oas, search_c05, clean_d2, ...
  analysis/                plot_results, plot_cdrh3_lengths
bash_scripts/
  train.sbatch, eval.sbatch, plot_results.sbatch
  utils/                   data-prep sbatches
  common_setup.sh          sourced by every sbatch
src/protein_design/
  model.py                 ESM2 wrapper (training + PLL scoring)
  eval.py                  MLM + PLL + CDR pseudo-ppl + Spearman (shared)
  constants.py             C05 sequences (shared)
  config.py                shared configs: ModelConfig, ScoringConfig, RunConfig
  utils.py                 fs + tokenizer helpers
  evotuning/
    config.py              DataConfig, TrainingConfig (MLM)
    pipeline.py            sequential stage runner
    train.py               unified trainer (evotuning + TTT)
    data.py                OAS FASTA dataloaders
  # dpo/ — to be added alongside evotuning/
```

---

## Troubleshooting

- **`Environment variable 'X' is not set`** → `.env.local` missing or
  incomplete; check against `.env.template`.
- **`module mmseqs2/14-7e284 cannot be loaded`** →
  `module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284` first (the
  relevant sbatches already do this).
- **wandb login on compute node fails** → run `wandb login` on a
  login node once; the key is cached.
- **Downloads fail on compute nodes** → `eth_proxy` module must be
  loaded (common_setup.sh does this).
