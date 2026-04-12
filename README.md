# protein-design — ESM2 Evotuning Pipeline

Continued masked language model (MLM) pretraining of ESM2 on human IgG
heavy chain sequences from the [Observed Antibody Space (OAS)](https://opig.stats.ox.ac.uk/webapps/oas/)
database. The goal is to bias ESM2 towards antibody sequence space before
downstream fine-tuning for antibody design tasks.

All instructions below are for the **ETH Euler cluster**.

---

## Prerequisites (one-time setup)

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.template .env
```

Edit `.env` with your actual paths:

```
SCRATCH_DIR=/cluster/scratch/$USER/protein-design
PROJECT_DIR=/cluster/project/infk/krause/$USER/protein-design
TRAIN_DIR=/cluster/scratch/$USER/protein-design/train
WANDB_PROJECT=protein-design-evotuning
WANDB_DIR=/cluster/scratch/$USER/protein-design/wandb
```

### 3. Log in to Weights & Biases

```bash
source .venv/bin/activate
wandb login   # paste your token from https://wandb.ai/authorize
```

### 4. Create logs directory

```bash
mkdir -p logs
```

---

## Storage layout

| What | Where | Why |
|------|-------|-----|
| Code | `~/protein-design/` (home) | Persistent, small |
| Raw/intermediate data | `$SCRATCH_DIR` | Large, ephemeral ok |
| Training runs | `$TRAIN_DIR/{run_name}/` | Large, scratch |
| Archived best checkpoints | `$PROJECT_DIR/checkpoints/` | Persistent |
| Training datasets | `$PROJECT_DIR/datasets/` | Persistent |
| Scoring datasets | `$PROJECT_DIR/datasets/scoring/` | Persistent |
| Reports & plots | `$PROJECT_DIR/reports/` | Persistent |
| W&B cache | `$WANDB_DIR` (scratch) | Large, synced to cloud |
| SLURM logs | `logs/` (project root) | Small, gitignored |

---

## Pipeline

### Step 1 — Download OAS data

**Get the URL list** (do this on your local machine):

1. Go to https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/
2. Set filters: **Species = Human**, **Chain = Heavy**, **Isotype = IGHG**
3. Click Search, then download the `bulk_download.sh` script from the results page
4. Extract the URLs:
   ```bash
   grep -o 'https://[^ ]*' bulk_download.sh > oas_ighg_urls.txt
   ```
5. Copy to Euler:
   ```bash
   scp oas_ighg_urls.txt euler:/cluster/home/$USER/protein-design/
   ```

**Submit the download job on Euler:**

```bash
sbatch bash_scripts/download_oas.sbatch
```

Expected output: ~2196 `.csv.gz` files in `$SCRATCH_DIR/oas_raw/`

---

### Step 2 — Filter sequences

Reads all `.csv.gz` files and applies quality filters:
- **Productive** rearrangements only
- **ANARCI status**: rejects truncated, insertion-containing sequences
- **CDR3 present**: non-null, non-empty `cdr3_aa`
- **Minimum length**: ≥ 50 amino acids

```bash
sbatch bash_scripts/filter_oas.sbatch
```

Expected: ~60–80% of raw sequences pass, resulting in tens of millions of sequences.

---

### Step 3 — Deduplicate with MMseqs2

Clusters sequences at **99% identity** and keeps one representative per cluster.

```bash
sbatch bash_scripts/mmseqs2.sbatch
```

Expected: 30–60% reduction, leaving ~5–20M unique sequences.

> Steps 2 and 3 can be run in a single job: `sbatch bash_scripts/prepare_oas.sbatch`

---

### Step 4 — Find optimal batch size (recommended before first full run)

Before committing to a long training job, run this interactively on the target GPU to find
the maximum batch size that fits in VRAM. This matters because training at a small batch size
(e.g. 32) can leave the GPU at <5% utilization, making jobs 10–20× slower than necessary.

**When to run this:**
- Before your first training run on a new GPU type
- After changing the model size (`model_name`) or sequence length (`max_seq_len`)
- After switching to a different `--gres=gpumem:Xg` allocation

```bash
srun --gpus=1 --gres=gpumem:24g --mem-per-cpu=4G --pty \
    uv run python scripts/find_max_batch_size.py --config configs/evotuning_base.yaml
```

The script sweeps batch sizes (32 → 64 → 128 → ...) and prints a table of throughput and
VRAM usage at each step, stopping at the first OOM. It then recommends a `batch_size` and
`gradient_accumulation_steps` to paste into your config.

**How to apply the results:**

1. Set `batch_size` in your config to the reported max (or half it for headroom)
2. Adjust `gradient_accumulation_steps` to keep the effective batch size around 512
3. Scale `learning_rate` with the square root of the effective batch size change —
   e.g. if effective batch grows 4×, multiply lr by √4 = 2

Example result on a Quadro RTX 6000 (24 GB) with `esm2_t12_35M_UR50D`, `max_seq_len=256`:

| Batch | Throughput | Status |
|-------|------------|--------|
| 32    | 16 seq/s   | OK     |
| 64    | 295 seq/s  | OK     |
| 128   | 321 seq/s  | OK     |
| 256   | 324 seq/s  | OK     |
| 512   | —          | OOM    |

→ Recommended: `batch_size: 256`, `gradient_accumulation_steps: 2`, `learning_rate: 2.0e-5`

> The VRAM column in the output shows post-backward memory (model footprint only), not the
> peak during the forward pass. The OOM boundary is the reliable signal, not the VRAM number.

---

### Step 6 — Debug run (recommended)

Quick end-to-end check on 10k sequences. Uses an isolated scratch directory.

```bash
sbatch bash_scripts/debug_run.sbatch configs/evotuning_base.yaml <run_name>
```

---

### Step 7 — Full training

```bash
sbatch bash_scripts/evotuning.sbatch configs/evotuning_base.yaml <run_name>
sbatch bash_scripts/evotuning.sbatch configs/my_experiment.yaml  <run_name>
```

Training outputs are saved to `$TRAIN_DIR/{run_name}/`:
```
$TRAIN_DIR/{run_name}/
├── config.yaml              # resolved config snapshot
├── checkpoints/
│   ├── step_5000.pt
│   ├── step_10000.pt
│   └── final.pt
└── best.pt                  # lowest validation perplexity
```

The best checkpoint is also archived to `$PROJECT_DIR/checkpoints/{run_name}/best.pt`.

Monitor training via `tail -f logs/evotuning_<jobid>.err` or the W&B dashboard.

### Starting from a previous checkpoint

To finetune from a previous run's best checkpoint, add `finetune` to your config:

```yaml
finetune: ${PROJECT_DIR}/checkpoints/previous_run_name/best.pt
```

This loads model weights only — optimizer and scheduler are initialized fresh.

---

## Sbatch scripts reference

| Script | Purpose | GPU |
|---|---|:---:|
| `download_oas.sbatch` | Download 2196 OAS data units | No |
| `filter_oas.sbatch` | Filter sequences (productive, ANARCI, CDR3, length) | No |
| `mmseqs2.sbatch` | Deduplicate at 99% sequence identity | No |
| `prepare_oas.sbatch` | Filter + deduplicate in one job | No |
| `debug_run.sbatch` | End-to-end test on 10k sequences | Yes (20GB) |
| `evotuning.sbatch` | Full ESM2 evotuning run | Yes (24GB) |

---

## Project structure

```
configs/                  Hyperparameter configs (YAML)
  evotuning_base.yaml     Full training config (copy & modify for experiments)
scripts/                  Pipeline entry points
  download_oas.sh         Download helper (reads URL file)
  filter_oas.py           OAS filtering → FASTA
  run_mmseqs2.sh          MMseqs2 deduplication wrapper
  train_evotuning.py      Training entry point (--config and --run-name required)
  find_max_batch_size.py  Interactive GPU VRAM sweep to find optimal batch size
  clean_d2.py             Filter enrichment datasets to distance-2 mutants
  search_c05.py           MMseqs2 search for C05-like sequences
  score_mutational_paths.py  Score mutants with ESM2
  migrate_project_dir.sh  One-time migration of project folder layout
bash_scripts/             SLURM job submission scripts
src/protein_design/
  model.py                ESM2 wrapper with layer freezing
  data.py                 FASTA dataset and DataLoader
  train.py                Training loop with gradient accumulation
  evaluate.py             Perplexity evaluation
  scoring.py              Mutational path scoring & Spearman correlation
  utils.py                Config loading and env var resolution
```

---

## Troubleshooting

**`Environment variable 'X' is not set and has no fallback`**
→ The `.env` file is missing or incomplete. Check that `.env` exists and contains all required variables (see `.env.template`).

**`These module(s) exist but cannot be loaded: mmseqs2/14-7e284`**
→ MMseqs2 requires prerequisite modules:
```bash
module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
```

**`wandb.errors.UsageError: No API key configured`**
→ Run `wandb login` on a login node before submitting training jobs.

**Downloads failing on compute nodes**
→ Compute nodes have no internet by default. The `eth_proxy` module must be loaded.
