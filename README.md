# protein-design — ESM2 Evotuning Pipeline

Continued masked language model (MLM) pretraining of ESM2 (650M) on human IgG
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

Edit `.env` with your actual paths. The values should be:

```
DATA_DIR=/cluster/scratch/$USER/protein-design
SCRATCH_DIR=/cluster/scratch/$USER/protein-design
MODEL_DIR=/cluster/scratch/$USER/models
WANDB_PROJECT=protein-design-evotuning
```

Or run this one-liner (expands `$USER` automatically):

```bash
cat > .env <<EOF
DATA_DIR=/cluster/scratch/$USER/protein-design
SCRATCH_DIR=/cluster/scratch/$USER/protein-design
MODEL_DIR=/cluster/scratch/$USER/models
WANDB_PROJECT=protein-design-evotuning
EOF
```

Verify it looks right: `cat .env`

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

Check:
```bash
ls $SCRATCH_DIR/oas_raw/ | wc -l   # should be 2196
```

> Note: compute nodes require `eth_proxy` for internet access — this is already loaded in the sbatch script.

---

### Step 2 — Filter sequences

Reads all `.csv.gz` files and applies quality filters:
- **Productive** rearrangements only
- **ANARCI status**: rejects truncated, insertion-containing sequences
- **CDR3 present**: non-null, non-empty `cdr3_aa`
- **Minimum length**: ≥ 50 amino acids

Writes a single FASTA to `$SCRATCH_DIR/oas_filtered.fasta`.

```bash
sbatch bash_scripts/filter_oas.sbatch
```

Check progress while running (file size should grow):
```bash
wc -c $SCRATCH_DIR/oas_filtered.fasta
```

Expected: ~60–80% of raw sequences pass, resulting in tens of millions of sequences.

---

### Step 3 — Deduplicate with MMseqs2

Clusters sequences at **99% identity** using `mmseqs easy-linclust` and keeps
one representative per cluster. This removes clonal duplicates that would bias
training.

```bash
sbatch bash_scripts/mmseqs2.sbatch
```

Expected output: `$SCRATCH_DIR/oas_dedup_rep_seq.fasta`

Check:
```bash
grep -c "^>" $SCRATCH_DIR/oas_dedup_rep_seq.fasta
```

Expected: 30–60% reduction, leaving ~5–20M unique sequences.

> Steps 2 and 3 can be run in a single job: `sbatch bash_scripts/prepare_oas.sbatch`

---

### Step 4 — Debug run (recommended)

Before launching full training, run a quick end-to-end check on 10k sequences.
This samples from `oas_filtered.fasta`, deduplicates, and trains for 100 steps.
Uses an isolated scratch directory (`$SCRATCH_DIR/debug/`) — does not affect main data.

```bash
sbatch bash_scripts/debug_run.sbatch
```

Monitor:
```bash
tail -f logs/debug_<jobid>.err
```

A successful run ends with:
```
=== Debug run complete ===
```

Look for decreasing perplexity (e.g. 2.65 → 2.48) to confirm the model is learning.

---

### Step 5 — Full training

```bash
sbatch bash_scripts/evotuning.sbatch
```

Config: `configs/evotuning_base.yaml`
- Model: ESM2 650M (`facebook/esm2_t33_650M_UR50D`)
- 50,000 steps, batch size 32, learning rate 1e-5 with linear warmup
- MLM probability: 15% (standard BERT-style masking)
- Embeddings frozen, all transformer layers trainable
- Evaluation every 1000 steps, checkpoint every 5000 steps

Expected duration: **8–12 hours** on a 40GB GPU.

Monitor training:
```bash
tail -f logs/<jobid>.err        # live loss and perplexity
```

Or via the wandb dashboard at https://wandb.ai.

Checkpoints are saved to `$MODEL_DIR/step_<N>/checkpoint.pt`.

---

## Sbatch scripts reference

| Script | Purpose | GPU |
|---|---|:---:|
| `download_oas.sbatch` | Download 2196 OAS data units | No |
| `filter_oas.sbatch` | Filter sequences (productive, ANARCI, CDR3, length) | No |
| `mmseqs2.sbatch` | Deduplicate at 99% sequence identity | No |
| `prepare_oas.sbatch` | Filter + deduplicate in one job | No |
| `debug_run.sbatch` | End-to-end test on 10k sequences (100 training steps) | Yes (20GB) |
| `evotuning.sbatch` | Full ESM2 evotuning run | Yes (40GB) |

---

## Project structure

```
configs/                  Hyperparameter configs (YAML)
  evotuning_base.yaml     Full training config
  evotuning_debug.yaml    Debug config (100 steps, small batch)
scripts/                  Pipeline entry points
  download_oas.sh         Download helper (reads URL file)
  filter_oas.py           OAS filtering → FASTA
  run_mmseqs2.sh          MMseqs2 deduplication wrapper
  train_evotuning.py      Training entry point
bash_scripts/             SLURM job submission scripts
src/protein_design/
  model.py                ESM2 wrapper with layer freezing
  data.py                 FASTA dataset and DataLoader
  train.py                Training loop with gradient accumulation
  evaluate.py             Perplexity evaluation
```

---

## Troubleshooting

**`SCRATCH_DIR: Set SCRATCH_DIR env var`**
→ The `.env` file is missing or not being sourced. Check that `.env` exists and contains `SCRATCH_DIR`. All sbatch scripts source it automatically via `set -a; source .env; set +a`.

**`These module(s) exist but cannot be loaded: mmseqs2/14-7e284`**
→ MMseqs2 requires prerequisite modules. The sbatch scripts load them in the correct order (`stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284`). If running manually:
```bash
module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
```

**`wandb.errors.UsageError: No API key configured`**
→ Run `wandb login` on a login node before submitting training jobs. Your key is stored in `~/.netrc` and will be available to compute nodes.

**Downloads failing on compute nodes (all `[FAIL]`)**
→ Compute nodes have no internet by default. The `eth_proxy` module must be loaded — it is already included in `download_oas.sbatch` and all other scripts that need internet access.
