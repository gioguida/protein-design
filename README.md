# protein-design

ESM2 evotuning pipeline for antibody design — continued masked language model
pretraining on human IgG heavy chain sequences from the OAS database.

## Setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync

# Configure environment
cp .env.template .env
# Edit .env with your paths (SCRATCH_DIR, MODEL_DIR, etc.)
```

## Data Pipeline

### 1. Download OAS data units

Prepare a text file with one OAS URL per line (human, heavy, IGHG), then:

```bash
bash scripts/download_oas.sh urls.txt
```

### 2. Filter sequences

```bash
python scripts/filter_oas.py
```

Applies quality filters (productive, ANARCI status, sequence length, CDR3
presence) and writes a single FASTA file.

### 3. Deduplicate with MMseqs2

```bash
bash scripts/run_mmseqs2.sh
```

Clusters at 99% sequence identity using MMseqs2 easy-linclust.

## Training

```bash
# Local (CPU/GPU)
python scripts/train_evotuning.py --config configs/evotuning_base.yaml

# Euler cluster
sbatch bash_scripts/evotuning.sbatch
```

## Project Structure

```
configs/           Hyperparameter configs (YAML)
scripts/           Entry points and data processing scripts
bash_scripts/      SLURM job submission scripts for Euler
src/protein_design/
  model.py         ESM2 wrapper with layer freezing
  data.py          FASTA dataset and DataLoader
  train.py         Training loop with gradient accumulation
  evaluate.py      Perplexity evaluation
```
