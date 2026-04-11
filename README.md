# ESM2 Direct Preference Optimization (DPO) Pipeline

This repository contains the pipeline for fine-tuning an ESM2 protein language model using **Direct Preference Optimization (DPO)**. It optimizes a policy ESM2 model against a frozen reference ESM2 model, encouraging the generation of sequences with higher binding enrichment scores (e.g., M22 or SI06 targets).

The pipeline automatically handles distance-2 (D2) mutational clustering, sequence pairing strategies, Masked Pseudo-Log-Likelihood (PLL) scoring specifically on mutated residues, and sequence perplexity evaluation.

All experiments are tracked via [Weights & Biases](https://wandb.ai) and configured dynamically using [Hydra](https://hydra.cc/). Instructions below are tailored for local execution and the **ETH Euler cluster**.

---

## ⚙️ Prerequisites & Setup

You can set up the environment using either `conda` (recommended for standard ML workflows) or `uv` (for ultra-fast Python package resolution).

### Option A: Using Conda (or Mamba/Micromamba)

```bash
conda create -n dpo_env python=3.10 -y
conda activate dpo_env

# Install PyTorch (adjust cuda version as needed for Euler)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```

### Option B: Using `uv` (Fast alternative)

```bash
# Create and sync the project environment from pyproject.toml
uv sync

# Include test dependencies (pytest)
uv sync --extra dev
```

### Weights & Biases Authentication
Make sure you are logged in to sync your runs online:
```bash
uv run wandb login
```

---

## 🚀 Running the Pipeline

### Data Preparation
The pipeline expects raw binding enrichment data (e.g., `M22_binding_enrichment.csv`) to be placed in `data/raw/`. 

Preprocessing and clustering (distance-2 neighborhood, `mut1`, `mut2` views) happen **automatically** the first time you run the training script. If you need to force a rebuild of the clustered data, append `data.force_rebuild=True`.

### Local Training
You can run the DPO training locally overriding parameters on the fly via Hydra:

```bash
# Basic run with default parameters
uv run python -m src.train_dpo

# Override pairing strategy and hyperparameters
uv run python -m src.train_dpo \
    data.pairing_strategy=positive_only_extremes \
    data.min_positive_delta=0.1 \
    data.min_delta_margin=0.5 \
    training.batch_size=64 \
    training.lr=5e-6

# Delta-based pairing with explicit boundary controls
uv run python -m src.train_dpo \
    data.pairing_strategy=delta_based \
    data.gap=0.6 \
    data.wt_pairs_frac=0.15
```

### ETH Euler Cluster Execution
For heavy jobs on the Euler cluster, submit the provided bash script. Check `bash_scripts/run_dpo_train.sh` to ensure it requests the correct hardware (e.g., A100/H100 GPUs) before submitting.

```bash
sbatch bash_scripts/run_dpo_train.sh
```

---

## 🧬 Key Features

- **Masked PLL DPO Loss**: The loss function strictly computes the Pseudo-Log-Likelihood (PLL) only for the specific residues that differ between the chosen and rejected sequences, preventing context swamping (`src/loss.py`).
- **Flexible Data Pairing**: Supports multiple strategies to build preference pairs from mutational clusters:
  - `positive_vs_tail`: Pairs the absolute best sequence against the absolute worst outside-in.
  - `positive_only_extremes`: Pairs all strictly positive sequences against the worst tail sequences.
  - `both_structured`: Combines `positive_vs_tail` and `positive_only_extremes` (deduplicated).
  - `delta_based`: Pairs by configurable rank gap and can add WT-boundary anchors.
- **Configurable Pair Controls**: Tune `min_positive_delta`, `min_delta_margin`, `gap`, and `wt_pairs_frac` via Hydra to control preference construction (`src/dataset.py`).
- **Eval Metrics**: Automatically computes test set Masked Sequence Perplexity ($\exp(-\frac{PLL}{N})$), implicit rewards, KL divergence, and reward accuracies (`src/eval.py`).

---

## 📁 Repository Structure

```text
├── conf/
│   └── config.yaml          # Hydra master configuration
├── data/
│   ├── raw/                 # Raw experimental data (e.g., M22_binding_enrichment.csv)
│   └── processed/           # Auto-generated D2 clustered datasets
├── src/
│   ├── data_processing.py   # Parsing & sequence extraction from raw data
│   ├── dataset.py           # DPO pair generation & splitting
│   ├── loss.py              # Masked DPO loss and monitoring metrics computation
│   ├── model.py             # ESM2PLLScorer wrapper for token/batch management
│   ├── eval.py              # Sequence perplexity and evaluation metrics
│   ├── train_dpo.py         # Main training loop & entrypoint
│   └── tests/               # Unit tests (Pytest)
└── bash_scripts/            # SLURM submission scripts for Euler
```
