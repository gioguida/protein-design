# protein-design

Task-based training for antibody design with shared config composition for:
- evotuning / C05 finetuning / TTT
- DPO

## Quickstart

```bash
# Pack a corpus once. Every masking policy reads the same pack.
sbatch bash_scripts/utils/pack_corpus.sbatch

# evotuning, masking policy selected by the data config
sbatch bash_scripts/train.sbatch evotuning data=evo/oas_cdr50
sbatch bash_scripts/train.sbatch evotuning data=evo/oas_hybrid

# the learning-rate sweep across dataset x policy x model size
bash_scripts/sweep_evotuning.sh --dry-run

# once a base run has finished, continue it under single-position masking
uv run scripts/analysis/branch_from_switch_point.py --run-dir $TRAIN_DIR/<run>

# TTT from a checkpoint
sbatch bash_scripts/train.sbatch ttt \
  model.init.source=checkpoint \
  model.init.checkpoint=$TRAIN_DIR/<run>/selected.pt

# DPO
sbatch bash_scripts/train.sbatch dpo
```

## Config layout

```text
conf/
  config.yaml          # single shared root (defaults, wandb, logging, hydra)
  data/
    evo/               # evotuning datasets
    dpo/               # dpo datasets
  model/               # model presets + init source/checkpoint
  task/                # task hyperparameter presets (including dpo)
  scoring/             # scoring presets (d2, none)
```

## Main config interface

- Select task/model/data/scoring from `conf/config.yaml` defaults.
- Model initialization is unified under:
  - `model.init.source: huggingface | checkpoint`
  - `model.init.checkpoint: <path or null>`

Example:

```bash
python scripts/train.py \
  task=evotuning \
  data=evo/oas_hybrid \
  model.init.source=checkpoint \
  model.init.checkpoint=/path/to/selected.pt
```

## Other entrypoints

```bash
# Standalone DPO script (defaults to task=dpo when task is omitted)
python scripts/train_dpo.py task=dpo training.batch_size=8 training.num_epochs=2

# Score a checkpoint
python scripts/eval.py +checkpoint=/path/to/best.pt scoring=d2
```
