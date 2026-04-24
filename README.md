# protein-design

Task-based training for antibody design with shared config composition for:
- evotuning / C05 finetuning / TTT
- DPO

## Quickstart

```bash
# evotuning (defaults: task=evotuning, data=evo/oas_full, scoring=d2)
sbatch bash_scripts/train.sbatch evotuning

# C05 finetune from a previous checkpoint
sbatch bash_scripts/train.sbatch evotuning_c05 \
  model.init.source=checkpoint \
  model.init.checkpoint=$PROJECT_DIR/checkpoints/<run>/best.pt

# TTT from a checkpoint
sbatch bash_scripts/train.sbatch ttt \
  model.init.source=checkpoint \
  model.init.checkpoint=$PROJECT_DIR/checkpoints/<run>/best.pt

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
  task=evotuning_c05 \
  model.init.source=checkpoint \
  model.init.checkpoint=/path/to/best.pt
```

## Other entrypoints

```bash
# Standalone DPO script (defaults to task=dpo when task is omitted)
python scripts/train_dpo.py task=dpo training.batch_size=8 training.num_epochs=2

# Score a checkpoint
python scripts/eval.py +checkpoint=/path/to/best.pt scoring=d2
```
