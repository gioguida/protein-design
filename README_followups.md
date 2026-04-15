# Follow-up finetuning runs + results plots

After the OAS evotuning job produces a `best.pt` (lowest val perplexity,
auto-tracked by `train.py`), launch three downstream combinations and
generate comparison plots for the supervisor meeting.

All three variants start from the same evotuned `best.pt`:

| Variant    | Chain                                   | Script                                      |
|------------|-----------------------------------------|---------------------------------------------|
| C05 only   | evotuned → C05-5k finetune              | `bash_scripts/evotuning.sbatch`             |
| TTT only   | evotuned → TTT on single C05 CDRH3      | `bash_scripts/ttt_c05.sbatch`               |
| C05 + TTT  | evotuned → C05-5k → TTT (chained)       | `bash_scripts/c05_then_ttt.sbatch`          |

All configs are fixed to `esm2_t12_35M_UR50D` so checkpoints are interchangeable.

Filled-in configs are written to `configs/_generated/` (gitignored) so the
originals stay as templates with `REPLACE_WITH_*` placeholders, and each
launch keeps a record of the exact config used.

---

## 0. One-time setup

```bash
cd /cluster/home/mdenegri/protein-design
set -a; source .env; set +a

export EVOTUNING_RUN=oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859

# The evotuning job writes best.pt to $TRAIN_DIR (scratch) during training.
# The archive to $PROJECT_DIR/checkpoints/ only happens when training finishes.
# If the job is still running (or was killed), manually archive best.pt first:
mkdir -p "$PROJECT_DIR/checkpoints/$EVOTUNING_RUN"
cp "$TRAIN_DIR/$EVOTUNING_RUN/best.pt" "$PROJECT_DIR/checkpoints/$EVOTUNING_RUN/best.pt"

# Verify:
ls -lh "$PROJECT_DIR/checkpoints/$EVOTUNING_RUN/best.pt"
```

---

## 1. Launch the three follow-ups

```bash
# --- C05 only (evotuning on 5k C05 sequences, seeded from evotuned best) ---
sed "s|REPLACE_WITH_EVOTUNING_RUN_NAME|$EVOTUNING_RUN|g" \
    configs/evotuning_c05_5k.yaml \
    > configs/_generated/c05_5k_from_${EVOTUNING_RUN}.yaml

sbatch bash_scripts/evotuning.sbatch \
    configs/_generated/c05_5k_from_${EVOTUNING_RUN}.yaml \
    c05_5k_from_${EVOTUNING_RUN}

# --- TTT only (TTT on single CDRH3, seeded from evotuned best) ---
sed "s|REPLACE_WITH_EVOTUNING_RUN_NAME|$EVOTUNING_RUN|g" \
    configs/ttt_c05_single.yaml \
    > configs/_generated/ttt_only_from_${EVOTUNING_RUN}.yaml

sbatch bash_scripts/ttt_c05.sbatch \
    configs/_generated/ttt_only_from_${EVOTUNING_RUN}.yaml \
    ttt_only_from_${EVOTUNING_RUN}

# --- C05 + TTT (chained in a single SLURM job) ---
sbatch bash_scripts/c05_then_ttt.sbatch $EVOTUNING_RUN
```

Check status:
```bash
squeue -u $USER
tail -f logs/*.out
```

Expected run dirs after all three finish (each gets a `_YYYYmmdd_HHMMSS`
suffix appended by the entrypoint scripts):

```
$PROJECT_DIR/checkpoints/c05_5k_from_${EVOTUNING_RUN}_<ts>/best.pt
$TRAIN_DIR/ttt_only_from_${EVOTUNING_RUN}_<ts>/checkpoints/final.pt
$TRAIN_DIR/ttt_from_c05_5k_from_${EVOTUNING_RUN}_<ts1>_<ts2>/checkpoints/final.pt
```

Resolve the exact paths with:
```bash
ls -td $PROJECT_DIR/checkpoints/c05_5k_from_${EVOTUNING_RUN}_*/ | head -1
ls -td $TRAIN_DIR/ttt_only_from_${EVOTUNING_RUN}_*/             | head -1
ls -td $TRAIN_DIR/ttt_from_c05_5k_from_${EVOTUNING_RUN}_*/      | head -1
```

---

## 2. Tonight's smoke test (before leaving)

Dry-run all three against the latest intermediate evotuning checkpoint to
catch path / config / plotting bugs early.

```bash
SMOKE_NAME=smoke
LATEST_STEP=$(ls -t $TRAIN_DIR/<evotuning_run_in_progress>/checkpoints/step_*.pt | head -1)
mkdir -p $PROJECT_DIR/checkpoints/$SMOKE_NAME
ln -sf "$LATEST_STEP" $PROJECT_DIR/checkpoints/$SMOKE_NAME/best.pt

EVOTUNING_RUN=$SMOKE_NAME  # re-run section 1 with this value
```

Let each job start, confirm it logs the first training step and the
`finetune:` path is loaded (look for "Loading finetune checkpoint: ..."
in the .out log), then `scancel` them.

---

## 3. Generate plots

Once the three follow-up runs have finished:

```bash
EVO_DIR=$TRAIN_DIR/$EVOTUNING_RUN
C05_DIR=$(ls -td $TRAIN_DIR/c05_5k_from_${EVOTUNING_RUN}_*/              | head -1)
TTT_DIR=$(ls -td $TRAIN_DIR/ttt_only_from_${EVOTUNING_RUN}_*/            | head -1)
C05TTT_DIR=$(ls -td $TRAIN_DIR/ttt_from_c05_5k_from_${EVOTUNING_RUN}_*/  | head -1)
OUT_DIR=$PROJECT_DIR/plots/meeting_$(date +%Y%m%d_%H%M%S)

sbatch bash_scripts/plot_results.sbatch \
    --run "$EVO_DIR"    --label evotuned \
    --run "$C05_DIR"    --label "+C05" \
    --run "$TTT_DIR"    --label "+TTT" \
    --run "$C05TTT_DIR" --label "+C05+TTT" \
    --scoring-config configs/evotuning_base.yaml \
    --out-dir "$OUT_DIR"
```

Outputs in `$OUT_DIR/`:
- `spearman_bar.png` — grouped bars per dataset × model, p-value stars above each bar (`*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` otherwise). Uses the "average" mutational-path strategy.
- `perplexity_comparison.png` — final validation perplexity per model (base ESM2 evaluated once on the OAS val split for reference).
- `metrics_summary.csv` — long-format table with rho, pval, strategy, final perplexity per (model × dataset).
- `manifest.txt` — which run dir was used for each label.

Copy them off the cluster with:
```bash
scp -r euler:$OUT_DIR ./
```

---

## Notes

- **Best checkpoint selection:** `train.py` auto-tracks the lowest val-perplexity checkpoint as `best.pt` and mirrors it to `$PROJECT_DIR/checkpoints/<run>/best.pt` (persistent). No manual picking needed.
- **TTT output:** `train_ttt.py` saves `checkpoints/final.pt` only (no best.pt); `plot_results.py` falls back to `final.pt` if `best.pt` is missing.
- **Re-scoring:** `plot_results.py` uses scoring results cached in each run's `metrics.json`. Pass `--force-rescore` to ignore the cache and re-score from the checkpoint.
- **Base ESM2 reference:** included automatically in the plots; pass `--skip-base` to omit.
