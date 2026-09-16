import marimo

app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DATASET

    [X] downlaod OAS with only filters: human race and chain heavy

    [X] once downloaded apply filtering:
    - remove duplicate sequences (by exact match)
    - remove sequences missing the conserved cysteins that form the canonical disulfide bond in the Ig fold. For this we trust OAS's ANARCI_status flags "Missing Conserved Cysteine"
    - remove heavily fragmented sequences, specifically those missing more than 16 residues from the N-terminus or more than 7 residues from the C-terminus
    - replace any non-standard amino acid with an X token rather than dropping the sequence

    [X] deduplication: cluster by 95% sequence identity and keep one representative per cluster. Use Linclust instead of MMseqs2.


    [ ] drop sequences whose CDR-H3 can't be located in the VH. The CDR-H3 window comes from finding OAS's annotated `cdr3_aa` as a substring of the VH, which fails in three ways: the metadata row is missing, `cdr3_aa` is null, or the VH no longer contains it verbatim. The third case is our own doing: filtering rewrites non-standard residues in the VH as X but leaves `cdr3_aa` untouched, so a CDR-H3 carrying one stops matching. Clean `cdr3_aa` exactly the way the VH is cleaned before matching, then drop whatever still doesn't resolve, and record the drop rate.
    <claude>
    Drop them here, once, rather than inside each masking variant. Otherwise the whole-chain runs would train on a larger corpus than the CDR runs, the epoch lengths would differ, and the batch to single-position branch could no longer resume on the same shuffled order.
    </claude>

    ## paths:
    - raw downloaded oas:  $SCRATCH_DIR/oas_raw/ (*.csv.gz) - [230.9 GB]
    - filtered dataset: $SCRATCH_DIR/oas_filtered.fasta [366,792,632 seqs - 49.5 GB] and $SCRATCH_DIR/oas_filtered.parquet [366,792,632 seqs - 35.2 GB]
    - Linclust dedup: $SCRATCH_DIR/oas_dedup_rep_seq.fasta
    - extract dedup metadata: $SCRATCH_DIR/oas_dedup_meta.parquet
    - train/val/test splits:  $SCRATCH_DIR/oas_split_{train,val,test}_ids.parquet only contains seq_id not actual sequences and not metadata.
    - promotion to project dir:
        -  $PROJECT_DIR/data/oas/oas_dedup_rep_seq.fasta
        -  $PROJECT_DIR/data/oas/oas_dedup_meta.parquet
        -  $PROJECT_DIR/data/oas/oas_filtered.fasta
        -  $PROJECT_DIR/data/oas/oas_filtered.parquet
        -  $PROJECT_DIR/data/oas/oas_split_{train,val,test}_ids.parquet


    <claude>
    ## Packed corpus

    Training reads a packed, memory-mapped copy of the corpus instead of the FASTA. This is not an optimization, it's what makes OAS-scale training possible at all: holding every sequence in memory as a fixed-width array costs tens of GB for the train split alone, a per-sequence window lookup keyed by seq_id costs about as much again, and every run currently pays a full double scan of a 29 GB FASTA before it starts.

    Built once per (corpus, flank). Stores, in corpus order: the packed residues and their offsets, the sequence ids, the CDR-H3 start/end, the ±flank window, and the train/val/test assignment. Random access is then an offset lookup into a memory map and the loader's memory stays flat. Non-standard positions need nothing stored: they survive filtering as the letter X, so they are identifiable straight from the packed bytes and are skipped when positions are selected.

    Build it at flank 5, the widest any variant needs. The no-flank region that step 1 masks is recovered from the stored CDR-H3 start/end, so one pack serves every masking variant.

    ## paths:
    - build script: `scripts/data_prep/pack_corpus.py`
    - pack: $SCRATCH_DIR/packed/\<corpus\>_flank5/

    ## LR-sweep subsample

    The learning-rate sweep runs on a random subsample of the OAS train split, not the full corpus. A single epoch over the full dedup corpus is roughly a day of A100 time for the 35m model and several times that for 150m, and the sweep is 120 runs, so the full corpus is not affordable as a sweep axis. Subsampling also makes the early points of the step 3 eval grid meaningful instead of landing inside the first batch.

    Only the train split is subsampled. Val and test stay whole, so every run is scored against the same held-out data.

    N = 10M.

    Once the lr is picked, the final runs go on the full OAS train split at that lr.
    </claude>


    ## WT-similar-set

    Biswas et al. (2021, eUniRep) build their "WT-similar" set with jackhmmer (profile-HMM search) plus a fixed edit-distance/length filter. That recipe doesn't transfer well to antibodies: jackhmmer's profile assumes family-wide conservation, which breaks down on CDR-H3 (hypervariable, no canonical structure), and a fixed similarity threshold breaks down whenever the WT is a CDR-H3 outlier in OAS. We instead use a procedure built for that failure mode, general-purpose across WTs (not C05-specific):

    **Step 1: rank by CDR-H3 similarity.** Score every OAS candidate's CDR-H3 against the WT's via BLOSUM62 global alignment, normalized by the WT's self-alignment score (same method as `extract_c05_cdrh3_blosum.py`).

    **Step 2: select top-N.** Take the top-N ranked sequences rather than a fixed similarity threshold. This always returns a same-size corpus regardless of how populated OAS is around that WT, instead of silently returning too few (or zero) hits for an outlier WT. N=5000 for now.

    **Step 3: report, don't gate.** Record the min/median/max normalized score of the selected top-N as a diagnostic.

    **Step 4: train/val/test split.** The selected sequences are themselves real OAS seq_ids, each already carrying a deterministic split assignment from the main corpus (`split_for`, same logic as `oas_split_{train,val,test}_ids.parquet`). Reuse that instead of defining a new split: intersect the top-N with the existing split files. Keeps every sequence's train/val/test role globally consistent across the project. On N=5000 this gives roughly 4500/250/250.

    First WT to run this on: C05 (`src/protein_design/constants.py`).

    <claude>
    Training on this set has to keep the same split salt and the same 90/5/5 ratios as the main corpus. The split is derived from the seq_id hash, so any other setting silently produces a different partition than the one recorded below and the set stops being consistent with the rest of the project.

    This set is never subsampled. At 5000 sequences it is already the small end of the dataset axis.
    </claude>

    ## paths:
    - build script (general-purpose, any WT): `scripts/data_prep/build_wt_similar_set.py`
    - scratch caches (per WT, reused across N/threshold re-runs):
        - unique-H3 -> seq_ids mapping: $SCRATCH_DIR/wt_similar/\\<wt-name\\>/h3_mapping.pkl
        - per-unique-H3 normalized BLOSUM score: $SCRATCH_DIR/wt_similar/\\<wt-name\\>/h3_scores.parquet
    - final outputs, promoted straight to project dir (no separate scratch->project promotion step):
        - selected top-N FASTA: $PROJECT_DIR/data/wt_similar/\\<wt-name\\>/\\<wt-name\\>_wt_similar_top\\<N\\>.fasta
        - per-sequence normalized scores: $PROJECT_DIR/data/wt_similar/\\<wt-name\\>/\\<wt-name\\>_wt_similar_top\\<N\\>_scores.csv
    - train/val/test split script (general-purpose, any WT): `scripts/data_prep/export_wt_similar_split_ids.py`
    - train/val/test split ids: $PROJECT_DIR/data/wt_similar/\<wt-name\>/\<wt-name\>_wt_similar_top\<N\>_{train,val,test}_ids.parquet
    - C05 run (wt-name=c05, N=5000):
        - $PROJECT_DIR/data/wt_similar/c05/c05_wt_similar_top5000.fasta [5,000 seqs]
        - $PROJECT_DIR/data/wt_similar/c05/c05_wt_similar_top5000_scores.csv
        - score distribution: min=0.261 median=0.276 max=0.433 (sanity-checked against a random-shuffle null: p99.99=0.082, max=0.187 over 200k shuffled real CDR-H3s -- the selection is real signal, not noise)
        - $PROJECT_DIR/data/wt_similar/c05/c05_wt_similar_top5000_{train,val,test}_ids.parquet [4,549 / 228 / 223 seqs]
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # TRAINING

    ## Models

    | HF checkpoint | params | layers | embed dim |
    | - | - | - | - |
    | esm2_t6_8M_UR50D | 8m | 6 | 320 |
    | esm2_t12_35M_UR50D | 35m | 12 | 480 |
    | esm2_t30_150M_UR50D | 150m | 30 | 640 |

    ## Hyperparams

    <claude>
    Everything except the learning rate is fixed, following Talaei et al.'s unpaired pretraining stage. That is the stage that matches ours (continued MLM on single-chain OAS), not their paired fine-tuning stage, which uses different values.

    | setting | value |
    | - | - |
    | optimizer | AdamW, weight decay 0.01 |
    | warmup | ratio 0.05 of total steps |
    | epochs | 1 |
    | global batch | 512 sequences |
    | precision | bf16 |
    | embeddings | frozen |

    warmup is a ratio, not a fixed step count, so it tracks corpus size and stays comparable across the lr grid and between the subsampled and full-corpus runs.

    Frozen embeddings are a deviation from the paper, which trains everything. Kept for consistency with the runs already done.
    </claude>

    - **sweep**: model size × lr, {8m, 35m, 150m} × {5e-6, 1e-5, 2e-5, 5e-5, 1e-4}

    <claude>
    lr is the only swept hyperparameter. The grid brackets both values the paper uses: 1e-5 for unpaired pretraining, 2e-5 for paired fine-tuning.
    </claude>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masking strategy

    <claude>
    Throughout this section "CDR" means **CDR-H3 only**, not all three CDR loops. The downstream task is the C05 DMS, which only varies H3, so masking H1 and H2 spends budget on positions the downstream task can't use. This is a narrowing of Talaei et al.'s definition and it has a consequence for the masking rate, noted under step 1.

    X positions (non-standard residues rewritten during filtering) are never eligible to be masked, in any variant, and are excluded from the recovery accuracies in step 3. Masking them would train and score the model on predicting X.
    </claude>

    <claude>
    **Step 1: whole-chain warm-up.** One WC-15% run per (dataset, model size, lr), from the base pretrained model, covering the first part of the epoch. Both base runs branch from its final checkpoint, which is also the reference the switch-point rule measures them against.

    Why it's here. The rule holds framework recovery inside a 0.1 percentage point band around a reference. That band only means something if the reference sits where framework accuracy has stopped moving on its own. Measured against the base pretrained model it does not: base ESM2 has never seen an antibody, so the opening steps of any antibody training move framework recovery by whole percentage points through plain domain adaptation, and a 0.1pp band excludes everything past the first few steps by construction. On 200k OAS sequences at the sweep's batch size, CDR-50% moves framework recovery 1.65pp by a tenth of an epoch and 5.4pp by a fifth; Hybrid moves it the other way, gaining 22pp on the WT-similar set. Both directions blow the band, and the rule ends up selecting a model that has learned nothing.

    This is what Talaei et al. do. Their Stage II always branches from a Stage I whole-chain checkpoint and is measured against it, never against the base model. Warming up first restores that.

    It also makes the comparison cleaner. CDR-50% and Hybrid branch from the same checkpoint, so they differ only in masking policy rather than in each policy's own adaptation transient.

    Length: {STILL OPEN} — set it where framework recovery under WC-15% flattens, measured on a whole-chain run over the sweep corpus. Everything stays inside one epoch: the warm-up takes the first part of it, the base runs resume mid-epoch on the same shuffled order through the skip_samples cursor and finish it. Talaei et al.'s own unpaired stage is a single epoch, and they seed downstream training from its 0.3-epoch checkpoint, so a fraction of an epoch is the expected order of magnitude rather than a whole one.

    **Step 2: two base runs**, resumed from the Step 1 warm-up, single-chain (unpaired) OAS, Talaei et al.'s definitions:
    </claude>

    | base run | definition |
    | - | - |
    | CDR-50% | mask 50% of residues within annotated CDR loops only (IMGT-defined), block/simultaneous masking |
    | Hybrid | <claude>within every batch, 80% of the samples get CDR-50% masking and 20% get whole-chain masking (WC-15%: standard 15% random masking across the full sequence, 80/10/10 mask/random/keep split)</claude> |

    <claude>
    Both variants use the 80/10/10 mask/random/keep split, and both mask every selected position in the same forward pass.

    The hybrid split is over training samples inside each batch, not over batches: "80% of training samples use CDR masking and 20% use WC masking within each batch". It is also exact rather than an independent per-sample coin flip, so every batch carries the same mixture.

    On the 50%: Talaei et al. chose 50% to match the absolute number of residues WC-15% masks, counted over all six CDRs of a paired VH+VL. On heavy-chain H3 alone that match is unreachable. On this corpus WC-15% masks 16.3 residues per sequence on average while H3 is only 15.6 residues long, so even masking all of H3 falls short. (For comparison, the rate that would match the count across all three heavy CDRs is 51%, which is a good sign that 50% really was a count match in their setting.)

    We keep the literal 50% anyway, so the arm means what its name says and stays directly comparable to the paper. The cost is that CDR-50% masks roughly half the residues per sequence that WC-15% does, so the two arms are not signal-matched. Log masked residues per sequence for every run so this is visible in the results rather than implied.
    </claude>

    **Step 3: branch each base run into 2 variants**. When the base run hit the switch-point rule (defined in step 4), we switch from batch masking to sinlge masking. The switch applies in the same way for CDR-50% and Hybrid → **4 trained models total.**

    **Step 4: stopping / switch-point rule**, self-supervised (no DMS labels), applied twice (once per base run, and again inside each batch→single branch):
    1. Track CDR(avg) and FR(avg): mean token-level masked-recovery accuracy within CDR vs. framework regions (IMGT-annotated, single chain), on held-out OAS validation, evaluated at [0.01%, 0.1%, 1%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%] of epoch. <claude>Where a fraction lands below one optimizer step it is clamped to one step and duplicates are dropped, so a small corpus simply gets a shorter grid.</claude>
    2. **Batch-phase switch/stop point:** earliest checkpoint maximizing CDR(avg), subject to <claude>|FR(avg)_reference − FR(avg)_checkpoint| ≤ 0.1 percentage points, with **the Step 1 warm-up checkpoint** as the reference. The constraint is two-sided, matching the paper: a checkpoint is ineligible if framework accuracy has moved too far in either direction, not only if it has dropped.</claude> Doubles as the final stop for batch-only variants and the branch point for batch→single variants.
    3. **Final stop point (batch→single variants):** same rule, re-applied under single-position masking, relative to the branch-point checkpoint's FR(avg).
    4. Aggregate MLM loss/perplexity: tracked as a diagnostic only, never used to pick a checkpoint.
    5. Post-hoc, non-decision check: compare selected checkpoints against whichever checkpoint would have maximized zero-shot Spearman correlation on held-out DMS labels — reported only, never fed back into checkpoint selection.

    <claude>
    Neither phase terminates early. A run trains its full epoch and the switch point is read off the eval curve afterwards, because "earliest maximizing" can't be decided until the later points exist. A checkpoint is kept at every eval point, which also means the 0.1pp tolerance can be re-checked at other values later without retraining anything.

    Talaei et al. only apply this rule to their second stage. Their first-stage anchor (epoch 5) is designated rather than derived, and the paper gives no rule for it. Applying the rule to the single-position phase as well is our own extension, not a reproduction. The warm-up itself is not selected by the rule, for the same reason: there is nothing before it to measure against.
    </claude>

    **Step 5: single-position phase** (batch→single branches only, after the Step 4 switch point):
    - Batching: unchanged from Step 2, same batch_size (sequences), same shuffled epoch order, resumed from Step 2's exact stopping point via the existing skip_samples cursor<claude>. One example per sequence, exactly as in the earlier phases, so epoch length and shuffle order are identical throughout and the cursor stays meaningful</claude>
    - Per sequenc: pool = all CDR positions (fixed) + flank positions, 5 each side (fixed) + round(0.3 × N_fw) framework positions, sampled without replacement from all N_fw, freshly redrawn each occurrence. <claude>X positions are never in the pool.</claude>
    - Per forward pass: one position drawn uniformly from that occurrence's pool (same seed), masked; everything else visible.
    - Branch parity: CDR→single and Hybrid→single mask the same positions in Step 5.

    <claude>
    Step 2 masks CDR-H3 with no flank, step 5 adds a fixed 5-residue flank on each side. The two phases deliberately disagree on the region boundary: step 2 reproduces the paper's policy, step 5 matches how we actually score a point mutation.
    </claude>

    **total number of train jobs:** <claude>$75 = 5 \space \text{phases} \cdot 3 \space \text{model sizes} \cdot 5 \space \text{lrs}$ (1 shared whole-chain warm-up + 2 base batch runs + 2 single-position branch continuations, full cross product with the model size × lr sweep). The warm-up is shared by both base runs of its cell, so it adds one job per cell rather than two.</claude>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## WT similarity
    In addition to evotuning training another thing that could be helpful for the downstream task is only letting the model see antibody sequences similar to the WT at hand. We can then use the WT-similar-set that is already a selection of these kind of sequences. the same exact masking strategy should be applied also for this selected dataset. by doing this we end up with:

    **total number of train jobs:** <claude>$150 = 2 \space \text{dataset} \cdot 5 \space \text{phases} \cdot 3 \space \text{model sizes} \cdot 5 \space \text{lrs}$ (full cross product: phase × dataset × model size × lr)</claude>

    <claude>
    These 150 don't run in parallel. Each phase needs the one before it: the base runs resume from the warm-up's checkpoint, and a single-position branch can't start until its base run has finished the epoch and the switch point has been read off the curve. So the grid runs as three sequential waves — 30 warm-ups, then 60 base runs, then 60 continuations.

    All 150 are the lr sweep, so the OAS arm uses the subsample. After the lr is chosen, the final runs go on the full OAS train split: 5 phases × 3 model sizes = 15 runs, again in three waves.

    Nothing selects a single winner across the grid. DPO on the C05 DMS runs on top of all of them, and the comparison between training strategies is the result.

    One caveat on the WT-similar arm. At 4549 training sequences and a 512-sequence batch it is about 9 optimizer steps per epoch, so most of the evaluation grid collapses onto the same few steps and its selection curve is far coarser than the OAS arm's. The two datasets are therefore not equally well resolved, which matters when reading the comparison between them.
    </claude>
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
