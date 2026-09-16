# Training and experimental machinery inventory

Repository audit for planning the final report. This document separates (a) implemented methods and experiment drivers, (b) artifacts present in this checkout, and (c) figures that can be regenerated from the analysis pipeline but are not present locally.

## 1. Main story the code supports

The project adapts ESM2 to design C05 influenza-antibody CDR-H3 variants through several stages and comparisons:

1. **Domain adaptation:** continue masked-language-model training on antibody/OAS sequence corpora, with alternative CDR-focused masking and data-neighborhood choices.
2. **Target specialization:** fine-tune toward C05-like sequences; optionally apply one-sequence test-time training (TTT).
3. **Preference optimization:** train full-model DPO or parameter-efficient LoRA-DPO from experimental binding preferences; compare alternative pair construction, data budgets, and an unlikelihood objective.
4. **Generation and baselines:** generate CDR-H3 candidates with Gibbs sampling or stochastic beam search; compare against PSSM and random library-mutant samplers.
5. **Evaluation:** measure sequence-model likelihood/naturalness, agreement with DMS binding measurements, sampling diversity/novelty, and representation changes.

The main reusable entry point is [`scripts/train.py`](../scripts/train.py), which dispatches via Hydra to the runners in [`src/protein_design/train_dispatch.py`](../src/protein_design/train_dispatch.py). Sampling and analysis use separate CLI scripts under `scripts/`.

## 2. Methods and experimental levers

| Component | What is implemented | Main config / entrypoints | Useful outcomes to report |
|---|---|---|---|
| ESM2 scoring and context handling | ESM2 wrappers, sequence context assembly, masked pseudo-log-likelihood (PLL), pseudo-perplexity and checkpoint loading. 35M and 650M model presets are present. | `conf/model/`; `src/protein_design/model.py`, `eval.py`, `checkpoint_loading.py`; `scripts/eval.py` | PLL/pseudo-perplexity; model size comparison; CDR-H3 vs framework behavior. |
| OAS evotuning | Continued MLM on antibody corpora with reproducible data splits, checkpoint/resume, scheduled evaluation and checkpoint selection. | `conf/task/evotuning.yaml`, `conf/data/evo/`; `src/protein_design/evotuning/`; `scripts/train.py` | Train/validation loss and perplexity; CDR/framework accuracy; DMS Spearman before/after adaptation. |
| Masking ablations | Whole-chain 15% (`wc15`), CDR-H3 50% (`cdr50`), mixed batches (`hybrid`), and single-position continuation (`single_pool`). CDR flank and masking probabilities are data-configurable. | `conf/data/evo/*.yaml`; `src/protein_design/evotuning/data.py` | Compare downstream DMS ranking and CDR pseudo-perplexity; describe the mask policy explicitly. |
| C05-neighborhood adaptation | C05-focused FASTA choices include VH identity, CDR-H3 identity, MMseqs, BLOSUM, WT-similar, and reference/single-sequence variants. | `conf/data/evo/c05_*.yaml`; `scripts/data_prep/search_c05.py`, `extract_c05_*`, `build_wt_similar_set.py` | Dataset size and similarity distribution; effect of sequence-neighborhood definition. |
| TTT | A short single-reference MLM stage for C05. The provided preset uses LoRA adapters, freezes the language-model head, and records snapshots across update steps. | `conf/task/ttt.yaml`; `scripts/train.py`; `src/protein_design/evotuning/train.py` | Change in CDR PLL/perplexity and DMS Spearman as update steps increase; compare base/evo/C05 seeds. |
| Full-parameter DPO | Reference-model preference optimization from DMS-derived chosen/rejected pairs. Supports standard and weighted DPO losses, configurable beta/temperature, schedulers, checkpoint selection, patience, resume and test diagnostics. | `conf/task/dpo.yaml`, `conf/data/dpo/default.yaml`; `src/protein_design/dpo/`; `scripts/train_dpo.py` | Validation reward accuracy/margin, loss, implicit KL, chosen-sequence perplexity, held-out DMS Spearman. |
| LoRA-DPO | DPO with trainable low-rank adapters and explicit frozen-reference behavior; same broad loss and evaluation options as full DPO. | `conf/task/lora_dpo.yaml`; `src/protein_design/lora_dpo/train.py`; `scripts/train_lora_dpo.py` | Compare held-out ranking and naturalness against full DPO, with parameter efficiency stated. |
| Preference data design | Delta-based pairs with configurable cross, WT-anchor, within-positive and within-negative components, score margins/thresholds, split constraints, excluded winner mutation positions, and deterministic low-data subsampling. | `conf/data/dpo/default.yaml`; `src/protein_design/dpo/data_processing.py`, `dataset.py`, `splitting.py`, `low_data.py` | Pair-count/composition table; low-data learning curve; robustness to pair strategy / position exclusions. |
| Unlikelihood training | MLM plus an unlikelihood penalty over configured unwanted residues/positions, with alpha controlling penalty strength. | `conf/task/unlikelihood.yaml`; `src/protein_design/unlikelihood/`; `scripts/train_unlikelihood.py` | Report as an alternative/negative-control objective; evaluate binding ranking and perplexity together. |
| PSSM sampler | Position-wise, frequency-weighted CDR-H3 sampling from a DMS training split, temperature-controlled. | `src/protein_design/pssm_baseline.py`; `scripts/pssm_sampling.py`; `scripts/analysis/run_pssm_baseline_sweep.py` | Baseline binding-score distribution, diversity, novelty, edit distance and temperature response. |
| Random library-mutant sampler | WT-centered mutations uniformly sampled from residues observed at each position; trust radius caps edits. Shares split resolution/helpers with PSSM for a fair baseline. | `src/protein_design/random_baseline.py`; `scripts/random_sampling.py`; `scripts/analysis/run_random_baseline_sweep.py` | Baseline against PSSM and DPO generation using matched library size and edit-distance caps. |
| Gibbs generation | Iterative single-position masked resampling with temperature, chains, burn-in/snapshots, initialization mode and optional mutation-distance constraints. | `scripts/gibbs_sampling.py`; `scripts/analysis/run_gibbs_temperature_sweep.py`; `conf/analysis/gibbs_temperature_sweep.yaml` | PLL trajectories, DMS score, entropy, novelty and diversity vs temperature and model. |
| Stochastic beam search | Iterative masked-position expansion with stochastic selection among beam candidates; can start from WT, DMS pool or top DMS seeds. | `scripts/stochastic_beam_search.py`; `scripts/analysis/run_temperature_sweep.py`; `conf/analysis/temperature_sweep.yaml` | Same quality/diversity/novelty measures as Gibbs; report seed source, beam size, steps and temperature. |
| Embedding and representation analysis | PLL PCA, per-model PCA, difference-vector PCA, CKA, Procrustes displacement and OAS UMAP drivers; model/dataset comparisons are config-driven. | `conf/analysis/full_analysis.yaml`; `scripts/analysis/compute_*.py`, `plot_*pca.py`, `compare_models.py` | Show whether tuning changes sequence-space organization, not only scalar ranking metrics. |

### Data and evaluation substrate

- **Antibody sequence data:** OAS download/filter/metadata/dedup/packing tools are under `scripts/data_prep/`. Filtering and C05-neighborhood extraction are separate steps; the available config presets distinguish whole VH and CDR-H3 similarity definitions. The report should state the exact corpus variant, sequence counts, deduplication threshold, and split used by the checkpoint being discussed.
- **Experimental data:** the DMS configuration includes C05 ED1/ED2/ED5/ED8–11 M22 panels, ED2/ED5 SI06 panels, an expression panel, and a Cetuximab heavy-chain AbAgym dataset. The central C05 comparisons use M22 and SI06 binding enrichment; the actual panel and strain should be labeled on every result. Split caching and stratification are implemented in `src/protein_design/dms_splitting.py` and the DPO split utilities.
- **Core metrics:** Spearman rank correlation between PLL and enrichment; PLL/pseudo-perplexity; DPO reward accuracy/margin and implicit KL; generated-sequence DMS scores; edit distance, per-position entropy, pairwise Hamming diversity, novelty, and score/diversity trade-offs.
- **Scorer baselines:** analysis scripts can score with the trained model, supervised ESME ensemble/uncertainty machinery, and experimental DMS truth. Keep a supervised predictor baseline distinct from an unsupervised PLL result.

## 3. Plot and analysis machinery

### A. Figures present in this checkout

There are currently **49 files under `plots/`: 34 PNG figures and 15 CSV diagnostics**. Links below point to the local assets.

**DMS data characterization (4 figures)**

- [count vs quality scatter](../plots/count_quality_scatter.png)
- [enrichment histogram](../plots/enrichment_histogram.png)
- [mean enrichment per position](../plots/mean_enrichment_per_position.png)
- [position/amino-acid heatmap](../plots/position_amino_acid_heatmap.png)

**M22 enrichment distributions by edit-distance panel (12 figures)**

- ED2: [enrichment](../plots/ED2_M22_binding_enrichment_M22_enrichment_distribution.png), [count: ED5 comparison](../plots/ED2_M22_binding_enrichment_count_ED2Ed5_distribution.png), [count: M22 positive](../plots/ED2_M22_binding_enrichment_count_ED2M22pos_distribution.png), [log positive/negative count ratio](../plots/ED2_M22_binding_enrichment_log_count_ED2M22pos_over_count_ED2M22neg_distribution.png).
- ED5: [enrichment](../plots/ED5_M22_binding_enrichment_M22_enrichment_distribution.png), [count: ED5](../plots/ED5_M22_binding_enrichment_count_ED5Ed5_distribution.png), [count: M22 positive](../plots/ED5_M22_binding_enrichment_count_ED5M22pos_distribution.png), [log positive/negative count ratio](../plots/ED5_M22_binding_enrichment_log_count_ED5M22pos_over_count_ED5M22neg_distribution.png).
- ED8–11: [enrichment](../plots/ED811_M22_enrichment_full_M22_enrichment_distribution.png), [count: ED8–11](../plots/ED811_M22_enrichment_full_count_ED811Ed811_distribution.png), [count: round-1 positive](../plots/ED811_M22_enrichment_full_count_ED811M22r1pos_distribution.png), [log positive/negative count ratio](../plots/ED811_M22_enrichment_full_log_count_ED811M22r1pos_over_count_ED811M22neg_distribution.png).

**Entropy/temperature and preference-pair diagnostics (18 figures + 15 CSVs)**

- Panel-level heatmaps: [ED2](../plots/ED2_train_temp_entropy_heatmap.png), [ED5](../plots/ED5_train_temp_entropy_heatmap.png), [ED8–11](../plots/ED811_train_temp_entropy_heatmap.png).
- DPO input diagnostics: [chosen sequence position entropy](../plots/dpo_train_chosen_temp_entropy_heatmap.png), [rejected sequence position entropy](../plots/dpo_train_rejected_temp_entropy_heatmap.png), [chosen/rejected pair difference](../plots/dpo_train_pair_difference_heatmap.png). Companion CSVs: [chosen entropy](../plots/dpo_train_chosen_position_entropy.csv), [rejected entropy](../plots/dpo_train_rejected_position_entropy.csv), [pair difference fractions](../plots/dpo_train_pair_difference_fraction.csv).
- Position-removal sensitivity: each cell links the heatmap and its per-position entropy CSV. The four variants are all positions, exclude 8–9, exclude 15–16, and exclude both pairs.

| Split | All positions | Exclude 8–9 | Exclude 15–16 | Exclude 8–9 and 15–16 |
|---|---|---|---|---|
| Train | [PNG](../plots/remove_positions/dpo_train_chosen_all_positions_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_train_chosen_all_positions_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_train_chosen_exclude_8_9_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_train_chosen_exclude_8_9_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_train_chosen_exclude_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_train_chosen_exclude_15_16_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_train_chosen_exclude_8_9_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_train_chosen_exclude_8_9_15_16_position_entropy.csv) |
| Validation | [PNG](../plots/remove_positions/dpo_val_chosen_all_positions_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_val_chosen_all_positions_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_val_chosen_exclude_8_9_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_val_chosen_exclude_8_9_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_val_chosen_exclude_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_val_chosen_exclude_15_16_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_val_chosen_exclude_8_9_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_val_chosen_exclude_8_9_15_16_position_entropy.csv) |
| Test | [PNG](../plots/remove_positions/dpo_test_chosen_all_positions_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_test_chosen_all_positions_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_test_chosen_exclude_8_9_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_test_chosen_exclude_8_9_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_test_chosen_exclude_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_test_chosen_exclude_15_16_position_entropy.csv) | [PNG](../plots/remove_positions/dpo_test_chosen_exclude_8_9_15_16_temp_entropy_heatmap.png) · [CSV](../plots/remove_positions/dpo_test_chosen_exclude_8_9_15_16_position_entropy.csv) |

These assets are useful for dataset/method diagnostics and appendix material. They do not replace the main model-comparison figures (PLL-vs-DMS, perplexity, generation quality/diversity), which are produced by other analysis code.

### B. Paper-figure pipeline (implemented; outputs not in this checkout)

[`report/figures.ipynb`](figures.ipynb) and [`src/protein_design/analysis/figures.py`](../src/protein_design/analysis/figures.py) define a reproducible, cache-backed paper figure workflow. Available figure functions cover:

- grouped PLL-vs-DMS Spearman bars;
- PLL vs supervised scorer Spearman comparison;
- pseudo-perplexity comparison;
- probe Spearman;
- embedding PCA;
- DPO low-data learning curves.

The notebook records generated PDFs such as `pll_spearman.pdf`, `pll_vs_scorer_spearman.pdf`, `pseudo_perplexity.pdf`, `probe_spearman.pdf`, and `embedding_pca.pdf` under `report/figures/` on a cluster. **That directory and those PDFs are absent locally.** The underlying checkpoint/cache paths in analysis configs point to cluster locations, so regenerate/copy the artifacts before final typesetting.

### C. Sampling/evaluation plots available from drivers (regeneration targets)

Sweep configurations and plot scripts support:

- per-chain PLL trajectories/violins and sequence logos;
- pairwise Hamming diversity, edit-distance distribution, mutation-position frequency;
- generated CDR-H3 amino-acid heatmaps;
- generated PLL vs DMS enrichment, fraction above WT, top-k enrichment recovery;
- PLL distributions by temperature and PLL/diversity trade-off;
- per-position entropy and Jensen–Shannon divergence vs temperature;
- novelty against OAS/DMS reference sets;
- beam PLL vs number of mutations and beam-member DMS histograms;
- sampled-sequence PCA overlays and PLL-vs-enrichment overlays.

Relevant drivers include `scripts/analysis/gibbs_diagnostics.py`, `plot_temp_*.py`, `plot_beam_*.py`, `plot_pll_vs_enrichment_overlays.py`, `plot_novelty_analysis.py`, `plot_*pca.py`, and `run_full_analysis_from_config.py`. The sampler output directories exist locally under `outputs/{gibbs,beam_search,baseline_sampler}`, but **no generated sampling CSVs or sweep plots are present in them** in this checkout.

## 4. Evidence/status checklist for report assembly

| Evidence item | Status in this checkout | Report action |
|---|---|---|
| Training and analysis implementations/configs | Present in source tree | Describe the actual run config/checkpoint for each headline result; configs include many possible arms. |
| DMS raw tables and cached ED2/ED5/ED8–11 M22 train/val/test splits | Present under `data/` | Use exact row counts from the tables, label strain/readout, report split seed and filtering. |
| Local plot assets | 34 PNGs + 15 CSVs under `plots/` | Use data-characterization and DPO-pair diagnostics where they answer the narrative; move/copy selected figures into a versioned report figure folder when finalizing. |
| Local training checkpoints and histories | Not surfaced under local `outputs/`; cluster paths appear in configs/model catalog | Pull exact run IDs, checkpoint selection rule, seeds, hardware/time and metrics from cluster run folders/W&B. Do not infer these from preset defaults. |
| Local sampler run tables | Output directories exist but are empty of files | Recover the selected Gibbs/SBS/PSSM/random run CSVs and sidecar metadata before making generation claims. |
| Canonical PLL/perplexity/probe/PCA PDFs | Not in local `report/figures/`; notebook logs show cluster-side outputs | Re-run extraction/preflight on the analysis cache or retrieve the cluster PDFs. |
| `report/report.tex` figure paths | Refers to meeting/embedding plot directories not present locally | Refresh paths and captions after selecting the actual final figures; the report still has TODO counts and older scope. |

## 5. Suggested report structure and figure sequence

1. **Question and system overview** — C05 CDR-H3 design objective, ESM2 backbone, two-stage adaptation/optimization concept. *Figure:* compact pipeline schematic (assemble from actual stages; avoid implying every optional arm was run).
2. **Data and experimental signal** — OAS filtering/dedup/splits; C05 similarity subsets; M22/SI06 and edit-distance panels. *Figures:* one DMS distribution/sequence-position panel from local assets, plus a concise corpus-similarity/length plot if recovered.
3. **Training methods and ablations** — evotuning masking/data choices, TTT, full DPO vs LoRA-DPO, pair construction/low-data, unlikelihood as a negative control. *Figures:* corpus/train curves, chosen-vs-rejected diagnostics if useful, and a method/config table.
4. **Model evaluation** — PLL/pseudo-perplexity and held-out Spearman across panels/strains, compare vanilla → evo → C05/TTT → DPO variants and the supervised scorer reference. *Figures:* canonical Spearman bars and pseudo-perplexity; add PLL-vs-enrichment overlays for representative panels.
5. **Generation experiments** — Gibbs and beam search; compare PSSM/random baselines under matched settings. *Figures:* quality-diversity frontier, enrichment recovery, entropy/diversity, mutation-distance and novelty.
6. **Representation analysis** — include only if it clarifies changes from tuning; use PCA/CKA/Procrustes as supporting evidence rather than a replacement for held-out functional metrics.
7. **Limitations and next experiments** — computational/data limitations, distinction between model likelihood and binding, and whether designs have experimental validation. Make the status of wet-lab validation explicit.

## 6. Reproducibility notes

- Treat `conf/` as the experiment source of truth. For every reported number record the composed config/overrides, model checkpoint, seed, split, and selected checkpoint metric.
- Report training settings from the run's saved config/metadata, not only the current Hydra preset; the presets have evolved and some report drafts describe older values.
- Keep training-set metrics, validation checkpoint-selection metrics, held-out DMS test results, supervised-scorer results, and sampling diagnostics clearly separated.
- Use one dataset/panel naming convention throughout (ED2/ED5/ED8–11; M22/SI06; enrichment vs expression). Include `n` and the actual split used in each figure caption.
- For sampling figures, record initialization (WT/DMS/top-DMS), temperature/trust radius, chain/beam count, steps, mutation cap, seed, scorer/checkpoint, and whether duplicates are retained.
