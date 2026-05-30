
---
## slide 1 — project goal
Generative model for producing antibody sequences with high affinity to a target antigen.

- Concrete target: C05, a broadly neutralizing anti-influenza antibody. We want to generate variants of the C05 CDR-H3 region with high predicted binding (log-enrichment) to the antigen.
- Starting point: a pretrained protein language model (ESM2). We do **not** train from scratch.
- Two-stage adaptation strategy:
  1. **Evotuning** — domain-adapt ESM2 to the antibody distribution via masked language modeling.
  2. **DPO** — align the resulting model toward sequences with high experimental binding using paired preference data.
- End goal: sample new CDR-H3 variants from the adapted model and score them; expect them to be enriched for high-affinity sequences vs. random sampling.
- Working with ESM2-t12 (35M) for fast iteration and ESM2-t33 (650M) for the final models.

---
### slide 2 — DMS datasets (what we evaluate against)
Deep mutational scanning datasets give us per-mutant binding scores; we use them as ground truth for ranking.

- Three DMS panels used for evaluation (**verify exact wording with supervisor**):
  - **M22**: binding-enrichment panel against one HA antigen (the largest panel, used for most reporting). Raw counts: ED2 = 97,977; ED5 = 428,821; ED811 = 518,197 variants.
  - **SI06**: binding-enrichment panel against A/Solomon Islands/3/2006 (H1N1) HA. Raw counts: ED2 = 88,929; ED5 = 365,588.
  - **exp**: experimental / expression panel (ED2 only) — 93,772 variants. Orthogonal signal / control.
- "ED2 / ED5 / ED811" = number of mutations from the C05 wild-type CDR-H3 in the library (edit distance 2, 5, and 8–11). Higher edit distance → bigger combinatorial library, harder to predict.
- Each (sequence, score) pair lets us correlate model log-likelihood / pseudo-log-likelihood against measured binding.
- Headline evaluation metric: **Spearman correlation** between model score and DMS score.
- We additionally bucket by **flank position** (`left{1,3,5}` / `right{1,3,5}` of the mutation site) to check whether the model picks up signal locally or only globally. (Optional — drop if the slide is too dense.)
- Why this matters: DMS is the only direct, quantitative signal we have for "is this a good antibody?" — without it we'd only be measuring perplexity, which doesn't necessarily correlate with binding.

---
## slide 3 — ESM2 backbone
What it is and why we picked it.

- ESM2 = encoder-only Transformer protein language model from Meta/FAIR (Lin et al., *Science* 2023, https://www.science.org/doi/10.1126/science.ade2574).
- Trained on ~65M unique UniRef50 sequences with a standard **masked language modeling** objective (predict masked amino acids from context).
- Architecture: BERT-style stack of pre-LN Transformer blocks, learned token embeddings over the 20 amino acids + special tokens, RoPE-style positional encoding.
- Two sizes we use:
  - **t12 / 35M params** — cheap, used for sweeps and ablations.
  - **t33 / 650M params** — used for the final/best runs.
- Why ESM2 (vs. ESM-IF, AbLang, IgBert, AntiBERTy): general-purpose, well-benchmarked, scales cleanly, and unbiased toward antibodies (good base for evotuning experiments).
- Used in two modes:
  - **Likelihood scorer**: pseudo-log-likelihood (PLL) of a sequence as a fitness proxy.
  - **Conditional generator**: sample CDR-H3 by iterative masking (Gibbs) or stochastic beam search.

---
## slide 4 — Evotuning: domain adaptation on antibodies
Continue MLM pretraining on antibody-only sequences so the model "speaks antibody".

- Objective: same **masked language modeling** loss as ESM2 pretraining (mask 15% of tokens, cross-entropy on masked positions).
- Why bother: ESM2 sees mostly non-antibody proteins. Antibodies have very specific statistics (V/D/J segments, conserved framework, hypervariable CDRs) that the base model only partially captures.
- Training recipe (35M, current best):
  - lr = 2e-5, 3 epochs, batch handled by HF Trainer, fp16/bf16
  - Dataset: deduplicated, filtered OAS (see next slide).
- Monitored: train/val MLM loss, val perplexity, CDR-H3 pseudo-perplexity, plus periodic Spearman on the DMS panels to make sure adaptation isn't hurting downstream usefulness.
- Output: a checkpoint we use as the **seed** for all downstream stages.

---
## slide 5 — OAS: the antibody corpus
Observed Antibody Space (Olsen et al.) (https://onlinelibrary.wiley.com/doi/10.1002/pro.4205) — public repository of B-cell receptor repertoire sequencing.

- Raw OAS is huge and noisy (many studies, species, chains, isotypes, redundant clonotypes).
- **Filter to a relevant slice** before training:
  - species = `human`
  - chain = `heavy`
  - vaccine = `flu` (matches our target antigen biology — flu hemagglutinin)
- After filter: **~192.3 M sequences** stored as a FASTA at `$PROJECT_DIR/datasets/oas_filtered.fasta` (+ matching `oas_filtered.csv.gz` with per-sequence metadata: subject, study, V/J calls, isotype, CDR-H3, etc.).
- Useful to mention: CDR-H3 length distribution of the filtered corpus (we have a plot: /cluster/home/mdenegri/protein-design/report/plots/cdrh3_length_distribution.pdf).

---
### slide 6 — OAS deduplication, splits, sanity checks
Why dedup matters, what we did, what we found.

- B-cell repertoires are **clonally redundant**: thousands of near-identical sequences from clonal expansion. Without dedup we overfit to a few clones and inflate val metrics.
- **MMseqs2 easy-linclust** at 99% sequence identity, ≥ 90% coverage → collapses clonal families to one representative.
- Sequence count: filtered → dedup goes from **~192.3 M → ~143.6 M** (≈ 25% redundancy removed at 99% identity).
- Split: train / val / test = **129.3 M / 7.18 M / 7.18 M** (≈ 90 / 5 / 5), saved as `oas_dedup_rep_seq_{split}.fasta`.
- Sanity check: searched for C05 itself in the corpus → **not present** (expected — C05 is a known, characterized antibody not in OAS).
- Searched for C05-similar sequences (full VH and CDR-H3 only) using MMseqs2 + BLOSUM62 → found very few close hits, especially in the CDR-H3 (C05's H3 is an outlier in OAS). This motivated the C05-specific sub-corpora (see C05 datasets section).

---
## slide 7 — DPO: what and why
Direct Preference Optimization (Rafailov et al., NeurIPS 2023). The "RLHF without RL" recipe.

- After evotuning, the model is good at modeling antibody-like sequences, but it has no notion of *binding*. We need a signal that pulls it toward high-affinity variants.
- Standard option would be RLHF: train a reward model from preference data, then PPO. Expensive, unstable.
- DPO bypasses the reward model: given pairs `(seq_winner, seq_loser)`, optimize the policy directly so that `log π(winner) − log π(loser)` increases relative to a frozen reference.
- Fits our setting cleanly: from DMS we can construct an arbitrary number of (winner, loser) pairs based on log-enrichment differences.
- Same data, no RL loop, no reward model — just a contrastive log-likelihood loss against a frozen reference copy of the evotuned model.

---
### slide 8 — DPO loss
Formula and intuition.

- Loss (per pair):
  `L_DPO = − log σ( β · [ log π_θ(y_w | x) − log π_ref(y_w | x) − ( log π_θ(y_l | x) − log π_ref(y_l | x) ) ] )`
  where `y_w` is the preferred sequence, `y_l` the dispreferred, `π_ref` is the frozen evotuned model, and `β` controls how far the policy can drift from the reference.
- Interpretation: it's a **logistic loss on the implicit reward** `r(y) = β · log(π_θ(y) / π_ref(y))`.
- In our setup `log π(y)` is the per-token sum over the CDR-H3 region only (we don't push gradients through the framework, which is fixed).
- Hyperparams used: β ≈ 0.04, lr 1e-5, batch 128, ~15 epochs.
- Variant we tried: **weighted DPO** — weight each pair by the magnitude of the log-enrichment gap so larger preference gaps contribute more.

---
### slide 9 — Pair construction strategies
"How do you turn a DMS table into preference pairs?" — several options, each with tradeoffs.

- Source of truth: per-mutant log-enrichment scores from the DMS panels.
- Strategies explored:
  1. **All pairs**: every (i, j) with `score_i > score_j`. Explodes O(N²), heavy class imbalance toward small gaps.
  2. **Top-vs-bottom**: pair top-k mutants against bottom-k. Strong signal but few pairs.
  3. **Margin-thresholded pairs**: keep only pairs with `|score_i − score_j| > τ`. Tunable, our current default.
  4. **Weighted pairs**: take all valid pairs but weight loss by gap magnitude (the "weighted DPO" variant).
- Also explored **unlikelihood training** as an alternative (push down likelihood of losers directly). Outcome: it pushes the model off-distribution and tanked perplexity → abandoned, kept here as a negative result.
- Final strategy: margin-thresholded + weighted DPO, with the evotuned 35M as reference.

---
### slide 10 — LoRA adapters for DPO
Avoid catastrophic forgetting of the evotuned base.

- Problem: full-parameter DPO can drift the model far from the reference distribution, hurting general antibody likelihood (and Spearman on held-out panels).
- Solution: **LoRA** (Hu et al., 2021). Freeze the evotuned weights, inject low-rank adapters (`W + αBA`, `r` small) into the attention and/or MLP projections.
- Benefits in our setting:
  - Only the adapter is trained → ~1–2% of parameters → smaller effective drift.
  - Reference model is literally the base + zero-adapter, so KL stays bounded.
  - Cheap to train and to swap in/out → can A/B different DPO recipes from the same evotuned seed.
- We compare full-DPO vs. LoRA-DPO on the same pairs / same `β` to quantify the forgetting tradeoff.

---
### slide 11 — Evaluation metrics
What we report and why each one matters.

- **Validation MLM perplexity** (on held-out OAS): basic sanity — has the model stayed a good antibody LM?
- **CDR-H3 pseudo-perplexity**: same idea but restricted to the hypervariable region; this is where evotuning + DPO actually do their work.
- **Spearman correlation** between model PLL and DMS log-enrichment, on M22 / SI06 / exp:
  - Overall (all positions).
  - Flank-restricted: `left{1,3,5}` and `right{1,3,5}` around the mutation site — tells us *where* the model picks up signal.
- **Headline plot**: per-checkpoint bar chart of overall Spearman across datasets, plus the flank breakdown subplot.
- Loss curves (train/val) as a sanity layer for every run.
- Why Spearman (not Pearson): we care about ranking sequences, not predicting absolute log-enrichment.

---
## slide 12 — Results
Putting it together: evotuning → C05 finetune → DPO.

- **Evotuning alone** (vs. base ESM2-35M):
  - CDR-H3 pseudo-perplexity ↓ substantially.
  - Spearman on DMS ↑ (number from the comparison plot).
- **C05-specific finetune** on top of evotuned seed:
  - Sweep over 5 corpora (`c05_vh_pid{30,60}`, `c05_cdrh3_posid30`, `c05_cdrh3_blosum{25,30}`).
  - Observation: full-VH corpora (`vh_pid*`) give the best Spearman; CDR-H3-only corpora are too sparse (BLOSUM30 → 1k seqs; MMseqs CDR-H3 → 809 seqs, dropped).
  - **C05's CDR-H3 is an outlier in OAS** → CDR-H3-targeted retrieval is fundamentally limited.
- **DPO on top of evotuned + C05-finetuned**:
  - LoRA-DPO with weighted pairs > full-param DPO on Spearman, and preserves OAS perplexity.
  - Unlikelihood baseline destroyed perplexity, didn't help Spearman → reported as negative result.
- **Sampling**: Gibbs sampling and stochastic beam search from the DPO model produce CDR-H3 variants that score higher in PLL than random samples / base ESM2 samples (show example sampled sequences if space allows).
- Open follow-ups: 650M version of the full pipeline; combining TTT with DPO; experimental validation of top sampled designs.
