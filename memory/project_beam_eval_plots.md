---
name: beam_eval_plots_and_temperature_sweep
description: Implemented Tasks 1-3: new SBS evaluation plots, temperature sweep pipeline, and top_dms start mode
type: project
---

Added Task 1 (beam eval plots), Task 2 (temperature sweep), Task 3 (top_dms mode).

**Why:** User requested full SBS evaluation pipeline for antibody CDR-H3 design.

**How to apply:** When working on analysis pipeline, be aware of new scripts and config keys added.

## New scripts (scripts/analysis/)
- `plot_beam_pll_vs_dms.py` — PLL histogram: SBS vs DMS (config: `beam_pll_vs_dms_histogram`)
- `plot_beam_diversity.py` — 3-panel diversity (Hamming, entropy, n_mut) (config: `beam_diversity_diagnostics`)
- `plot_beam_pll_vs_nmut.py` — scatter PLL vs n_mutations + Spearman (config: `beam_pll_vs_nmut`)
- `plot_beam_aa_heatmap.py` — 20×24 AA frequency heatmaps SBS vs DMS (config: `beam_aa_heatmap`)
- `run_temperature_sweep.py` — temperature sweep orchestrator
- `plot_temp_pll_vs_diversity.py` — cross-temp summary: PLL vs diversity tradeoff
- `plot_temp_pll_distributions.py` — cross-temp summary: overlaid PLL histograms

## New configs
- `conf/analysis/temperature_sweep.yaml` — temperature sweep config
- `conf/analysis/full_analysis.yaml` — 5 new flat-bool plot keys added under `plots`

## New bash script
- `bash_scripts/utils/run_temperature_sweep.sh`

## Key code decisions
- Beam eval plots use flat booleans in YAML (not enabled/datasets dict); `_validate_plots` 
  skips them, `_beam_plot_enabled()` reads them.
- DMS-needing plots (beam_pll_vs_dms_histogram, beam_aa_heatmap) run inside the 
  `for ds in required_datasets:` loop; non-DMS plots run after it.
- `top_dms` in SBS runs one full beam per top-k seed; chain_id = seed_idx * beam_size + member_idx.
- `gibbs_diagnostics.py` already does exact-PLL trajectory (`beam_pll_trajectory` flag routes to it).
- `plot_pairwise_hamming` and `plot_edit_distance` already exist in gibbs_diagnostics.py; 
  plot_beam_diversity.py re-implements them locally for a standalone 3-panel figure.
