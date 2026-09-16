# Publication-Quality Scientific Plotting Standard

This file defines the plotting standard for all figures produced for the paper.

The goal is not merely to make plots that are "clean". The goal is to produce figures that look immediately publication-ready: precise, readable at paper scale, visually balanced, and consistent across the entire manuscript.

The visual target is the style of strong ML / scientific-computing conference figures:
- serif / LaTeX-like typography;
- restrained but distinctive colours;
- high legibility;
- generous spacing;
- thin, precise axes and lines;
- no clutter;
- consistent styling across all figures;
- vector PDF output for the paper and high-resolution PNG output for quick inspection.

Do not improvise a new style for every plot. Use the conventions below as the default plotting system unless the data genuinely requires something else.

---

## 1. Non-negotiable requirements

Every final figure must satisfy all of the following:

1. **Readable at final paper size**
   - Axis labels, tick labels, legends, annotations, and subplot labels must remain legible when the figure is inserted into a paper at its intended width.
   - Never rely on zooming to read a figure.

2. **Serif / LaTeX-quality typography**
   - Do not use default Matplotlib sans-serif styling.
   - Prefer LaTeX-rendered text when available.
   - Mathematical notation must be typeset as mathematics, not improvised Unicode text.

3. **Restrained, professional colours**
   - Use muted, publication-safe colours.
   - Avoid neon colours, overly saturated colours, rainbow palettes, and decorative gradients.
   - Distinguish methods using a combination of colour, line style, marker, hatch, or outline where useful.

4. **No overlaps or cramped layout**
   - Labels must never collide.
   - Legends must not obscure data.
   - Multi-panel figures must have enough spacing.
   - Long category names must be wrapped or plotted horizontally.

5. **Consistent visual language**
   - The same model must use the same colour across all figures whenever possible.
   - The same quantity must use the same notation across all figures.
   - Similar plots should share axis ranges when comparison benefits from it.

6. **Two output formats**
   - Save every final figure as:
     - `.pdf` for the manuscript;
     - `.png` for quick browsing.
   - Both must be generated from the same figure object and filename stem.

---

## 2. Base Matplotlib configuration

Use a centralized plotting style. Do not scatter arbitrary `rcParams` across scripts.

Recommended starting point:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",

    # Use LaTeX if the environment supports it robustly.
    # Otherwise keep this False and rely on Computer Modern mathtext.
    "text.usetex": False,

    # Font sizes
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.0,

    # Axes
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,

    # Ticks
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Lines
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fancybox": False,
    "legend.edgecolor": "0.80",

    # Saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
```

If `text.usetex = True` is used, ensure the environment supports LaTeX consistently in CI / reproducible runs. Do not make LaTeX rendering a fragile hidden dependency.

---

## 3. Figure sizes

Figures should be designed for their final manuscript width, not resized arbitrarily afterward.

Use these defaults:

```python
SINGLE_COL_WIDTH = 3.35   # inches
DOUBLE_COL_WIDTH = 7.0    # inches
```

Recommended aspect ratios:

- single-panel single-column: `(3.35, 2.5)` to `(3.35, 2.8)`
- single-panel double-column: `(7.0, 3.0)` to `(7.0, 4.2)`
- 2-panel horizontal: `(7.0, 2.8)` to `(7.0, 3.4)`
- 2x2 grid: `(7.0, 5.5)` to `(7.0, 6.3)`
- 2x3 grid: `(7.0, 5.8)` to `(7.0, 6.6)`

Do not create excessively wide or tall figures unless scientifically necessary.

Use:

```python
fig, ax = plt.subplots(figsize=(3.35, 2.6), constrained_layout=True)
```

or, for more controlled multi-panel layouts:

```python
fig, axes = plt.subplots(
    2, 3,
    figsize=(7.0, 5.8),
    sharex=False,
    sharey=False,
)
fig.subplots_adjust(
    left=0.08,
    right=0.99,
    bottom=0.10,
    top=0.93,
    wspace=0.28,
    hspace=0.34,
)
```

Prefer explicit spacing for complex figures rather than hoping `tight_layout()` fixes everything.

---

## 4. Typography and labels

### 4.1 Axis labels

Use precise, short labels with units where applicable.

Good:

```python
ax.set_xlabel(r"Mutation count")
ax.set_ylabel(r"Preference score")
ax.set_ylabel(r"$\Delta$ log-likelihood")
ax.set_xlabel(r"Round")
```

Bad:

```python
ax.set_xlabel("x")
ax.set_ylabel("score")
```

unless the quantity is genuinely defined that way in the paper.

### 4.2 Mathematical notation

Use raw strings and LaTeX math:

```python
r"$\Delta \log p_\theta(x)$"
r"$\mathrm{AUC}$"
r"$k_{\mathrm{off}}$"
r"$\mu \pm \sigma$"
```

Avoid manually inserting Greek Unicode symbols if the same symbol is used mathematically in the paper.

### 4.3 Titles

Titles should be concise and informative.

Prefer:
- `DPO vs. ESM2`
- `Held-out preference accuracy`
- `Score distribution on test variants`

Avoid:
- full-sentence titles;
- redundant titles that repeat both axes;
- giant title text.

In dense conference figures, panel-specific titles are preferable to one oversized global title.

### 4.4 Tick formatting

- Use sensible precision.
- Do not show unnecessary decimals.
- Do not show scientific notation unless values justify it.
- For percentages, use percent formatting.
- Avoid more than ~6–8 major tick labels per axis unless necessary.

Examples:

```python
from matplotlib.ticker import MaxNLocator, PercentFormatter

ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
```

---

## 5. Colour system

Use a small stable palette throughout the paper.

Recommended primary palette:

```python
COLORS = {
    "green":  "#1B9E77",
    "orange": "#D95F02",
    "blue":   "#5E81AC",
    "purple": "#7570B3",
    "pink":   "#E65A9A",
    "grey":   "#8A8A8A",
    "dark":   "#2B2B2B",
}
```

Alternative slightly softer palette:

```python
COLORS = {
    "teal":   "#1F9D8A",
    "orange": "#E17C3A",
    "blue":   "#5B6FB5",
    "rose":   "#D95F8D",
    "grey":   "#8C8C8C",
}
```

### Rules

- Assign each major model a stable colour and reuse it everywhere.
- Avoid more than 4–5 saturated colours in one panel.
- If >5 methods are shown, add line styles / markers rather than only more colours.
- Baselines should often use neutral grey.
- The focal method may use the most visually salient colour.
- Never use a rainbow colormap for categorical models.

### Colour accessibility

Figures should remain interpretable:
- in grayscale;
- for common forms of colour-vision deficiency.

For line plots, combine:
- colour;
- line style;
- marker style.

For bars:
- colour;
- hatch;
- edge outline if needed.

---

## 6. Line plots

Use line plots for trajectories, learning curves, calibration curves, optimization rounds, or cumulative metrics.

Example style:

```python
ax.plot(
    x, y,
    color=COLORS["green"],
    lw=1.8,
    label="Model A",
)

ax.fill_between(
    x,
    y_low,
    y_high,
    color=COLORS["green"],
    alpha=0.22,
    linewidth=0,
)
```

### Rules

- Main lines: `lw ≈ 1.6–2.2`
- Confidence bands: alpha `0.15–0.25`
- Do not use extremely thick lines.
- If curves overlap strongly, vary both colour and line style.
- Uncertainty bands must not hide other curves.
- Put the strongest or focal method visually on top using `zorder`.

Example:

```python
ax.plot(x, y_main, lw=2.0, zorder=5)
```

### Error representation

Prefer:
- confidence bands for dense x-grids;
- error bars for sparse x values;
- box / violin / dot plots for distributions.

Clearly define in the caption whether bands show:
- standard deviation;
- standard error;
- 95% confidence interval;
- bootstrap interval.

---

## 7. Comparison plots: default choices

Do not default automatically to histograms for every comparison.

Choose plot type based on what is being compared.

### 7.1 Two or more scalar metrics across models

Prefer:
- grouped bar chart;
- point-range plot;
- dot plot with confidence intervals.

A point-range plot is often cleaner than bars:

```python
ax.errorbar(
    x,
    means,
    yerr=errors,
    fmt="o",
    capsize=3,
    elinewidth=1.2,
    markersize=5,
)
```

Use bars mainly when magnitude from a zero or natural baseline is meaningful.

### 7.2 Distribution comparison

Prefer:
- violin + box overlay;
- boxplot + jittered raw points;
- ECDF;
- ridge / KDE only when justified;
- overlaid histograms only when they remain readable.

### 7.3 Paired model comparison

If models are evaluated on the same examples, prefer paired plots over independent histograms.

Useful options:
- scatter plot `Model A score` vs `Model B score` with identity line;
- paired dot plot;
- histogram of score differences;
- ECDF of score differences.

A paired scatter plot often communicates superiority more directly:

```python
ax.scatter(score_a, score_b, s=18, alpha=0.55, edgecolors="none")
lims = [
    min(score_a.min(), score_b.min()),
    max(score_a.max(), score_b.max()),
]
ax.plot(lims, lims, "--", color="0.45", lw=1.0)
ax.set_xlim(lims)
ax.set_ylim(lims)
```

If the question is "how much better is B than A?", plotting
`score_B - score_A`
is usually more informative than two separate histograms.

---

## 8. Histograms

Use histograms only when the distribution shape itself matters.

### 8.1 Bin choice

Never choose arbitrary visually convenient bins.

Use:
- Freedman–Diaconis;
- Scott;
- or a fixed domain-relevant binning rule used consistently.

Example:

```python
bins = "fd"
```

### 8.2 Overlayed histograms

If comparing two models:

```python
ax.hist(
    x1,
    bins=bins,
    density=True,
    alpha=0.45,
    color=COLORS["blue"],
    label="Model A",
)

ax.hist(
    x2,
    bins=bins,
    density=True,
    alpha=0.45,
    color=COLORS["orange"],
    label="Model B",
)
```

But if overlap becomes visually muddy, use:
- step histograms;
- faceted panels;
- ECDFs;
- violin plots.

Example step histograms:

```python
ax.hist(
    x1,
    bins=bins,
    density=True,
    histtype="step",
    lw=1.8,
    color=COLORS["blue"],
    label="Model A",
)
```

### 8.3 Mean / median markers

If meaningful, add subtle vertical markers:

```python
ax.axvline(
    x1.mean(),
    color=COLORS["blue"],
    lw=1.2,
    ls="--",
)
```

Do not overload the plot with mean, median, quartiles, and confidence bands simultaneously unless they are scientifically necessary.

---

## 9. Bar charts

For grouped model comparisons:

```python
width = 0.34
x = np.arange(len(categories))

ax.bar(
    x - width/2,
    model_a,
    width,
    label="Model A",
    color=COLORS["blue"],
    edgecolor="white",
    linewidth=0.6,
)

ax.bar(
    x + width/2,
    model_b,
    width,
    label="Model B",
    color=COLORS["orange"],
    edgecolor="white",
    linewidth=0.6,
)
```

### Rules

- Avoid excessive bar width.
- Keep group spacing visually clear.
- Avoid 3D bars.
- Avoid gradients.
- Add error bars if values are estimated.
- Use horizontal bars when category labels are long.

For long model names:

```python
ax.barh(...)
```

is usually better than heavily rotated x-axis labels.

---

## 10. Scatter plots

Use clean, small markers.

Recommended:

```python
ax.scatter(
    x,
    y,
    s=22,
    alpha=0.65,
    linewidth=0,
    color=COLORS["blue"],
)
```

For dense plots:
- reduce marker size;
- increase transparency;
- consider hexbin only for very large point clouds.

For fitted relationships:
- show the fitted line distinctly;
- show uncertainty in a light band;
- do not make the regression line visually dominate the raw data excessively.

---

## 11. Multi-panel figures

Multi-panel figures should look like one coherent figure.

### Panel labels

Use lowercase bold panel labels:

```python
for label, ax in zip("abcdef", axes.flat):
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )
```

Panel labels should:
- be outside the data region;
- align consistently;
- not collide with titles.

### Shared axes

Use `sharex=True` or `sharey=True` when direct comparison benefits from the same scale.

When panels compare the same metric:
- strongly prefer identical y-axis limits;
- strongly prefer identical tick locations.

Do not use different scales across panels unless necessary and clearly justified.

### Repetition

If all panels share the same legend:
- prefer one figure-level legend.

Example:

```python
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=len(labels),
    frameon=False,
)
```

Do not repeat identical legends in every panel unless the layout genuinely benefits from it.

---

## 12. Legends

Legends must not hide relevant data.

Preferred order:
1. empty area inside the axes;
2. above / below the axes;
3. outside the axes.

Example:

```python
ax.legend(
    loc="best",
    frameon=True,
    borderpad=0.5,
    handlelength=2.2,
)
```

For multi-panel figures:

```python
fig.legend(...)
```

is often cleaner.

Legend labels should be short model names, not explanations.

Bad:
`"Fine-tuned ESM2 model trained using DPO on all train data"`

Better:
`"DPO"`
or
`"ESM2-DPO"`

Explain details in the caption.

---

## 13. Gridlines and backgrounds

Default background:
- white.

Gridlines:
- optional;
- subtle;
- only if they improve value reading.

Recommended:

```python
ax.grid(
    axis="y",
    color="0.90",
    linewidth=0.6,
    zorder=0,
)
```

Avoid:
- dark grids;
- grids on both axes when not needed;
- default heavy seaborn-style backgrounds.

---

## 14. Spines

Use either:
- all four thin spines;
- or only left / bottom spines.

Choose one convention and stay consistent across a figure family.

If removing top/right:

```python
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

Do not combine inconsistent spine conventions within one figure unless deliberate.

---

## 15. Error bars and uncertainty

Error bars must be visually lighter than the central estimate.

Example:

```python
ax.errorbar(
    x,
    y,
    yerr=err,
    fmt="o-",
    lw=1.6,
    elinewidth=1.0,
    capsize=2.5,
    capthick=1.0,
)
```

Avoid giant caps and thick error bars.

The manuscript or caption must define the uncertainty convention.

---

## 16. Significance annotations

Do not fill figures with stars unless required.

Prefer reporting:
- effect sizes;
- confidence intervals;
- exact p-values in caption/table.

If significance brackets are necessary:
- keep them minimal;
- place them with enough vertical padding;
- avoid stacking many brackets over bars.

---

## 17. Annotations

Annotations should explain scientifically meaningful structure, not decorate.

Examples:
- wild-type reference;
- decision threshold;
- baseline;
- known functional region;
- theoretical optimum.

Use subtle reference lines:

```python
ax.axhline(
    baseline,
    color="0.45",
    lw=1.0,
    ls="--",
    zorder=1,
)
```

Text annotations should be placed manually where they do not overlap data.

---

## 18. Scientific consistency

When comparing models:

- use the same axis scale;
- use the same binning for histograms;
- use the same test set;
- use the same metric definition;
- use the same sample filtering;
- use the same uncertainty calculation;
- use matched seeds / folds when comparisons are paired.

The plotting script must not silently introduce an unfair comparison.

---

## 19. Ordering of models and categories

Use an intentional order.

Preferred:
- baseline first;
- intermediate methods second;
- proposed / focal method last;

or:
- natural scientific ordering;
- increasing model complexity;
- decreasing / increasing performance where the ordering itself is informative.

Do not reorder methods differently in every figure.

---

## 20. Recommended model style registry

Use a shared style dictionary.

Example:

```python
MODEL_STYLE = {
    "ESM2": {
        "color": "#8A8A8A",
        "linestyle": "--",
        "marker": "o",
    },
    "EvoTune": {
        "color": "#5B6FB5",
        "linestyle": "-.",
        "marker": "s",
    },
    "DPO": {
        "color": "#1F9D8A",
        "linestyle": "-",
        "marker": "o",
    },
}
```

This must be imported by all plotting scripts rather than redefined ad hoc.

---

## 21. Export requirements

Every final plotting function / script must export both PDF and PNG.

Use:

```python
from pathlib import Path

def save_figure(fig, out_dir, stem, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    return pdf_path, png_path
```

### PDF

- must remain vector whenever possible;
- text should remain vector text where supported;
- do not rasterize the entire figure.

### PNG

- use at least `dpi=300`;
- use `dpi=400–600` for small publication figures with fine text.

Never rely on screenshots as final figures.

---

## 22. File naming

Use descriptive stable names.

Good:

```text
test_score_distribution.pdf
test_score_distribution.png
dpo_vs_evotune_pair_accuracy.pdf
dpo_vs_evotune_pair_accuracy.png
generation_diversity_comparison.pdf
generation_diversity_comparison.png
```

Bad:

```text
plot1.pdf
final2.png
newfigure.pdf
```

---

## 23. Plotting API structure

Plot generation should be reproducible and modular.

Preferred structure:

```python
def plot_model_comparison(df, output_dir):
    fig, ax = plt.subplots(figsize=(3.35, 2.6))

    # plotting code

    style_axes(ax)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "model_comparison",
    )

    plt.close(fig)
```

Use reusable helpers:
- `style_axes`
- `save_figure`
- `add_panel_label`
- `format_metric_axis`
- `get_model_style`

Do not duplicate styling code in every script.

---

## 24. Suggested common helper

```python
def style_axes(ax):
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9.5,
        width=0.8,
        length=3.5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.set_axisbelow(True)
```

---

## 25. Default hierarchy of visual emphasis

When designing any figure:

1. **Data / main scientific result**
2. **Axis labels**
3. **Legend / method identity**
4. **Uncertainty**
5. **Reference lines**
6. **Gridlines / secondary visual structure**

The main result should be visually dominant.

Avoid giving decorative elements more visual weight than the data.

---

## 26. Comparison-plot recommendations for this project

Because many figures will compare multiple models or fine-tuning strategies, use the following default decision tree.

### Case A — one scalar score per model

Use:
- dot plot with confidence intervals;
- or bar chart with error bars.

Prefer dot plots if zero has no special meaning.

### Case B — score distribution across sequences

Use:
- violin + box;
- ECDF;
- histogram only if distribution shape is central.

### Case C — same sequences scored by two models

Use:
- paired scatter plot with identity line;
- difference distribution;
- paired dot / slope plot.

### Case D — performance vs mutation count / sequence property

Use:
- line plot with mean ± CI;
- or binned point plot with error bars.

### Case E — several metrics × several models

Use:
- compact grouped bars;
- or small multiples.

Avoid radar charts.

### Case F — performance across seeds / folds

Show:
- mean and interval;
- plus raw seed/fold points if the number of runs is small.

Do not hide run-to-run variability.

---

## 27. Preferred distribution plot

For two or three models, a clean violin + box + raw-points plot is often superior to a histogram.

Example concept:

```python
# pseudocode structure
# violin = overall distribution
# box = median and IQR
# points = actual observations
```

Keep raw points:
- small;
- semi-transparent;
- horizontally jittered;
- behind the median marker.

---

## 28. Categorical x-axis labels

If labels are long:

Prefer:
- short display names;
- abbreviations defined in caption;
- horizontal bars;
- multi-line labels.

Avoid:
- 60–90 degree rotation;
- overlapping text;
- tiny fonts.

If rotation is unavoidable:
- prefer ~25–35 degrees;
- right-align labels.

---

## 29. Captions and plot self-sufficiency

The plot itself should contain:
- model identities;
- metric names;
- units;
- meaningful reference lines.

The caption should contain:
- dataset / split;
- sample count if important;
- definition of uncertainty;
- statistical test if any;
- notable filtering;
- exact meaning of panels.

Do not force the reader to search the main text to understand basic plot semantics.

---

## 30. Raster-heavy figures

If a figure contains:
- very large scatter clouds;
- images;
- dense heatmaps;

rasterize only those artists if needed while keeping text and axes vector:

```python
ax.scatter(
    x,
    y,
    rasterized=True,
)
```

The output should still be saved as PDF.

---

## 31. Heatmaps

Heatmaps should be used only where a matrix structure is scientifically meaningful.

Rules:
- use perceptually uniform sequential or diverging colormaps;
- use a diverging map only when there is a meaningful central value;
- label the colorbar;
- avoid rainbow maps;
- do not annotate every cell if text becomes cramped.

Recommended:
- `viridis`
- `cividis`
- `magma`
- `coolwarm` only for genuinely signed / centered quantities.

---

## 32. Tables vs plots

Do not plot something that is better shown as a table.

Use a table when:
- there are only a few exact values;
- exact numeric comparison matters more than shape;
- the figure would reduce to labels + numbers.

Use a figure when:
- trend;
- distribution;
- uncertainty;
- interaction;
- ranking pattern;
- paired behavior;
- trajectory

is visually informative.

---

## 33. Visual QA checklist

Before accepting any figure, inspect both the PNG and the PDF.

Mandatory checks:

- [ ] all labels readable at final paper size;
- [ ] no overlapping labels;
- [ ] no clipped labels;
- [ ] no clipped legend;
- [ ] no title collision;
- [ ] no subplot-label collision;
- [ ] no duplicated legend when unnecessary;
- [ ] no excessive empty space;
- [ ] no cramped panels;
- [ ] line widths are visually balanced;
- [ ] confidence bands do not hide important curves;
- [ ] colours are restrained;
- [ ] colours are distinguishable;
- [ ] model colours are consistent with other figures;
- [ ] axes use scientifically sensible limits;
- [ ] comparisons use matched scales where appropriate;
- [ ] units are present where needed;
- [ ] tick precision is sensible;
- [ ] PDF is vector where possible;
- [ ] PNG is at least 300 dpi;
- [ ] filename is descriptive;
- [ ] final output looks good both on screen and when printed / viewed in grayscale.

---

## 34. Automatic sanity checks

Where practical, plotting scripts should assert basic validity:

```python
assert np.isfinite(values).all()
assert len(labels) == len(values)
```

For comparisons:

```python
assert set(df_a["sequence_id"]) == set(df_b["sequence_id"])
```

when paired support is required.

Never silently drop large fractions of the data because plotting code encountered NaNs.

---

## 35. Avoid these common failure modes

Do not produce:

- default Matplotlib blue/orange plots without deliberate styling;
- Comic Sans / generic sans-serif paper figures;
- tiny 7 pt labels;
- oversized titles;
- huge legends;
- legends on top of data;
- 90° rotated category labels;
- saturated rainbow palettes;
- 3D bars;
- pie charts for serious model comparison;
- radar charts;
- over-smoothed KDE curves with no raw distribution context;
- unnecessary gridlines;
- large marker outlines;
- excessive use of bold;
- arbitrary y-axis truncation that exaggerates differences;
- non-matched histogram bins across compared methods;
- different model colours in different figures;
- bitmap-only final export;
- `plt.show()` as the only output;
- figures that depend on manual post-processing to become publication-ready.

---

## 36. Standard export example

Every script should end with something equivalent to:

```python
fig, ax = plt.subplots(
    figsize=(3.35, 2.6),
    constrained_layout=True,
)

# ... plot ...

ax.set_xlabel(r"Mutation count")
ax.set_ylabel(r"Preference score")

save_figure(
    fig,
    output_dir="figures",
    stem="preference_score_by_mutation_count",
    dpi=400,
)

plt.close(fig)
```

---

## 37. Final principle

The standard is:

> every plot should look intentionally designed, not merely generated.

The plotting code should optimize simultaneously for:
- scientific correctness;
- legibility;
- consistency;
- visual hierarchy;
- reproducibility;
- publication quality.

If a figure looks crowded, do not shrink the fonts. Redesign the layout.

If a figure needs too many visual encodings, split it into panels.

If a legend is too large, simplify labels.

If a plot is hard to interpret without explanation, choose a better plot type.

The figure should communicate the scientific comparison immediately and precisely.
