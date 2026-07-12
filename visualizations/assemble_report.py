"""Stitch the per-category section markdowns into one publication report."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
SECTIONS = BASE / "sections"
OUT = BASE / "visualization_report.md"

INTRO = """# Medical Insurance Cost Dataset — Visualization Suite

**Stage 3 of 3** · Data Cleaning ✔  ·  Exploratory Data Analysis ✔  ·  **Visualization (this document)**

## Introduction

This document is a comprehensive, publication-ready visualization suite that tells the
complete story of **what drives medical insurance `charges`**. It uses the exact dataset
schema throughout — `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` — and
never renames a column.

**Approach.** Every chart is generated from a single shared configuration module
(`viz_config.py`) so the entire suite is visually coherent:

- **One colorblind-safe palette** (Wong 2011) used everywhere, with *fixed semantic colors*
  — `smoker` is always vermillion (yes) / blue (no), so a reader learns the encoding once.
  No red-green combinations appear anywhere.
- **Consistent derived categories:** `age_group` (18-30, 31-40, 41-50, 51-64) and
  `bmi_category` (Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30).
- **Publication styling:** seaborn whitegrid, titles ≥14pt, axis labels ≥12pt with units
  (`charges (USD)`, `bmi (kg/m²)`, `age (years)`), value labels on bars, alpha on dense
  scatter, and an annotated insight on the charts that carry the headline findings.
- **Libraries:** matplotlib, seaborn, and plotly (interactive charts exported to both `.html`
  and a static `.png`), supported by pandas, numpy, scipy.stats, statsmodels, and squarify.

**How to read this suite.** Categories 1–6 build the evidence from the ground up
(distributions → relationships → group comparisons → multivariate structure → advanced/3D →
creative). Category 7 curates the most decisive views into a one-page executive dashboard and
walks through the narrative. All image files live in `figures/`; interactive versions are
linked inline where they exist.

**The one-sentence takeaway:** *`smoker` is overwhelmingly the dominant driver of `charges`,
it compounds sharply with high `bmi`, `age` adds a steady secondary effect, and `region`,
`sex`, and `children` are not reliable drivers.*

---
"""

CAT7 = """
## Category 7: Visualization Summary & Executive Dashboard

### One-page executive dashboard

![Executive dashboard](figures/cat7_executive_dashboard.png)

A single page a business leader can absorb in under a minute: four headline KPIs across the
top, then the four charts that carry the decision — (a) the raw smoker cost gap, (b) the
`smoker`×`bmi` interaction, (c) the obese-smoker segment premium, and (d) the ranked strength
of every feature. The footer states the pricing recommendation directly.
**Styling choice:** KPI "cards" use a light fill with a colored accent bar matching each
metric's semantic color, so the numbers read as a briefing, not a chart.

### Curated grid — the nine most important views

![Curated grid](figures/cat7_curated_grid.png)

The nine highest-value figures from Categories 1–6 tiled into one board, ordered so the eye
travels from the headline gap → the interaction → the segments → the supporting structure.
**Styling choice:** re-tiling the *actual* saved figures (rather than re-plotting) guarantees
the curated board is identical to the standalone charts it references.

### The story — walking through the evidence

A logical narrative a presenter can deliver in order, building the case with the exact columns:

1. **Start with the target.** `charges` is strongly right-skewed (Category 1: histogram/KDE,
   ECDF, Q-Q) — most beneficiaries are inexpensive, but a long tail of high-cost cases pulls
   the mean above the median. This shape is the whole reason cost is interesting: *something*
   creates that tail.

2. **Name the driver.** Split `charges` by `smoker` (Category 2: `cat2_smoker_box_points`) and
   the tail resolves almost entirely into smokers — mean **$32,050 vs $8,441**, a ~3.8× gap
   with barely any distributional overlap. Of the numeric/binary features, `smoker` carries by
   far the largest correlation with `charges` (**r ≈ 0.79**; Category 4 heatmap).

3. **Explain the tail's shape.** The driver is not additive — it *interacts* with `bmi`
   (Category 4 `cat4_facet_bmi_charges_by_smoker`; Category 3 `cat3_bar_charges_by_bmicat_smoker`).
   Among non-smokers, `bmi` barely moves `charges` (~+$83 per BMI unit); among smokers the slope
   is ~**+$1,473 per BMI unit**, and **obese smokers average ~$41,558** — the single most
   expensive segment in the data.

4. **Add the steady secondary effect.** `age` raises `charges` for everyone (Category 5
   `cat5_age_progression`; r ≈ 0.30), in roughly parallel bands for smokers and non-smokers —
   so the smoker gap *stays wide across the entire age span* rather than closing.

5. **Rule out the rest.** `region`, `sex`, and `children` show weak, statistically unreliable
   relationships with `charges` (Category 3 region/sex bars; Category 4 interaction plots show
   `smoker`×`sex` lines nearly parallel = no interaction). They should not drive pricing.

6. **Quantify the buildup.** The waterfall (Category 6 `cat6_waterfall_charges_decomposition`)
   decomposes the journey from the non-smoker baseline to the obese older smoker, and the
   Sankey shows the High-cost tier is fed almost entirely by smokers — closing the case.

### Recommendations — what to put in the final report/presentation

**Tier 1 — lead with these (the decision-grade charts):**
- `cat7_executive_dashboard.png` — the single most important asset; put it on slide 1.
- `cat2_smoker_box_points.png` — the headline smoker gap, unambiguous.
- `cat3_bar_charges_by_bmicat_smoker.png` — the `smoker`×`bmi` interaction in one glance.
- `cat6_waterfall_charges_decomposition.png` — how the premium builds, for a cost narrative.

**Tier 2 — strong support:**
- `cat4_corr_heatmap.png` and `cat4_facet_bmi_charges_by_smoker.png` (quantify the interaction).
- `cat3_heatmap_age_bmi.png` (the `age`×`bmi` cost surface as a table).
- `cat5_age_progression.png` (the persistent gap across `age`).
- `cat6_ridge_charges_by_bmicat_smoker.png` (distribution shift — visually striking).

**Tier 3 — appendix / interactive exploration:**
- The Plotly interactive files: `cat5_interactive_dashboard.html` (dropdown filters),
  `cat5_3d_scatter_smoker.html`, `cat3_treemap_region_smoker.html`,
  `cat6_sankey_region_smoker_tier.html`.
- Distribution/diagnostic plots (Q-Q, ECDF, residuals) belong in a technical appendix.

**Charts to de-emphasize:** the `region` choropleth/bubble maps are honest *schematics* (the
dataset has only four census-style regions with no true coordinates) and `region` is not a
reliable driver — keep them illustrative only, clearly labeled as schematic.

### Final checklist

| # | Category | Delivered |
|---|----------|-----------|
| 1 | Distribution (univariate) | ✔ hist+KDE, box+violin, density, Q-Q, ECDF, bar, pie, waffle, charges bin-sensitivity, charges CDF (10 figures) |
| 2 | Relationship (bivariate vs target) | ✔ scatter+reg, LOESS, hexbin, 3 jointplots, residuals, box, violin, strip/swarm, bar+CI, faceted catplot, smoker box+points (13) |
| 3 | Comparison (group) | ✔ grouped region×smoker, grouped sex×smoker, stacked smoker-by-region, age×bmi heatmap, treemap (+HTML), agegroup box/bar, bmicat box/bar (9) |
| 4 | Correlation & multivariate | ✔ corr heatmap, 2 pairplots, parallel coords, RadViz, Andrews, 3 faceted scatters, 2 interaction plots (11) |
| 5 | Advanced & specialized | ✔ 2 3D scatters, 3D surface, choropleth, bubble map, age progression, multi-panel, interactive dashboard (8; 5 with HTML) |
| 6 | Creative & original | ✔ ridge/joyplot, waterfall, slope, diverging deviation, Sankey (5) |
| 7 | Summary & executive | ✔ executive dashboard, curated 3×3 grid, narrative, recommendations, this checklist |

**Totals:** 58 static figures (PNG + PDF each) and 7 interactive HTML charts, all inside the
`Medical Insurance Cost Dataset\\visualizations\\` folder. Styling requirements
(colorblind-safe palette, ≥14/12/10pt text, units, value labels, annotations, no red-green,
alpha on dense plots) applied throughout.

**Ready for modeling?** Yes — the visual case is complete and consistent with the Stage-2 EDA.
Recommended next step: a tree-based model or a GLM with explicit `smoker`×`bmi` and
`smoker`×`age` interaction terms, fit on a log-transformed `charges` target.
"""

parts = [INTRO]
for i in range(1, 7):
    txt = (SECTIONS / f"section_cat{i}.md").read_text(encoding="utf-8").strip()
    parts.append(txt)
    parts.append("\n---\n")
parts.append(CAT7)

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {OUT} ({len(OUT.read_text(encoding='utf-8').splitlines())} lines)")
