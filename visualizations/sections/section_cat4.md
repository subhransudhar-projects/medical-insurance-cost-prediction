## Category 4: Correlation & Multivariate Visualizations

These figures move beyond one-variable-at-a-time views to expose how the
predictors relate to each other and, above all, how **smoking status
interacts** with the other features to drive insurance charges. All numeric
axes use exact schema columns (`age`, `bmi`, `children`, `charges`), smoker is
colored consistently (`no` = blue, `yes` = vermillion), and every projection
that mixes units is min-max normalized first.

### Correlation heatmap
![Pearson correlation heatmap of numeric features with binary smoker](figures/cat4_corr_heatmap.png)

**Interpretation:** Binary-encoded smoker dominates every other signal with a
Pearson r of 0.79 against charges, dwarfing age (0.30) and bmi (0.20), while
the predictors themselves are essentially uncorrelated (all |r| <= 0.11), so
there is no meaningful collinearity to worry about. The single strong red cell
in the smoker row is the headline of the entire analysis.

**Styling choice:** The diverging `RdBu_r` colormap fixed at vmin=-1, vmax=1,
center=0 with a square grid makes the sign and magnitude of each correlation
readable at a glance, and every cell is annotated with its value.

### Pairplot colored by smoker
![Scatter-matrix of numeric features colored by smoker](figures/cat4_pairplot_smoker.png)

**Interpretation:** Coloring the scatter-matrix by smoker splits the
`charges` panels into two cleanly separated bands, and the bimodal charges KDE
on the diagonal shows smokers forming their own high-cost population; no such
separation appears in age, bmi, or children on their own.

**Styling choice:** KDE diagonals with transparent scatter (alpha 0.45) keep
the dense 1,300-point matrix legible while still showing where the two smoker
groups overlap versus diverge.

### Pairplot colored by region
![Scatter-matrix of numeric features colored by region](figures/cat4_pairplot_region.png)

**Interpretation:** Unlike smoker, region produces heavily overlapping,
intermingled clouds in every panel, confirming that geography contributes
little structure to the numeric relationships and is a weak predictor of
charges relative to smoking.

**Styling choice:** The fixed four-color region palette (blue/green/vermillion/
orange, all colorblind-safe and never red-green paired) keeps region identity
consistent with the rest of the suite.

### Parallel coordinates
![Parallel coordinates of normalized numerics by smoker](figures/cat4_parallel_coords.png)

**Interpretation:** After normalizing each axis to [0, 1], the smoker (orange)
and non-smoker (blue) lines are thoroughly mixed on age, bmi, and children but
fan apart on the final `charges` axis, where orange lines ride high and blue
lines stay low, visually restating the r = 0.79 relationship.

**Styling choice:** Low alpha (0.28) on thin lines controls overplotting across
all rows, and a rebuilt legend maps colors explicitly to smoker levels.

### RadViz projection
![RadViz projection of normalized numerics by smoker](figures/cat4_radviz.png)

**Interpretation:** In the RadViz spring layout the two smoker classes drift
toward different regions of the circle, pulled by the `charges` anchor, giving
a compact 2-D confirmation that smoking status is the dimension that best
separates the observations.

**Styling choice:** Inputs are min-max normalized before projection (required
for RadViz to be meaningful) and drawn with alpha 0.45 so overlapping points
remain distinguishable.

### Andrews curves
![Andrews curves of normalized numerics by smoker](figures/cat4_andrews.png)

**Interpretation:** The Fourier-encoded curves for smokers and non-smokers
trace visibly different envelopes, indicating the two groups occupy different
multivariate signatures even though individual features overlap.

**Styling choice:** A balanced ~300-row sample (150 per smoker class) with
alpha 0.35 prevents the curve bundle from turning into an unreadable smear.

### Faceted regression: age vs charges by smoker
![Faceted regression of charges on age by smoker](figures/cat4_facet_age_charges_by_smoker.png)

**Interpretation:** Both smokers and non-smokers show charges rising with age,
but the smoker facet sits on a much higher intercept and shows wider spread,
so age adds cost within each group while smoking sets the baseline level.

**Styling choice:** `sns.lmplot` with per-facet 95% CI bands and a neutral black
fit line keeps the regression readable against the smoker-colored points.

### Faceted regression: bmi vs charges by smoker
![Faceted regression of charges on bmi by smoker](figures/cat4_facet_bmi_charges_by_smoker.png)

**Interpretation:** This is the central multivariate story: for non-smokers the
bmi slope is nearly flat (about +$83 per bmi unit), while for smokers it is
steep (about +$1,473 per bmi unit), a textbook smoker x bmi interaction where
obesity is costly only in combination with smoking.

**Styling choice:** Each facet carries an annotated slope callout boxed in its
smoker color so the flat-versus-steep contrast is stated numerically, not just
implied.

### Faceted regression: age vs charges by region
![Faceted regression of charges on age by region](figures/cat4_facet_age_charges_by_region.png)

**Interpretation:** The age-charges relationship is nearly identical across all
four regions, with the same positive slope and the same smoker-driven upper
band appearing everywhere, reinforcing that region is not a meaningful
moderator.

**Styling choice:** A 2x2 `col_wrap` layout with shared axes lets the four
regional fits be compared directly at the same scale.

### Interaction plot: smoker x region
![Interaction plot of mean charges by region and smoker](figures/cat4_interaction_smoker_region.png)

**Interpretation:** The smoker line sits far above the non-smoker line in every
region and the two lines stay roughly parallel, so there is a large smoker main
effect with only modest region-to-region variation in the penalty (a weak
interaction at most).

**Styling choice:** Direct end-of-line labels plus a highlighted takeaway box
replace a cramped legend and state the parallel-lines conclusion explicitly.

### Interaction plot: smoker x sex
![Interaction plot of mean charges by sex and smoker](figures/cat4_interaction_smoker_sex.png)

**Interpretation:** The smoker and non-smoker lines run almost perfectly
parallel across male and female, meaning the smoking penalty is essentially the
same for both sexes and there is no meaningful smoker x sex interaction.

**Styling choice:** Distinct markers (circle vs square) plus consistent smoker
colors make the two nearly-parallel lines easy to track without relying on
color alone.
