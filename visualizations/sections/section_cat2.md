## Category 2: Relationship Visualizations (Bivariate — Feature vs. Target)

This category examines how each individual feature relates to the target, `charges (USD)`.
Numeric features (`age`, `bmi`, `children`) are explored with scatter, smoothing, density,
joint distributions and residual diagnostics; categorical features (`sex`, `smoker`, `region`)
with boxes, violins, point clouds, confidence-interval bars and faceting. Semantic colors are
fixed throughout: smokers are vermillion, non-smokers blue, all from the colorblind-safe Wong
palette (no red-green pairings).

### Linear fit with 95% confidence interval

![Scatter with linear regression line and CI for age, bmi and children vs charges](figures/cat2_scatter_regline.png)

**Interpretation:** `age` shows the clearest positive linear slope (r ≈ 0.30), `bmi` a weaker
positive tilt (r ≈ 0.20), and `children` an essentially flat line (r ≈ 0.07); the wide vertical
scatter around every line signals that no single numeric feature explains charges on its own.
**Styling choice:** Points are drawn at low alpha with a light sky-blue so the vermillion
regression line and its shaded CI band stay legible over ~1,300 overlapping observations.

### Nonparametric LOWESS trend

![Scatter with LOWESS smooth and Pearson r annotated for each numeric feature](figures/cat2_scatter_loess.png)

**Interpretation:** The LOWESS curves confirm the linear reads — `age` rises steadily while
`bmi` and `children` are nearly flat at the population level — and the two visible parallel bands
in the `age` panel foreshadow the smoker split explored later. **Styling choice:** Each panel is
annotated with its Pearson r and p-value in a boxed callout so the strength and significance of
the relationship are readable without a separate table.

### Hexbin density

![Hexbin density plots of charges vs age, bmi and children](figures/cat2_hexbin.png)

**Interpretation:** Binning reveals where the mass actually sits — a dense low-charge ridge
(< $15k) across all ages and bmis — which the raw scatter obscures through overplotting.
**Styling choice:** A perceptually-uniform, colorblind-safe viridis sequential colormap encodes
`count`, with an explicit colorbar so darker cells unambiguously mean more observations.

### Joint distribution — charges vs age

![Jointplot of charges vs age with marginal histograms](figures/cat2_jointplot_age.png)

**Interpretation:** The joint scatter shows the same two rising bands, and the charges marginal
is strongly right-skewed with a long high-cost tail. **Styling choice:** Marginal histograms on
the top and right axes summarize each variable's univariate distribution alongside their joint
relationship in a single frame.

### Joint distribution — charges vs bmi (colored by smoker)

![Jointplot of charges vs bmi colored by smoker status](figures/cat2_jointplot_bmi.png)

**Interpretation:** This is the most revealing panel: the bmi–charges effect is *conditional* —
the vermillion smoker cluster climbs sharply as bmi crosses ~30, while blue non-smokers stay
flat and low regardless of bmi, explaining why the overall r is only 0.20. **Styling choice:**
Coloring by `smoker` with the fixed semantic palette exposes an interaction that a single-color
scatter would completely hide.

### Joint distribution — charges vs children

![Jointplot of charges vs children with marginal histograms](figures/cat2_jointplot_children.png)

**Interpretation:** Charges are distributed almost identically across every family size, visually
confirming the negligible r ≈ 0.07 for `children`. **Styling choice:** The marginal on the x-axis
doubles as a count of records per child-number, showing most families have 0–2 children.

### OLS residual diagnostics

![Residual-vs-fitted plots for OLS charges~age and charges~bmi](figures/cat2_residuals.png)

**Interpretation:** Both single-predictor models have very low R² (0.089 and 0.039) and their
residuals fan out and split into layers rather than scattering evenly around zero — clear
heteroscedasticity driven by the skewed target and the omitted smoker effect. **Styling choice:**
A dashed zero reference line plus a LOWESS residual trend and a highlighted R² callout make the
non-constant variance immediately visible.

### Charges by category — boxplots

![Boxplots of charges by sex, smoker and region](figures/cat2_box.png)

**Interpretation:** `smoker` produces a dramatic separation in medians and spread, whereas `sex`
and `region` boxes overlap heavily — early evidence that only smoking status is a reliable driver.
**Styling choice:** Consistent per-category semantic colors let the reader compare the three
panels at a glance while keeping smoker vermillion/blue coding intact.

### Charges by category — violin plots

![Violin plots of charges by sex, smoker and region](figures/cat2_violin.png)

**Interpretation:** The violins expose distribution *shape*: non-smokers (and both sexes, all
regions) form a single low peak, while smokers are distinctly bimodal — a moderate cluster around
$20k and a high one near $40k. **Styling choice:** Inner quartile lines and `cut=0` give an honest
density that stops at the observed data range rather than extrapolating beyond it.

### Charges by category — individual observations

![Strip and swarm plots of charges by category](figures/cat2_strip_swarm.png)

**Interpretation:** Plotting every point confirms the smoker bimodality is real (two dense point
clouds, not an artifact of smoothing) and that `sex`/`region` are thoroughly intermixed.
**Styling choice:** A non-overlapping swarm is used for the small-cardinality `sex` panel, while
jittered semi-transparent strips handle the denser `smoker` and `region` panels without a solid ink blob.

### Mean charges with 95% confidence intervals

![Bar chart of mean charges per category with 95% CI error bars and value labels](figures/cat2_bar_ci.png)

**Interpretation:** Only `smoker` shows non-overlapping confidence intervals (~$8.4k vs ~$32k);
the `sex` and `region` means differ by small amounts with overlapping CIs, i.e. they are not
statistically reliable drivers. **Styling choice:** Each bar is value-labeled with its exact mean
and capped 95% CI whiskers, so magnitude and uncertainty are both quantified on the figure.

### Faceted catplot — smoker effect across regions

![Faceted boxplots of charges by smoker status within each region](figures/cat2_catplot_faceted.png)

**Interpretation:** The smoker gap holds in every region with the same direction and roughly the
same magnitude, demonstrating that the effect is robust and not confounded by geography.
**Styling choice:** Faceting by `region` on a shared y-axis lets the reader compare the smoker
split across panels directly, with the fixed smoker palette repeated in each facet.

### Smoker status — box plus points with mean gap annotated

![Boxplot of charges by smoker with overlaid individual points and mean-gap annotation](figures/cat2_smoker_box_points.png)

**Interpretation:** This headline figure quantifies the single strongest relationship in the
dataset — non-smokers average ≈$8,441 versus ≈$32,050 for smokers, a ~$23.6k (≈3.8×) gap that
dwarfs every other feature effect. **Styling choice:** Overlaying jittered points on a translucent
box, marking each mean with a diamond, and annotating the gap with a double-headed arrow makes the
magnitude and its supporting data unmistakable in one prominent panel.
