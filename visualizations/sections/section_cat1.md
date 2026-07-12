## Category 1: Distribution Visualizations (Univariate)

This category profiles each variable on its own — the four numerics (age, bmi, children, charges) and the three categoricals (sex, smoker, region) — to establish the shape, spread, and skew of every field before any relationship is examined. Throughout, `charges` is the modelling target and `smoker` is flagged as its dominant driver.

### cat1_hist_kde — Histogram with KDE overlay

![Histograms with KDE overlay for the four numeric variables](figures/cat1_hist_kde.png)

**Interpretation:** `charges` is strongly right-skewed (skew +1.51) with a long high-cost tail, `bmi` is near-normal centred around 30, `age` is roughly uniform across 18-64 with a pronounced spike at 18, and `children` is right-skewed with most policyholders having zero. The heavy charges tail is the first sign that a log transform will help downstream modelling.

**Styling choice:** The KDE line is drawn in the vermillion accent over blue bars so the smoothed density stands out against the histogram without introducing a red-green pairing; `children` uses one bin per integer value to avoid smoothing over its discrete nature.

### cat1_box_violin — Violin + box overlay

![Violin plots with boxplot overlay for the four numeric variables](figures/cat1_box_violin.png)

**Interpretation:** The boxes confirm the medians and IQRs while the violins expose shape detail the box hides — the multimodal "shoulders" of `children`, the tight symmetric core of `bmi`, and the dense low cluster of `charges` with many high-side outliers. Only `charges` and `bmi` carry a meaningful number of flagged outliers, and the charges outliers are the high-cost cases worth watching.

**Styling choice:** A translucent sky-blue violin sits behind a narrow solid box with a yellow median line, giving both the full density silhouette and the exact quartile summary in a single readable glyph.

### cat1_density — Smooth density (KDE)

![Filled KDE density plots for the four numeric variables](figures/cat1_density.png)

**Interpretation:** With the histogram bars removed, the mean-vs-median gap becomes the story: for `charges` the mean sits far right of the median, quantifying the right skew, whereas `bmi` shows the two nearly coincident (symmetry). This mean/median separation is the cleanest single-number cue for which variables violate normality.

**Styling choice:** Mean (dashed vermillion) and median (dotted green) reference lines are overlaid with numeric labels so the skew is read directly off the chart rather than inferred from the curve.

### cat1_qq — Q-Q plots vs normal

![Q-Q normality plots for the four numeric variables](figures/cat1_qq.png)

**Interpretation:** `bmi` hugs the normal reference line almost perfectly (R^2 = 0.994), confirming approximate normality, while `charges` bows well above the line at the upper quantiles (R^2 = 0.815, skew +1.51) — a textbook right-skew signature. `age` and `children` deviate at the extremes due to the 18-year spike and the discrete integer steps respectively.

**Styling choice:** Each panel carries an in-plot annotation box with the skew value, its verbal tag, and the fit R^2, so the normality verdict is explicit rather than left to visual judgement.

### cat1_ecdf — Empirical cumulative distribution (ECDF)

![Empirical CDF plots for the four numeric variables](figures/cat1_ecdf.png)

**Interpretation:** The ECDFs read off percentiles directly — the steep early rise of `charges` shows roughly 70% of policies fall under ~$15,000, with the curve flattening into a thin expensive tail. The near-diagonal `age` curve reiterates its uniformity, while `children` climbs in visible integer steps.

**Styling choice:** A median crosshair (horizontal 0.5 guide plus a labelled vertical line) anchors each panel, letting the reader locate the 50th percentile without tracing the curve by eye.

### cat1_bar_counts — Categorical counts

![Count bar charts for sex, smoker, and region](figures/cat1_bar_counts.png)

**Interpretation:** `sex` is essentially balanced, the four `region` groups are near-even, and `smoker` is the notable imbalance at roughly 80% non-smokers to 20% smokers. That skew in the strongest cost driver matters for later modelling, since the expensive smoker group is the minority class.

**Styling choice:** Value labels sit above every bar and each category uses its fixed semantic colour (e.g. smoker via the vermillion/blue palette) so colour meaning stays identical across the entire suite.

### cat1_pie — Percentage breakdown

![Pie charts of percentage breakdown for sex, smoker, and region](figures/cat1_pie.png)

**Interpretation:** The percentage view reinforces the counts: sex splits ~49.5/50.5, region hovers near a 25% quarter each, and smoker sits at ~20.5% yes / 79.5% no. Pies are used only here, where the parts genuinely sum to a meaningful whole.

**Styling choice:** Percentages are printed inside each wedge in bold white and the semantic palette is reused, keeping the categorical colours consistent with the bar, waffle, and downstream charts.

### cat1_waffle — Waffle charts

![Waffle charts (10x10 grid) for sex, smoker, and region](figures/cat1_waffle.png)

**Interpretation:** Each 10x10 grid frames one square as one percent, making the ~20 smoker squares vs ~80 non-smoker squares immediately tangible — a more intuitive read of the class imbalance than either the bar or pie. Sex reads as a near-even split and region as four comparable blocks.

**Styling choice:** Implemented manually (pywaffle unavailable) with largest-remainder rounding so the squares total exactly 100, and the legend prints the precise percentage next to each colour for accuracy alongside the visual impact.

### cat1_charges_binsensitivity — Charges histogram bin sensitivity

![Charges histogram at 15, 30, 60, and 100 bins](figures/cat1_charges_binsensitivity.png)

**Interpretation:** Across 15 to 100 bins the right-skewed shape and low-cost mode are stable, but finer binning progressively resolves a secondary bump around $40,000 — the high-cost (largely smoker) cluster that coarse bins blur away. This confirms the distribution's key features are real, not binning artefacts.

**Styling choice:** A 2x2 grid holds the bin resolution as the only variable, with identical colour and axis treatment across panels so the reader compares shape, not styling.

### cat1_charges_cdf — Charges cumulative distribution by smoker

![Cumulative distribution of charges, overall and split by smoker status](figures/cat1_charges_cdf.png)

**Interpretation:** Splitting the CDF by smoker status separates the two curves dramatically — nearly all non-smokers fall below ~$15,000, whereas the smoker curve is shifted far to the right and barely begins until charges the non-smoker group has almost finished. This single panel is the clearest univariate evidence that `smoker` is the dominant cost driver.

**Styling choice:** The overall CDF is a neutral dark line with the two smoker groups dashed in their semantic colours, and an arrow annotates the overall median (~$9,386) so the split is anchored to a concrete reference value.
