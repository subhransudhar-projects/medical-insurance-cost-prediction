## Category 6: Creative & Original Visualizations

This category moves beyond the standard bar/box/scatter vocabulary of the earlier sections to a set of five unconventional, high-impact charts. Each is chosen specifically to expose a real driver of `charges` — above all the dominant `smoker` effect and its interaction with BMI and age — in a way a conventional chart cannot.

### cat6_ridge_charges_by_bmicat_smoker — Ridge / joyplot of charges

![Ridgeline of charges density by BMI category and smoker status](figures/cat6_ridge_charges_by_bmicat_smoker.png)

**Rationale:** A ridgeline stacks one density per BMI category so the *whole distribution* — not just a mean or median — can be compared across four groups at once, with smoker and non-smoker drawn as two overlaid densities inside each row. (Implemented manually with matplotlib and `scipy.gaussian_kde`, since `joypy` is broken under this pandas version.)

**Interpretation:** In every row the non-smoker density (blue) stays a tall, narrow spike anchored below ~$15k regardless of BMI, while the smoker density (orange) sits far to the right and slides further rightward as BMI rises — culminating in the Obese row, where smokers form a broad hump peaking near $40k+. This is the clearest single view of the smoker × obesity interaction: obesity barely moves non-smokers but massively amplifies smokers.

**Styling choice:** Rows are ordered Underweight → Obese top-to-bottom with translucent (alpha 0.55) fills so overlapping curves remain legible, and category labels sit inside each ridge to keep the left margin clean.

### cat6_waterfall_charges_decomposition — Waterfall cost-premium decomposition

![Waterfall decomposing the smoker cost premium](figures/cat6_waterfall_charges_decomposition.png)

**Rationale:** A waterfall turns an abstract "the expensive segment costs a lot more" into an auditable running total, attributing the gap between the non-smoker baseline and the obese, older smoker segment across three additive increments.

**Interpretation:** Starting from the $8,441 non-smoker baseline, the smoker step alone adds ~$26k — dwarfing the older-age (~$7.3k) and obesity (~$5.4k) increments — to reach the ~$47.4k obese-older-smoker mean, a ~$38.9k total gap. Smoking is not one factor among equals; it is the overwhelming component of the premium.

**Styling choice:** A green-free encoding (blue baseline, orange increments, near-black summary bar) with dashed connectors linking each bar's top to the next, and every step labelled with its signed dollar delta.

### cat6_slope_region_smoker — Slope chart of the smoker penalty by region

![Slope chart of mean charges from non-smoker to smoker, per region](figures/cat6_slope_region_smoker.png)

**Rationale:** A slope chart connects each region's non-smoker mean to its smoker mean with a single line, making the *magnitude and consistency* of the smoker penalty across regions readable at a glance — something grouped bars obscure.

**Interpretation:** All four lines fan steeply upward, roughly quadrupling from ~$8–9k (non-smoker) to ~$30–35k (smoker); the southeast starts lowest yet climbs highest, but the overall ordering is broadly preserved, confirming region is a weak, secondary driver next to the near-universal smoker effect.

**Styling choice:** Lines use the fixed `REGION_PALETTE`, endpoints are labelled with dollar values at both axes, and the tightly-clustered non-smoker labels are decluttered with short colored leader lines so no text overlaps.

### cat6_diverging_deviation — Diverging deviation-from-mean bars

![Diverging bars: each BMI x smoker segment's deviation from the overall mean charge](figures/cat6_diverging_deviation.png)

**Rationale:** Plotting each segment as a signed deviation from the overall mean (rather than an absolute level) centers the comparison on a single reference and instantly sorts winners from losers around a common baseline.

**Interpretation:** A clean split emerges — every smoker segment lands to the right (above average), every non-smoker segment to the left (below average), with the sole exception nowhere in sight: the Obese-smoker bar towers at +$28k while even the "cheapest" smoker segment beats every non-smoker segment. Smoker status, not BMI, decides which side of the mean a segment falls on.

**Styling choice:** Positive deviations are orange and negative blue (never red-green), a heavy center reference line marks the overall mean ($13,279, boxed), and bars are sorted so the gradient reads top-to-bottom.

### cat6_sankey_region_smoker_tier — Sankey flow: region → smoker → charge tier

![Sankey diagram of beneficiary flow from region to smoker status to charge tier](figures/cat6_sankey_region_smoker_tier.png)

Interactive HTML: [figures/cat6_sankey_region_smoker_tier.html](figures/cat6_sankey_region_smoker_tier.html)

**Rationale:** A Sankey traces beneficiaries as flows through three stages, revealing *which upstream nodes feed which cost tier* — a routing story that bars and heatmaps cannot tell.

**Interpretation:** The non-smoker node fans out almost entirely into the Low (<$10k) and Mid ($10–30k) tiers, whereas the thin smoker node feeds nearly the entire High (≥$30k) tier; regions distribute smokers and non-smokers in similar proportions, so the High tier is fed almost exclusively by smokers regardless of region.

**Styling choice:** Colorblind-safe Wong node colors with semi-transparent links tinted by their source node, and a subtitle stating the key finding directly on the figure; exported as both interactive HTML and a static PNG via kaleido.
