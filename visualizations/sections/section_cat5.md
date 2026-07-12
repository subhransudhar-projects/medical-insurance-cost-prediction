## Category 5: Advanced & Specialized Visualizations

This category moves beyond standard 2D statistical plots into 3D structure, geographic schematics, longitudinal-style age progression, and interactive dashboards. It is designed to communicate the headline findings of the dataset — that `smoker` status dominates `charges`, that `age` and `bmi` push costs up steadily, and that `region` differences are small — to a broad audience, including interactive HTML views that let a reader explore the data themselves. Every Plotly figure is exported both as an interactive HTML file and as a static PNG preview.

### cat5_3d_scatter_smoker — 3D scatter of age x bmi x charges by smoker

![3D scatter of age, bmi and charges colored by smoker status](figures/cat5_3d_scatter_smoker.png)

[Interactive version](figures/cat5_3d_scatter_smoker.html)

**Interpretation:** Smokers (vermillion) form a clearly elevated band of `charges` that sits far above the dense blue non-smoker cloud, and within the smoker group cost climbs further as `bmi` rises past 30 — a visibly two-tier structure that no single 2D projection captures as cleanly. This is the strongest visual statement of smoking as the dominant cost driver.

**Styling choice:** Marker opacity is set to 0.55 with small markers so the dense non-smoker cloud does not occlude the elevated smoker points, and the fixed SMOKER_PALETTE (yes=#D55E00, no=#0072B2) keeps the colorblind-safe semantics consistent with every other figure.

### cat5_3d_scatter_region — 3D scatter of age x bmi x charges by region

![3D scatter of age, bmi and charges colored by region](figures/cat5_3d_scatter_region.png)

[Interactive version](figures/cat5_3d_scatter_region.html)

**Interpretation:** Coloring the same age/bmi/charges cloud by `region` shows the four regions thoroughly intermixed at every height, confirming that region carries little independent signal for charges once the dominant smoker/age/bmi structure is present. The elevated high-charge band contains all four region colors in similar proportion.

**Styling choice:** The four-way REGION_PALETTE (Wong blue/green/vermillion/orange) avoids any red-green pairing, and the same 0.55 opacity keeps the overlapping regional points legible.

### cat5_3d_surface_age_bmi — OLS-predicted charges surface

![3D surface of OLS-predicted charges over age and bmi](figures/cat5_3d_surface_age_bmi.png)

**Interpretation:** An OLS fit of `charges ~ age + bmi + age:bmi` predicts a smoothly rising surface — charges increase with both `age` and `bmi`, and the mild interaction tilts the surface so the highest predicted costs sit at the old-and-high-bmi corner. The modest R² (0.117) is expected: this pooled model deliberately excludes `smoker`, so it captures the demographic gradient but not the dominant smoking split.

**Styling choice:** Rendered with matplotlib mplot3d using the colorblind-safe viridis (SEQ_CMAP) so height and color reinforce the same quantity, with axis labels carrying explicit units and the z-axis and colorbar formatted in USD.

### cat5_region_choropleth — Schematic US-state choropleth of mean charges

![Schematic choropleth mapping the four dataset regions onto US states, colored by mean charges](figures/cat5_region_choropleth.png)

[Interactive version](figures/cat5_region_choropleth.html)

**Interpretation:** The southeast group (yellow) carries the highest mean charges (~$14,735) and the southwest the lowest (~$12,347), with northeast and northwest in between — but the total spread is under ~$2,400, reinforcing that region is a weak, statistically unreliable driver. **This map is a SCHEMATIC only:** the dataset has no finer geography than its four census-style regions, so each US state is colored by the mean charges of whichever of the four dataset regions it was assigned to, not by any real state-level data.

**Styling choice:** The sequential viridis scale encodes the single continuous quantity (mean charges) with a USD-prefixed colorbar, and the title explicitly flags the schematic nature so the map is not mistaken for real state-level measurement.

### cat5_region_bubble_map — Schematic region bubble map

![Bubble map with four bubbles at schematic region centroids sized by mean charges](figures/cat5_region_bubble_map.png)

[Interactive version](figures/cat5_region_bubble_map.html)

**Interpretation:** Four bubbles placed at approximate quadrant centroids show the near-equal bubble sizes across regions, making the small magnitude of regional differences in mean charges immediately obvious. **The centroid positions are SCHEMATIC** approximations of the four dataset regions, not measured coordinates — the dataset contains no latitude/longitude information.

**Styling choice:** Bubble size encodes mean charges while the REGION_PALETTE color redundantly identifies each region, and the caption/title flag the schematic centroids so the placement is read as illustrative rather than precise.

### cat5_age_progression — Charges-vs-age progression by smoker

![Smoothed mean charges versus age with 95% CI bands, one line per smoker status](figures/cat5_age_progression.png)

**Interpretation:** Both smoker and non-smoker mean charges rise steadily and roughly in parallel with age, but the vertical gap between the two lines stays wide across the entire 18-64 span (reaching ~$25,398 at ages 60+). Age adds cost for everyone, yet it never closes the smoking penalty.

**Styling choice:** A light 3-point rolling smooth on the per-age means plus translucent 95% CI bands tame the small-sample noise (especially in the smaller smoker group) while the SMOKER_PALETTE lines and faint raw-mean dots preserve the underlying data.

### cat5_summary_multipanel — Static key-insights dashboard

![Six-panel static dashboard summarizing charges distribution, smoker effect, bmi interaction, age and region breakdowns, and correlations](figures/cat5_summary_multipanel.png)

**Interpretation:** The six panels tell the whole story at a glance: charges are right-skewed (a), smokers cost far more (b), the smoker cost explosion kicks in above bmi 30 (c), cost rises with age within both smoker groups (d), regional means are nearly flat (e), and age/bmi are the strongest numeric correlates of charges among a generally weak set (f). Together they preview the same conclusions the earlier categories establish in detail.

**Styling choice:** A single 2x3 GridSpec with lettered panel titles, one bold suptitle, consistent USD-formatted axes, and the shared semantic palettes gives a cohesive one-page executive summary.

### cat5_interactive_dashboard — Interactive bmi-vs-charges explorer

![Interactive Plotly dashboard of bmi vs charges with dropdown filters for smoker, sex, and region](figures/cat5_interactive_dashboard.png)

[Interactive version](figures/cat5_interactive_dashboard.html)

**Interpretation:** The default view scatters `bmi` against `charges` colored by smoker; three dropdowns let a reader filter the points by smoker, sex, or region and watch the elevated smoker band appear, disappear, or thin out — an interactive confirmation that the smoker split, not sex or region, restructures the plot. The static PNG shows the unfiltered default state.

**Styling choice:** Built with plotly graph_objects `updatemenus` restyle buttons so filtering is instant and self-contained in one HTML file, keeping the SMOKER_PALETTE colors, USD-prefixed axis, and the plotly_white template consistent with the rest of the suite.
