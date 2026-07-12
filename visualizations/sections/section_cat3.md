## Category 3: Comparison Visualizations (Group Comparisons)

These figures compare mean and total insurance charges across groups (region, sex, age group, BMI category) with smoker status as the recurring cross-cutting factor. All monetary axes and value labels are in USD, colors are drawn from the colorblind-safe Wong palette, and smoker status uses the fixed semantic mapping (blue = non-smoker, orange = smoker) everywhere.

### Grouped bar — mean charges by region and smoker status
![Mean charges by region grouped by smoker status](figures/cat3_grouped_region_smoker.png)

**Interpretation:** Within every region smokers pay roughly 3.5-4x what non-smokers pay, and that smoker gap dwarfs any between-region differences; the southeast edges out other regions largely because it carries the highest smoker share. Region on its own is a weak driver once smoker status is accounted for.

**Styling choice:** Grouping by the fixed smoker palette (blue/orange) keeps the dominant smoker signal visually separate from the secondary region axis.

### Grouped bar — mean charges by sex and smoker status
![Mean charges by sex grouped by smoker status](figures/cat3_grouped_sex_smoker.png)

**Interpretation:** Male and female non-smokers cost almost the same on average, and the modest male premium among smokers reflects BMI/behavior differences rather than sex itself. Smoker status, not sex, is the decisive split.

**Styling choice:** Value labels sit directly above each bar so the small sex-level differences can be read precisely instead of estimated from bar height.

### Stacked bar — smoker proportion within each region
![100 percent stacked smoker share by region](figures/cat3_stacked_smoker_by_region.png)

**Interpretation:** Smoker prevalence is fairly even across regions (about 18-25%), with the southeast highest at 25.0% and the southwest lowest at 17.8%. This even spread is why region correlates only weakly with charges once smoking is controlled for.

**Styling choice:** A 100%-stacked layout normalizes each region to full height so shares are directly comparable, with percentages labeled inside each segment.

### Heatmap — mean charges by age group and BMI category
![Heatmap of mean charges by age group and BMI category](figures/cat3_heatmap_age_bmi.png)

**Interpretation:** Mean charges rise moving down (older) and right (higher BMI), peaking for 51-64 Obese enrollees (~$19,567) versus the ~$5,662 minimum for the youngest, underweight group. The blank 41-50 Underweight cell simply reflects that no enrollees fall in that rare combination.

**Styling choice:** The sequential viridis colormap (perceptually uniform, colorblind-safe) encodes magnitude while $-formatted cell annotations give exact values.

### Treemap — total charges by region and smoker status
![Treemap of total charges by region and smoker status](figures/cat3_treemap_region_smoker.png)

**Interpretation:** The southeast is the largest single block of total charges, and within it smokers alone account for ~59% of spend despite being a minority of members. Across every region the smoker tiles are disproportionately large relative to their headcount, visualizing how a small smoking population drives a large share of total cost.

**Styling choice:** Built with Plotly (region colored via the shared region palette) so tiles are labeled with group, dollar total, and share of parent; an interactive version is saved at `figures/cat3_treemap_region_smoker.html` alongside the static PNG.

### Boxplot — charges by age group
![Boxplot of charges by age group](figures/cat3_box_charges_by_agegroup.png)

**Interpretation:** Median and mean charges climb steadily with age (mean ~$9,415 at 18-30 rising to ~$18,085 at 51-64), while a persistent cluster of high outliers in every group marks the smokers. Age shifts the whole distribution upward but the outlier smoker band is present at all ages.

**Styling choice:** A viridis-ordered box sequence plus an overlaid diamond mean marker separates the typical (median) from the outlier-inflated mean.

### Grouped bar — mean charges by age group and smoker status
![Mean charges by age group grouped by smoker status](figures/cat3_bar_charges_by_agegroup_smoker.png)

**Interpretation:** Both smokers and non-smokers get more expensive with age, but the smoker bars rise from a far higher base and the absolute smoker/non-smoker gap widens in older groups. Age amplifies, rather than replaces, the smoking effect.

**Styling choice:** Consistent blue/orange smoker hues let the reader track the widening gap across the age axis at a glance.

### Boxplot — charges by BMI category
![Boxplot of charges by BMI category](figures/cat3_box_charges_by_bmicat.png)

**Interpretation:** Medians barely move across BMI categories, but the Obese group shows a much longer upper tail and higher mean, signaling that high BMI raises cost only for a subset (the smokers). The mean-vs-median divergence in the Obese box previews the interaction shown next.

**Styling choice:** Ordered categories (Underweight to Obese) with a diamond mean overlay expose the mean/median split that a plain boxplot would hide.

### Grouped bar — mean charges by BMI category and smoker status (interaction)
![Mean charges by BMI category grouped by smoker status showing the interaction](figures/cat3_bar_charges_by_bmicat_smoker.png)

**Interpretation:** This is the key story: for non-smokers, mean charges are essentially flat across BMI (~$5.5k-$8.9k), but for smokers they jump sharply at obesity, with Obese smokers averaging ~$41,558 -- by far the most expensive segment. BMI matters almost entirely through its interaction with smoking.

**Styling choice:** The annotated arrow flags the Obese-smoker bar so the smoker x obesity interaction reads immediately, with $-formatted labels on every bar for exact comparison.
