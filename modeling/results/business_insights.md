# Stage 4 — Key Business Insights (Top 5 Cost Drivers)

Quantified from 1,337 policyholders, cross-checked against the SHAP analysis
of the production Gradient Boosting model. Dollar figures are mean annual charges.

## The 5 drivers, ranked

### 1. Smoking status - the dominant driver
- Smokers average **$32,050** vs non-smokers **$8,441** - a **$23,610** gap (~2.8x).
- Smokers are **20% of members but 49% of total claims cost**.
- SHAP: the smoking-related features swing a prediction by up to **+$25,000**.
- *Business meaning:* smoking status is the master switch for medical cost risk.

### 2. Smoking x obesity interaction - a compounding penalty
- Obese smokers average **$41,558**, vs non-obese smokers **$21,363** - an **$20,195** obesity premium *on top of* smoking.
- Among non-smokers, obesity barely moves cost (obese non-smokers **$8,856** vs all non-smokers $8,441).
- *Business meaning:* the risk is not additive - it multiplies. Obese smokers are a distinct, most-expensive tier.

### 3. Age - a steady, secondary driver
- Cost rises about **$267 per year** (~$2,671 per decade) among non-smokers, applying to smokers and non-smokers alike.
- *Business meaning:* age-banded pricing is justified, but age is a minor factor next to smoking.

### 4. Dependents (children) - minor
- Roughly **$679 per child** on the primary beneficiary's charges - small and secondary.
- *Business meaning:* family size is not a strong per-person risk signal.

### 5. Region and sex - not reliable drivers
- Both showed negligible SHAP impact and failed variance-robust significance tests in EDA.
- *Business meaning:* do not use region or sex as material rating factors.

## Surprises / cross-checks
- Every method (EDA, manual features, polynomial, single tree, SHAP) independently identified the smoking x BMI interaction as #1 - a strong consistency signal.
- The model's worst errors are high-cost **non-smokers** whose cost drivers are invisible in this data (likely chronic conditions) - a data gap, not a model flaw.

## Alignment with EDA/visualization phase
These modeling findings **confirm and quantify** the Stage-2/Stage-3 conclusions: smoking dominates, it compounds with obesity, age is secondary, and region/sex are noise.
