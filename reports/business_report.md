# The Insurance Cost Prediction Story
*A business-focused report · all figures are the project's actual verified results*

---

## CHAPTER 1: The Challenge
Medical insurance costs are rising, and insurers must predict them accurately to price policies
correctly, identify high-risk customers, design effective wellness programs, and manage financial risk.

**Illustrative scenario (used to frame the analysis):** a mid-sized insurer facing inaccurate premium
pricing, churn from unexpected rate increases, poorly-targeted wellness spend, and a lack of data-driven
insight. The goal: replace intuition with an accurate, explainable, defensible cost model.

## CHAPTER 2: The Data
1,337 policyholders (after cleaning) with:
- **Demographics:** age, sex, children
- **Health indicators:** BMI, smoking status
- **Geographic:** region (4 US census regions)
- **Target:** annual medical charges (USD)

A typical portfolio: ~20.5% smokers, balanced sex, even regional spread — 0 missing values, no duplicates.

## CHAPTER 3: The Investigation (EDA & Visualization)

**FINDING 1 — Smoking dominates costs.**
Smokers average **$32,050** vs **$8,441** for non-smokers — a **3.8× difference** and the single strongest
predictor. Smokers are 20.5% of members but **49.5% of total claims cost**.

**FINDING 2 — The smoker × BMI interaction.**
- Obese smokers: **$41,558** (the most expensive segment)
- Non-obese smokers: **$21,363**
- Non-smokers (any BMI): **~$8,400–8,900**
BMI drives cost **primarily for smokers**; an obese smoker pays ~$20,000 more than a non-obese smoker,
while an obese non-smoker pays almost the same as any non-smoker.

**FINDING 3 — Age is a secondary driver.**
Charges rise ~**$267/year** (~$2,700/decade), for smokers and non-smokers alike; the smoker gap stays wide
across all ages rather than closing.

**FINDING 4 — Regional variation is minimal.**
The Southeast is nominally highest, but the difference **fails variance-robust significance tests**
(Welch ANOVA p≈0.05, Kruskal-Wallis p≈0.20); region is not a reliable driver — and its slight edge tracks
its higher average BMI, not geography per se.

**FINDING 5 — Sex is not a driver.**
No practically meaningful difference between male and female charges (Cohen's d ≈ 0.12), and no meaningful
interaction with other variables.

## CHAPTER 4: The Solution (Modeling)
We built, tuned, and honestly evaluated **17 model configurations**. Selected results (test split / 5-fold CV):

| Model | Test R² | CV R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| SVR (target-scaled) | 0.924 | 0.847 | $3,317 | $1,756 |
| Voting (top 3) | 0.924 | 0.848 | $3,316 | $1,931 |
| Stacking (Linear meta) | 0.922 | — | $3,352 | $2,112 |
| **Gradient Boosting** | 0.920 | 0.840 | $3,404 | $2,108 |
| XGBoost | 0.919 | 0.844 | $3,422 | $2,106 |
| Random Forest | 0.918 | 0.840 | $3,440 | $2,150 |
| Ridge Regression | 0.898 | 0.825 | $3,845 | $2,534 |
| Linear Regression | 0.896 | 0.825 | $3,867 | $2,548 |

**Selected model: Gradient Boosting.** The top ~6 models are a *statistical tie* (CV-mean spread 0.008 vs
CV std 0.040), so rather than crown a marginally-higher ensemble we chose the model that is accurate,
**simpler, and natively explainable**. Honest performance: **CV R² ≈ 0.85, RMSE ≈ $3,400** — predictions
typically within ~$1,800–2,600 of actual annual cost.

*(Note: our ensembles **tied** rather than beat the best single model — their base learners make correlated
errors — so we did not overstate them. And the honest CV ~0.85 is quoted over the single-split 0.92.)*

## CHAPTER 5: The Interpretation (SHAP)

**Global feature importance (share of total SHAP weight):**
1. Smoking-related features (`smoker_bmi`, `smoker_age`, `smoker`, …) — **62%**
2. Age (raw) — **11%**
3. BMI and age×BMI terms — most remaining weight
4. Region + sex combined — **3.6%** (negligible)

The single most important feature, `smoker_bmi`, carries a **mean $6,183** impact per prediction and swings
an individual's estimate by up to **+$25,000**. Dependence plots reveal a **step-change at BMI 30** for
smokers — obesity flips smokers into a distinctly higher-cost tier.

**Individual prediction example (a real case from the test set):**
- 54-year-old **smoker**, BMI 30.8, 1 child → **predicted $42,667** (actual $42,000)
- Built from a $13,356 baseline: `smoker_bmi` **+$22,712**, `smoker_age` +$2,172, `age_bmi` +$1,367, …
- Contrast: a 19-year-old non-smoker (BMI 24.5) → predicted **$4,363** (`smoker_bmi` −$4,059, age −$2,151)

## CHAPTER 6: The Recommendations

**1. Pricing** — replace the flat smoker surcharge with a **smoker × BMI risk tier**; today's approach
under-prices obese smokers by ~$20,000 each and over-prices lean smokers.

**2. Wellness** — fund **smoking cessation** (largest addressable pool) and target **weight management at
smokers specifically** (obesity only costs money when paired with smoking).

**3. Customer segmentation (4 tiers):**
- Tier 1 — obese smokers (highest risk, ~$41,558)
- Tier 2 — non-obese smokers (~$21,363)
- Tier 3 — older non-smokers
- Tier 4 — young, healthy non-smokers (lowest risk)

**4. Data collection** — acquire claims history, diagnoses, and behavioral-health data; the model's worst
errors are high-cost non-smokers whose drivers are invisible in the current features.

**5. Operations** — deploy the model with per-quote SHAP explanations for transparent underwriting;
**exclude sex and region** as rating factors (fairness + they aren't predictive).

## CHAPTER 7: The Financial Impact
Conservative scenario analysis, **per 10,000 members** (explicit assumptions in `financial_impact.md`):

| Initiative | Net annual impact | Type |
|---|---|---|
| Risk-based repricing (smoker × BMI) | **~$13.1M** | premium adequacy (prevented losses) |
| Smoking cessation (10% quit) | **~$1.39M** | claims reduction |
| Weight management (obese smokers) | **~$0.78M** | claims reduction |
| **Total** | **~$15.3M (~$1,531/member)** | |

Scaled to a 50,000-member portfolio, the same assumptions imply a mid-eight-figure annual opportunity —
the largest lever (correct risk-based pricing) costing almost nothing to implement. *(Repricing is
premium-adequacy correction, not a reduction in medical claims — we keep the two categories distinct rather
than blurring them into one inflated "savings" number.)*

## CHAPTER 8: The Journey
- **Technical excellence:** 17 models tuned & evaluated; leakage-controlled feature engineering; SHAP
  explainability; reproducible, modular code.
- **Business acumen:** industry-grounded, ROI-ranked recommendations with dollar-quantified impact and a
  customer-segmentation strategy.
- **Communication:** one story in three registers (executive, technical, narrative) + 58 charts.
- **Critical thinking:** model choice justified on principle; the SVR failure diagnosed and fixed; the honest
  CV number and unreplicated literature claims reported plainly; limitations and next steps stated.

**Bottom line:** *One in five members drives half the cost; smoking-and-obesity is the engine; and the
business can now price, target, and explain that risk — accurately, honestly, and profitably.*
