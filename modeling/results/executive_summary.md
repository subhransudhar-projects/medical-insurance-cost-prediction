# Executive Summary — Predicting & Managing Medical Insurance Costs
*For a VP / executive audience · one page*

## The problem
Medical insurance is priced on risk, but risk is only as good as the model behind it. We were asked to
determine what drives individual medical charges and build a model to predict them — accurately enough to
inform pricing, and transparent enough to defend to regulators and customers.

## What we did
We took a clean dataset of 1,337 policyholders through cleaning, exploratory analysis, 58+ visualizations,
and a rigorous modeling program: 16 model types (from linear regression to XGBoost and stacked ensembles),
all tuned and honestly cross-validated, then explained with modern "explainable AI" (SHAP).

## What we found
1. **Smoking is the dominant cost driver.** Smokers average **$32,050/yr vs $8,441** for non-smokers. They
   are **1 in 5 members but half of all claims cost.**
2. **Smoking and obesity compound.** An obese smoker averages **$41,558** — nearly **5× an equally obese
   non-smoker ($8,856)**. Obesity barely matters *unless* the person also smokes. The risk multiplies; it
   doesn't add.
3. **Age matters, modestly** (~$2,700 per decade). **Region and sex do not** drive cost and should not be
   used in pricing.
4. **Our model predicts charges with ~85% accuracy** (cross-validated), typically within ~$1,800–2,600 of
   a customer's actual annual cost.
5. **Its blind spot is instructive:** the biggest errors are high-cost *non-smokers* whose drivers
   (chronic conditions) aren't in our data — telling us exactly what to collect next.

## What we recommend
1. **Reprice on a smoker×BMI tier**, not a flat smoker surcharge — today's approach under-prices obese
   smokers by ~$20,000 each.
2. **Fund smoking cessation** (the largest addressable cost pool) and **target weight-management at
   smokers specifically** (where it actually pays off).
3. **Acquire claims/diagnosis data** — the highest-value investment to improve accuracy further.
4. **Deploy the model with SHAP explanations**; exclude sex and region as rating factors.

## Expected financial impact (per 10,000 members, conservative)
- **~$13M/yr** in premium-adequacy correction from risk-based repricing (prevented losses)
- **~$1.4M/yr** net from smoking cessation + **~$0.8M/yr** from targeted weight management (claims reductions)
- **~$15M/yr total (~$1,500/member)** — the biggest lever costs almost nothing: just pricing risk correctly.

## Bottom line
One in five members drives half the cost, and we can now identify, price, and act on that risk
transparently. The path to savings is not a fancier algorithm — it is correct pricing and better data.
