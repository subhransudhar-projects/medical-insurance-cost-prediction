# What Drives Medical Insurance Costs — The Complete Story
*An end-to-end data science case study*

---

## ACT 1 — The Challenge
Health insurers live or die by one question: *what will this customer cost?* Price too high and you lose
them to a competitor; too low and you lose money on every claim. We were handed 1,337 policyholder
records — age, sex, BMI, number of children, smoking status, region, and the annual medical charges each
person incurred — and asked to turn that raw table into something an insurer can act on: an understanding
of what drives cost, a model that predicts it, and a plan that improves the business.

We began by earning the right to model. A clean, validated foundation (no missing values, no duplicates,
correct types, one duplicate row removed in Stage 1) meant every later result would rest on solid ground.

## ACT 2 — The Investigation
Before modeling, we *looked*. Exploratory analysis and 58 visualizations kept pointing at one pattern with
unusual force: **smoking**. Smokers cost about four times what non-smokers cost, and the two groups barely
overlapped. But the sharper discovery was an *interaction* — BMI barely moved costs for non-smokers, yet
for smokers each extra BMI point added over $1,400, and obese smokers formed a distinct, ultra-expensive
cluster. Age mattered steadily; region and sex, under proper statistical testing, did not. We entered the
modeling phase with a hypothesis the data had already half-proven.

## ACT 3 — The Solution
We engineered features that encoded what we'd learned (notably a `smoker × BMI` interaction), split the
data with leakage controls and smoker-stratification, and then ran a deliberate tournament of **16 models**:
- **Linear family** (Linear, Ridge, Lasso, Elastic-Net, Polynomial) — strong, interpretable, and they
  plateaued around 90% on our test split. Lasso independently *deleted* the region features, echoing our EDA.
- **Tree family** (Decision Tree, Random Forest, Gradient Boosting, XGBoost) — broke through the linear
  ceiling to ~92%.
- **Others** (SVR, KNN) and **ensembles** (Voting, two Stackings, AdaBoost, Bagging).

Then we did the thing that matters most: we **stress-tested our own success.** Cross-validation revealed
that the headline 92% was partly a lucky test split — the honest, decision-grade figure is **~85%**. It
also showed the top six models were a *statistical tie*, and that our fancy ensembles bought nothing over a
single well-tuned model. So we chose the **simpler** model — Gradient Boosting — for production: equally
accurate, faster, and explainable. Choosing restraint over a shinier number is itself a finding.

Two moments captured the craft. SVR, run by the book, was our *worst* model (36%); diagnosing that its
target needed scaling turned it into our *best* (92%). And a "research-inspired" AdaBoost that the brief
promised would win instead finished near the bottom — which we reported plainly rather than bending the
result to the citation.

## ACT 4 — The Insights
We opened the black box with SHAP. In dollars, the model starts every customer at a ~$13,400 baseline and
adds or subtracts from there. The `smoker × BMI` factor alone swings a prediction by up to **+$25,000**,
with a visible **cliff at BMI 30** for smokers. We can explain any individual quote: an older obese smoker
built up to $42,667 (actual $42,000); a young lean non-smoker down to $4,363. Where they live and their sex
changed almost nothing — the fourth independent confirmation of the same truth.

The model's *mistakes* were as informative as its accuracy. Its worst misses were high-cost **non-smokers**
it priced far too low — people expensive for reasons (chronic illness, surgeries) simply absent from our
seven columns. The model isn't broken; it's flying with incomplete instruments, and it told us exactly
which instrument to add.

## ACT 5 — The Recommendations
Quantified on a 10,000-member book:
1. **Reprice on a smoker×BMI tier** — today's flat surcharge under-prices obese smokers by ~$20k each;
   correcting it is worth **~$13M/yr** in prevented losses at near-zero cost.
2. **Fund smoking cessation** — the largest addressable pool (~$1.4M/yr net), since 20% of members drive
   50% of cost.
3. **Target weight-management at smokers specifically** (~$0.8M/yr) — obesity only costs money when paired
   with smoking, so that's where the program dollars belong.
4. **Buy better data** (claims/diagnoses) — the single highest-value move to push accuracy past ~85%.
5. **Deploy transparently** — SHAP-explained quotes, with sex and region excluded for fairness.

## ACT 6 — The Skills Demonstrated
- **Technical:** end-to-end pipeline; 16 tuned models; leakage-controlled, stratified, cross-validated
  evaluation; SHAP explainability; reproducible, modular code with a results registry.
- **Statistical integrity:** quantified test-split optimism (+0.076), reported the honest ~0.85 over the
  flattering 0.92, and refused to overstate ensembles or a cited-but-unreplicated result.
- **Business acumen:** translated `smoker_bmi` into a pricing tier, a targeted wellness rule, and a
  dollar-quantified, ROI-ranked action plan.
- **Communication:** one narrative, three registers — executive, technical, and this story — each matched
  to its audience.
- **Critical thinking & leadership:** diagnosed the SVR failure, designed the leakage fix, coordinated the
  work in disciplined chunks, and chose the simpler production model on purpose.

**The one-line takeaway:** *One in five members drives half the cost; smoking-and-obesity is the engine;
and the business can now price, target, and explain that risk — accurately, honestly, and profitably.*
