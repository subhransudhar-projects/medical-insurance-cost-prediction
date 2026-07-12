# Stage 4 — Actionable Recommendations

Grounded in the modeling + SHAP evidence. Dollar figures are mean annual charges
from 1,337 policyholders.

## 1. Pricing strategy — replace the flat smoker surcharge with a smoker x BMI risk tier
**What:** Rate on four tiers instead of a single smoker flag:
| Tier | Mean charges | vs non-smoker baseline |
|---|---|---|
| Non-smoker (any BMI) | ~$8,400-8,900 | baseline |
| Smoker, non-obese | $21,363 | +$12,900 |
| Smoker, obese (BMI>=30) | $41,558 | +$33,100 |

**Why it helps:** a flat smoker surcharge **under-prices obese smokers by ~$20,000** and
**over-prices non-obese smokers**, inviting adverse selection (lean smokers overpay and leave;
obese smokers underpay and stay). The model shows the risk is interactive, not additive.
**Impact:** more accurate premiums, reduced adverse-selection loss, defensible (SHAP-explainable) rates.

## 2. Wellness — smoking cessation as the #1 cost lever
**What:** Fund and aggressively enroll smokers (prioritizing obese smokers) into cessation programs;
tie premium credits to participation/quit-status.
**Why it helps:** smokers are ~20% of members but ~50% of cost; each sustained quit avoids up to
**~$23,600/yr** (up to ~$33,000 for an obese smoker moving toward the non-smoker baseline).
**Impact:** the single largest addressable cost pool in the book.

## 3. Wellness — target weight management at SMOKERS specifically
**What:** Offer weight/BMI-management programs, but prioritize **smokers with BMI>=30**, not the general obese population.
**Why it helps:** obesity adds ~$20,000/yr **for smokers** but almost nothing for non-smokers
($8,856 obese non-smoker vs $8,441 overall). Spending weight-program dollars on obese non-smokers has
little cost ROI; spending them on obese smokers is high ROI. This is a non-obvious, evidence-based targeting rule.
**Impact:** concentrates limited wellness budget where it actually moves cost.

## 4. Customer segmentation — a 4-tier risk model for care management
**What:** Operationalize the tiers above for underwriting, outreach, and case management. Flag the
obese-smoker tier for proactive disease/case management.
**Why it helps:** this tier is the cost tail; managing it early avoids the largest claims.
**Impact:** focuses clinical resources on the ~few % of members who drive disproportionate cost.

## 5. Data collection — close the blind spot
**What:** Acquire claims history, diagnosis/procedure codes, and chronic-condition flags; add lifestyle
(exercise, alcohol) and family-history fields.
**Why it helps:** the model's worst errors are **high-cost non-smokers** whose drivers are invisible in
the current 7 columns (e.g., a $28k non-smoker the model priced at $8k). No tuning fixes a missing signal.
**Impact:** would lift accuracy beyond the current ~0.85 CV R2 ceiling and reduce large individual mispricings.

## 6. Operational — deploy interpretable, auditable, fair
**What:** Deploy the Gradient Boosting model with per-quote SHAP explanations; monitor for drift; run
periodic fairness checks and **exclude sex and region** as rating factors.
**Why it helps:** SHAP gives regulators and customers a transparent reason for each price; excluding
sex/region (which are not reliable drivers anyway) avoids fairness/compliance exposure.
**Impact:** a model that is accurate, explainable, and defensible.
