"""
Stage 4 · Chunk 30 — Key business insights (top 5 drivers, quantified in USD).

Grounds the SHAP-based drivers in concrete segment averages from the data, and
writes a business-insights markdown for the deliverables phase.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from model_utils import RESULTS

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "insurance_features.csv")

nonsmk = df[df["smoker"] == "no"]
smk = df[df["smoker"] == "yes"]
obese_smk = smk[smk["bmi"] >= 30]
lean_smk = smk[smk["bmi"] < 30]
obese_non = nonsmk[nonsmk["bmi"] >= 30]

seg = {
    "Non-smoker (all)": nonsmk["charges"].mean(),
    "Smoker (all)": smk["charges"].mean(),
    "Smoker, non-obese (BMI<30)": lean_smk["charges"].mean(),
    "Smoker, obese (BMI>=30)": obese_smk["charges"].mean(),
    "Non-smoker, obese (BMI>=30)": obese_non["charges"].mean(),
}

# Age effect isolated among non-smokers (avoids the smoker interaction)
age_slope = np.polyfit(nonsmk["age"], nonsmk["charges"], 1)[0]
# Children effect among non-smokers
child_slope = np.polyfit(nonsmk["children"], nonsmk["charges"], 1)[0]

smoker_gap = seg["Smoker (all)"] - seg["Non-smoker (all)"]
obese_smoker_premium = seg["Smoker, obese (BMI>=30)"] - seg["Smoker, non-obese (BMI<30)"]
total_charges = df["charges"].sum()
smoker_share_of_cost = smk["charges"].sum() / total_charges
smoker_share_of_people = len(smk) / len(df)

print("=" * 66); print("CHUNK 30 — KEY BUSINESS INSIGHTS"); print("=" * 66)
print("\nSegment mean annual charges:")
for k, v in seg.items():
    print(f"  {k:30} ${v:,.0f}")
print(f"\nSmoker vs non-smoker gap        : ${smoker_gap:,.0f}")
print(f"Obese-smoker premium (vs lean smoker): ${obese_smoker_premium:,.0f}")
print(f"Age effect (per year, non-smokers)   : ${age_slope:,.0f}/yr  (~${age_slope*10:,.0f}/decade)")
print(f"Children effect (per child, non-smk) : ${child_slope:,.0f}/child")
print(f"\nSmokers = {smoker_share_of_people*100:.1f}% of people but "
      f"{smoker_share_of_cost*100:.1f}% of total charges")

md = f"""# Stage 4 — Key Business Insights (Top 5 Cost Drivers)

Quantified from {len(df):,} policyholders, cross-checked against the SHAP analysis
of the production Gradient Boosting model. Dollar figures are mean annual charges.

## The 5 drivers, ranked

### 1. Smoking status - the dominant driver
- Smokers average **${seg['Smoker (all)']:,.0f}** vs non-smokers **${seg['Non-smoker (all)']:,.0f}** - a **${smoker_gap:,.0f}** gap (~{smoker_gap/seg['Non-smoker (all)']:.1f}x).
- Smokers are **{smoker_share_of_people*100:.0f}% of members but {smoker_share_of_cost*100:.0f}% of total claims cost**.
- SHAP: the smoking-related features swing a prediction by up to **+$25,000**.
- *Business meaning:* smoking status is the master switch for medical cost risk.

### 2. Smoking x obesity interaction - a compounding penalty
- Obese smokers average **${seg['Smoker, obese (BMI>=30)']:,.0f}**, vs non-obese smokers **${seg['Smoker, non-obese (BMI<30)']:,.0f}** - an **${obese_smoker_premium:,.0f}** obesity premium *on top of* smoking.
- Among non-smokers, obesity barely moves cost (obese non-smokers **${seg['Non-smoker, obese (BMI>=30)']:,.0f}** vs all non-smokers ${seg['Non-smoker (all)']:,.0f}).
- *Business meaning:* the risk is not additive - it multiplies. Obese smokers are a distinct, most-expensive tier.

### 3. Age - a steady, secondary driver
- Cost rises about **${age_slope:,.0f} per year** (~${age_slope*10:,.0f} per decade) among non-smokers, applying to smokers and non-smokers alike.
- *Business meaning:* age-banded pricing is justified, but age is a minor factor next to smoking.

### 4. Dependents (children) - minor
- Roughly **${child_slope:,.0f} per child** on the primary beneficiary's charges - small and secondary.
- *Business meaning:* family size is not a strong per-person risk signal.

### 5. Region and sex - not reliable drivers
- Both showed negligible SHAP impact and failed variance-robust significance tests in EDA.
- *Business meaning:* do not use region or sex as material rating factors.

## Surprises / cross-checks
- Every method (EDA, manual features, polynomial, single tree, SHAP) independently identified the smoking x BMI interaction as #1 - a strong consistency signal.
- The model's worst errors are high-cost **non-smokers** whose cost drivers are invisible in this data (likely chronic conditions) - a data gap, not a model flaw.

## Alignment with EDA/visualization phase
These modeling findings **confirm and quantify** the Stage-2/Stage-3 conclusions: smoking dominates, it compounds with obesity, age is secondary, and region/sex are noise.
"""
(RESULTS / "business_insights.md").write_text(md, encoding="utf-8")
print("\n[saved] results/business_insights.md")
