"""
Stage 4 · Chunk 32 — Financial impact estimates (illustrative scenario analysis).

Scales the dataset's segment economics to a standardized 10,000-member book and
applies EXPLICIT, conservative participation/efficacy assumptions. These are
planning estimates, not guarantees.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from model_utils import set_style, save_fig, RESULTS, GREEN, BLUE, VERMILLION, NAVY

set_style()
BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "insurance_features.csv")

BOOK = 10_000
smoker_rate = (df["smoker"] == "yes").mean()
obese_smoker_rate = ((df["smoker"] == "yes") & (df["bmi"] >= 30)).mean()

n_smokers = BOOK * smoker_rate
n_obese_smokers = BOOK * obese_smoker_rate

smoker_gap = 23_610          # smoker vs non-smoker (Chunk 30)
obese_premium = 20_195       # obese vs non-obese smoker (Chunk 30)

print("=" * 66); print("CHUNK 32 — FINANCIAL IMPACT (per 10,000-member book)"); print("=" * 66)
print(f"Smokers per 10k        : {n_smokers:,.0f}")
print(f"Obese smokers per 10k  : {n_obese_smokers:,.0f}")

scenarios = []

# 1. Smoking cessation: enroll all smokers, 10% sustained quit, 50% near-term cost realization
quit_rate, realize = 0.10, 0.50
cost_per_enroll = 500
quitters = n_smokers * quit_rate
gross = quitters * smoker_gap
net = gross * realize - n_smokers * cost_per_enroll
scenarios.append(["Smoking cessation program", f"{n_smokers:,.0f} smokers",
                  gross, gross * realize, n_smokers * cost_per_enroll, net,
                  "10% quit, 50% yr-1 realization, $500/enrollee"])

# 2. Weight management for OBESE SMOKERS: 15% move below BMI30, 50% realization
move_rate = 0.15
cost_per_enroll2 = 800
movers = n_obese_smokers * move_rate
gross2 = movers * obese_premium
net2 = gross2 * realize - n_obese_smokers * cost_per_enroll2
scenarios.append(["Weight mgmt (obese smokers only)", f"{n_obese_smokers:,.0f} obese smokers",
                  gross2, gross2 * realize, n_obese_smokers * cost_per_enroll2, net2,
                  "15% drop below BMI30, 50% realization, $800/enrollee"])

# 3. Repricing obese smokers correctly: recover under-pricing (flat surcharge underprices by ~obese_premium)
recover = 0.60
exposure = n_obese_smokers * obese_premium
net3 = exposure * recover
scenarios.append(["Risk-based repricing (smoker x BMI)", f"{n_obese_smokers:,.0f} obese smokers",
                  exposure, net3, 0.0, net3,
                  "recover 60% of the ~$20k/head under-pricing"])

cols = ["Initiative", "Target", "gross_annual", "realized_annual", "program_cost", "net_annual", "assumptions"]
fin = pd.DataFrame(scenarios, columns=cols).sort_values("net_annual", ascending=False)

print("\n--- Scenario table (USD/year per 10,000 members) ---")
show = fin.copy()
for c in ["gross_annual", "realized_annual", "program_cost", "net_annual"]:
    show[c] = show[c].map(lambda v: f"${v:,.0f}")
print(show[["Initiative", "Target", "realized_annual", "program_cost", "net_annual"]].to_string(index=False))
print(f"\nTOTAL conservative net annual impact per 10k members: ${fin['net_annual'].sum():,.0f}")
print("Assumptions (explicit):")
for _, r in fin.iterrows():
    print(f"  - {r['Initiative']}: {r['assumptions']}")

# Bar chart of net annual impact
fig, ax = plt.subplots(figsize=(11, 6))
order = fin.sort_values("net_annual")
colors = [GREEN, BLUE, VERMILLION][:len(order)]
ax.barh(order["Initiative"], order["net_annual"] / 1e6, color=colors)
for i, v in enumerate(order["net_annual"]):
    ax.text(v/1e6 + 0.03, i, f"${v/1e6:.2f}M", va="center", fontsize=10, color=NAVY, fontweight="medium")
ax.set_title("Conservative Net Annual Impact per 10,000 Members", loc="left")
ax.set_xlabel("net annual impact (USD millions)")
ax.margins(x=0.15)
save_fig(fig, "chunk32_financial_impact")

# Write markdown
md = ["# Stage 4 — Financial Impact Estimates (illustrative)\n",
      f"Scaled to a **10,000-member book** ({n_smokers:,.0f} smokers, {n_obese_smokers:,.0f} obese smokers). "
      "Figures use dataset segment economics + explicit conservative assumptions. **Planning estimates, not guarantees.**\n",
      "| Initiative | Target | Realized $/yr | Program $/yr | Net $/yr |",
      "|---|---|---:|---:|---:|"]
for _, r in fin.iterrows():
    md.append(f"| {r['Initiative']} | {r['Target']} | ${r['realized_annual']:,.0f} | "
              f"${r['program_cost']:,.0f} | **${r['net_annual']:,.0f}** |")
md.append(f"\n**Total conservative net annual impact: ~${fin['net_annual'].sum():,.0f} per 10,000 members "
          f"(~${fin['net_annual'].sum()/BOOK:,.0f} per member).**\n")
md.append("### Assumptions\n")
for _, r in fin.iterrows():
    md.append(f"- **{r['Initiative']}:** {r['assumptions']}.")
md.append("\n### Prioritization (by ROI)\n"
          "1. **Risk-based repricing** — near-zero cost, immediate; fixes structural under-pricing.\n"
          "2. **Smoking cessation** — largest addressable pool; ROI positive even at a conservative 10% quit rate.\n"
          "3. **Targeted weight management** — high per-head value but smaller eligible population.")
(RESULTS / "financial_impact.md").write_text("\n".join(md), encoding="utf-8")
print("\n[saved] results/financial_impact.md")
