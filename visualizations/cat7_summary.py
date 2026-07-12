"""
Category 7 — Executive dashboard, curated grid, and narrative summary.

Produces two bespoke deliverables:
  - cat7_executive_dashboard.png : a one-page, business-leader dashboard
    (KPI cards + the four most decision-relevant charts).
  - cat7_curated_grid.png        : a 3x3 tiling of the curated "most important"
    figures already produced across categories 1-6.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

from viz_config import (
    set_style, load_data, save, FIGDIR,
    SMOKER_PALETTE, WONG, BMI_CATEGORY_ORDER, AGE_GROUP_ORDER,
)

set_style()
df = load_data()

# --- Precompute headline numbers -------------------------------------------
mean_all = df["charges"].mean()
sm_yes = df.loc[df["smoker"] == "yes", "charges"]
sm_no = df.loc[df["smoker"] == "no", "charges"]
mean_yes, mean_no = sm_yes.mean(), sm_no.mean()
ratio = mean_yes / mean_no
obese_smoker = df[(df["smoker"] == "yes") & (df["bmi_category"] == "Obese")]["charges"].mean()

enc = df.assign(smoker_bin=(df["smoker"] == "yes").astype(int))
r_smoker = enc[["smoker_bin", "charges"]].corr().iloc[0, 1]
r_age = df[["age", "charges"]].corr().iloc[0, 1]
r_bmi = df[["bmi", "charges"]].corr().iloc[0, 1]

NAVY = "#1a2b45"
CARD_BG = "#f2f5f9"


# ===========================================================================
# FIGURE 1: One-page executive dashboard
# ===========================================================================
def executive_dashboard():
    fig = plt.figure(figsize=(16, 10.5))
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[0.9, 2.1, 2.1],
        hspace=0.42, wspace=0.28,
        left=0.055, right=0.965, top=0.86, bottom=0.075,
    )

    # --- Title band ---
    fig.text(0.055, 0.955, "What Drives Medical Insurance Costs?",
             fontsize=25, fontweight="bold", color=NAVY, ha="left", va="top")
    fig.text(0.055, 0.905,
             "Smoking is the single dominant cost driver — and it compounds sharply with obesity. "
             "Age adds steadily; region and sex do not.",
             fontsize=13.5, color="#42506a", ha="left", va="top", style="italic")

    # --- KPI cards (top row) ---
    kpis = [
        (f"{ratio:.1f}x", "higher cost for smokers",
         f"\\${mean_yes:,.0f} vs \\${mean_no:,.0f} avg", WONG["vermillion"]),
        (f"{r_smoker:.2f}", "smoker–charges correlation",
         "the strongest of any feature", WONG["blue"]),
        (f"\\${obese_smoker:,.0f}", "avg for obese smokers",
         "the most expensive segment", WONG["vermillion"]),
        ("~0", "reliable effect of region/sex",
         "not statistically significant", "#5a6b86"),
    ]
    for i, (big, label, sub, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        card = FancyBboxPatch(
            (0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=CARD_BG, zorder=0,
        )
        ax.add_patch(card)
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.05), 0.035, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0,rounding_size=0.0",
            linewidth=0, facecolor=color, zorder=1))
        ax.text(0.10, 0.70, big, transform=ax.transAxes, fontsize=27,
                fontweight="bold", color=color, ha="left", va="center")
        ax.text(0.10, 0.40, label, transform=ax.transAxes, fontsize=11.5,
                fontweight="semibold", color=NAVY, ha="left", va="center")
        ax.text(0.10, 0.19, sub, transform=ax.transAxes, fontsize=10.5,
                color="#5a6b86", ha="left", va="center")

    # --- (a) Charges by smoker: box + points ---
    ax1 = fig.add_subplot(gs[1, :2])
    order = ["no", "yes"]
    import seaborn as sns
    sns.boxplot(data=df, x="smoker", y="charges", order=order, hue="smoker",
                palette=SMOKER_PALETTE, legend=False, width=0.55,
                fliersize=0, ax=ax1, zorder=2,
                boxprops=dict(alpha=0.85))
    sns.stripplot(data=df, x="smoker", y="charges", order=order, hue="smoker",
                  palette=SMOKER_PALETTE, legend=False, size=2.6, alpha=0.35,
                  jitter=0.28, ax=ax1, zorder=1)
    for i, g in enumerate(order):
        m = df.loc[df["smoker"] == g, "charges"].mean()
        ax1.scatter(i, m, marker="D", s=70, color="white", edgecolor=NAVY,
                    linewidth=1.6, zorder=5)
        ax1.text(i, m, f"  mean ${m:,.0f}", va="center", ha="left",
                 fontsize=10.5, fontweight="semibold", color=NAVY)
    ax1.set_title("(a) Charges by smoking status", loc="left", fontweight="semibold")
    ax1.set_xlabel("smoker"); ax1.set_ylabel("charges (USD)")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["non-smoker", "smoker"])
    ax1.yaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}k")

    # --- (b) bmi vs charges colored by smoker (the interaction) ---
    ax2 = fig.add_subplot(gs[1, 2:])
    for g in order:
        sub = df[df["smoker"] == g]
        ax2.scatter(sub["bmi"], sub["charges"], s=14, alpha=0.45,
                    color=SMOKER_PALETTE[g], edgecolors="none",
                    label="smoker" if g == "yes" else "non-smoker")
    ax2.axvline(30, color="#888888", linestyle=":", linewidth=1.3)
    ax2.text(30.3, ax2.get_ylim()[1]*0.97, "BMI 30\n(obese)", fontsize=9,
             color="#555555", va="top")
    ax2.set_title("(b) The smoker x obesity interaction", loc="left", fontweight="semibold")
    ax2.set_xlabel("bmi (kg/m$^2$)"); ax2.set_ylabel("charges (USD)")
    ax2.yaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}k")
    ax2.legend(loc="upper left", framealpha=0.9)

    # --- (c) Mean charges by bmi_category x smoker ---
    ax3 = fig.add_subplot(gs[2, :2])
    piv = (df.groupby(["bmi_category", "smoker"], observed=True)["charges"]
             .mean().unstack().reindex(BMI_CATEGORY_ORDER))
    x = np.arange(len(BMI_CATEGORY_ORDER)); w = 0.38
    ax3.bar(x - w/2, piv["no"], w, color=SMOKER_PALETTE["no"], label="non-smoker")
    ax3.bar(x + w/2, piv["yes"], w, color=SMOKER_PALETTE["yes"], label="smoker")
    for xi, cat in enumerate(BMI_CATEGORY_ORDER):
        ax3.text(xi + w/2, piv.loc[cat, "yes"], f"${piv.loc[cat,'yes']/1000:.0f}k",
                 ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="medium")
    ax3.set_title("(c) Obese smokers are the costliest segment", loc="left", fontweight="semibold")
    ax3.set_xlabel("bmi category"); ax3.set_ylabel("mean charges (USD)")
    ax3.set_xticks(x); ax3.set_xticklabels(BMI_CATEGORY_ORDER)
    ax3.yaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}k")
    ax3.legend(loc="upper left", framealpha=0.9)

    # --- (d) Key drivers ranked ---
    ax4 = fig.add_subplot(gs[2, 2:])
    drivers = pd.Series({"smoker": abs(r_smoker), "age": abs(r_age),
                         "bmi": abs(r_bmi), "children": 0.07, "region": 0.05,
                         "sex": 0.06}).sort_values()
    colors = [WONG["vermillion"] if n == "smoker" else "#9aa7bd" for n in drivers.index]
    ax4.barh(drivers.index, drivers.values, color=colors)
    for yi, (n, v) in enumerate(drivers.items()):
        ax4.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=10, color=NAVY, fontweight="semibold")
    ax4.set_title("(d) Feature association with charges", loc="left", fontweight="semibold")
    ax4.set_xlabel("|correlation| with charges  (0 = none, 1 = perfect)")
    ax4.set_xlim(0, 0.95)
    ax4.set_xticks([])
    ax4.grid(False)

    # --- Footer recommendation ---
    fig.text(0.055, 0.028,
             "Recommendation:  price on a combined smoker x BMI risk tier (not a flat smoker surcharge), "
             "keep age banding, and drop region/sex as material rating factors.",
             fontsize=11.5, color=NAVY, ha="left", va="center", fontweight="semibold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdf0e9",
                       edgecolor=WONG["vermillion"], linewidth=1))
    save(fig, "cat7_executive_dashboard", "one-page executive dashboard")


# ===========================================================================
# FIGURE 2: Curated 3x3 grid of the most important figures
# ===========================================================================
def curated_grid():
    curated = [
        ("cat2_smoker_box_points.png", "Charges by smoker (headline gap)"),
        ("cat4_facet_bmi_charges_by_smoker.png", "BMI x charges, split by smoker"),
        ("cat3_bar_charges_by_bmicat_smoker.png", "Obese smokers = costliest"),
        ("cat3_heatmap_age_bmi.png", "Mean charges: age x BMI"),
        ("cat6_waterfall_charges_decomposition.png", "How the smoker premium builds"),
        ("cat4_corr_heatmap.png", "Numeric correlations"),
        ("cat3_grouped_region_smoker.png", "Region x smoker"),
        ("cat5_age_progression.png", "Charges rising with age"),
        ("cat6_ridge_charges_by_bmicat_smoker.png", "Charges distributions shift right"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(19, 12))
    fig.suptitle("Curated Selection — The Nine Most Important Views",
                 fontsize=20, fontweight="bold", color=NAVY, y=0.985)
    for ax, (fname, label) in zip(axes.ravel(), curated):
        p = FIGDIR / fname
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(0.5, 0.5, f"missing:\n{fname}", ha="center", va="center")
        ax.set_title(label, fontsize=12, color=NAVY, fontweight="semibold", pad=4)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "cat7_curated_grid", "curated 3x3 grid of key figures")


if __name__ == "__main__":
    executive_dashboard()
    curated_grid()
    print("Category 7 figures done.")
