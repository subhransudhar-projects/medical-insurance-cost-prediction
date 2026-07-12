"""
CATEGORY 3 — Comparison / Group visualizations
Medical Insurance Cost Dataset.

Produces publication-quality group-comparison figures: grouped bars, stacked
proportion bars, a mean-charges heatmap (age_group x bmi_category), a
hierarchical treemap (region x smoker), and age/BMI comparison charts that
surface the smoker x obesity interaction.

Run (from the dataset dir):
  "<.venv>\\Scripts\\python.exe" visualizations\\cat3_comparisons.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Make the script's own folder importable so `from viz_config import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_config import (  # noqa: E402
    set_style, load_data, save,
    WONG, SMOKER_PALETTE, SEX_PALETTE, REGION_PALETTE, SEQ_CMAP, ACCENT,
    AGE_GROUP_ORDER, BMI_CATEGORY_ORDER, REGION_ORDER, TARGET, FIGDIR,
)

set_style()
df = load_data()

SMOKER_ORDER = ["no", "yes"]
SEX_ORDER = ["female", "male"]
SMOKER_LABELS = {"no": "non-smoker", "yes": "smoker"}


def _usd(x):
    return f"${x:,.0f}"


def _label_grouped_bars(ax, fmt=_usd, fontsize=9.5, pad_frac=0.012):
    """Label every bar patch above its top (works for grouped/hue bars)."""
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        h = p.get_height()
        if h is None or np.isnan(h) or h == 0:
            continue
        ax.text(p.get_x() + p.get_width() / 2, h + ymax * pad_frac,
                fmt(h), ha="center", va="bottom",
                fontsize=fontsize, color="#222222", weight="medium")


def _smoker_hue_bar(ax, data, x, order, xlabel, title):
    """Shared helper: grouped mean-charges bar with smoker hue + value labels."""
    sns.barplot(
        data=data, x=x, y=TARGET, hue="smoker",
        order=order, hue_order=SMOKER_ORDER,
        palette=SMOKER_PALETTE, errorbar=None,
        edgecolor="white", ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean charges (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    top = data.groupby([x, "smoker"], observed=True)[TARGET].mean().max()
    ax.set_ylim(0, top * 1.16)
    _label_grouped_bars(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [SMOKER_LABELS[s] for s in SMOKER_ORDER],
              title="smoker status", loc="upper left", frameon=True)


# ===========================================================================
# 1. cat3_grouped_region_smoker — mean charges by region, grouped by smoker
# ===========================================================================
def fig_grouped_region_smoker():
    fig, ax = plt.subplots(figsize=(11, 7))
    _smoker_hue_bar(
        ax, df, x="region", order=REGION_ORDER,
        xlabel="region",
        title="Mean charges by region, grouped by smoker status",
    )
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    save(fig, "cat3_grouped_region_smoker",
         "grouped mean charges: region x smoker")


# ===========================================================================
# 2. cat3_grouped_sex_smoker — mean charges by sex, grouped by smoker
# ===========================================================================
def fig_grouped_sex_smoker():
    fig, ax = plt.subplots(figsize=(9.5, 7))
    _smoker_hue_bar(
        ax, df, x="sex", order=SEX_ORDER,
        xlabel="sex",
        title="Mean charges by sex, grouped by smoker status",
    )
    fig.tight_layout()
    save(fig, "cat3_grouped_sex_smoker",
         "grouped mean charges: sex x smoker")


# ===========================================================================
# 3. cat3_stacked_smoker_by_region — 100% stacked smoker proportion by region
# ===========================================================================
def fig_stacked_smoker_by_region():
    ct = (df.groupby("region", observed=True)["smoker"]
            .value_counts(normalize=True)
            .unstack()
            .reindex(index=REGION_ORDER, columns=SMOKER_ORDER))
    fig, ax = plt.subplots(figsize=(11, 7))
    bottom = np.zeros(len(REGION_ORDER))
    for level in SMOKER_ORDER:
        vals = ct[level].values * 100
        bars = ax.bar(REGION_ORDER, vals, bottom=bottom,
                      color=SMOKER_PALETTE[level], edgecolor="white",
                      width=0.62, label=SMOKER_LABELS[level])
        for bar, v, b in zip(bars, vals, bottom):
            if v <= 0:
                continue
            txtcol = "white" if level == "yes" else "white"
            ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                    f"{v:.1f}%", ha="center", va="center",
                    fontsize=11, color=txtcol, weight="bold")
        bottom += vals
    ax.set_title("Smoker vs non-smoker proportion within each region")
    ax.set_xlabel("region", labelpad=8)
    ax.set_ylabel("share of residents (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(title="smoker status", loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "cat3_stacked_smoker_by_region",
         "100% stacked smoker share by region")


# ===========================================================================
# 4. cat3_heatmap_age_bmi — mean charges heatmap (age_group x bmi_category)
# ===========================================================================
def fig_heatmap_age_bmi():
    pivot = (df.pivot_table(index="age_group", columns="bmi_category",
                            values=TARGET, aggfunc="mean", observed=True)
               .reindex(index=AGE_GROUP_ORDER, columns=BMI_CATEGORY_ORDER))
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sns.heatmap(
        pivot, ax=ax, cmap=SEQ_CMAP, annot=True, fmt=",.0f",
        annot_kws={"fontsize": 11, "weight": "medium"},
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "mean charges (USD)", "shrink": 0.85},
    )
    # Prefix annotations with $ by rewriting cell texts.
    for t in ax.texts:
        try:
            val = float(t.get_text().replace(",", ""))
        except ValueError:
            continue
        t.set_text(f"${val:,.0f}")
    ax.set_title("Mean charges by age group and BMI category")
    ax.set_xlabel("BMI category")
    ax.set_ylabel("age group (years)")
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    save(fig, "cat3_heatmap_age_bmi",
         "mean charges heatmap age_group x bmi_category")


# ===========================================================================
# 5. cat3_treemap_region_smoker — hierarchical treemap (plotly)
# ===========================================================================
def fig_treemap_region_smoker():
    agg = (df.groupby(["region", "smoker"], observed=True)[TARGET]
             .sum().reset_index())
    agg["smoker_label"] = agg["smoker"].map(SMOKER_LABELS)

    import plotly.express as px

    color_map = {
        "(?)": "#DDDDDD",
        **{r: REGION_PALETTE[r] for r in REGION_ORDER},
    }
    fig = px.treemap(
        agg,
        path=[px.Constant("all regions"), "region", "smoker_label"],
        values=TARGET,
        color="region",
        color_discrete_map=color_map,
        title="Total charges by region and smoker status (hierarchical)",
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percentParent:.1%} of parent",
        textfont_size=15,
        marker_line_width=1.5, marker_line_color="white",
        tiling_pad=3,
    )
    fig.update_layout(
        margin=dict(t=70, l=10, r=10, b=10),
        title_font_size=20,
        font=dict(family="DejaVu Sans, Arial", size=14),
        width=1200, height=760,
    )
    html_path = FIGDIR / "cat3_treemap_region_smoker.html"
    fig.write_html(str(html_path))
    png_path = FIGDIR / "cat3_treemap_region_smoker.png"
    fig.write_image(str(png_path), scale=2)
    print(f"[saved] {html_path.name} (interactive)")
    print(f"[saved] {png_path.name}")


# ===========================================================================
# 6. cat3_box_charges_by_agegroup — boxplot of charges by age_group
# ===========================================================================
def fig_box_charges_by_agegroup():
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = sns.color_palette(SEQ_CMAP, len(AGE_GROUP_ORDER))
    sns.boxplot(
        data=df, x="age_group", y=TARGET, order=AGE_GROUP_ORDER,
        hue="age_group", legend=False, palette=palette, ax=ax,
        width=0.6, linewidth=1.2, fliersize=3,
        flierprops={"marker": "o", "markerfacecolor": "#888888",
                    "markeredgecolor": "none", "alpha": 0.4},
        medianprops={"color": WONG["yellow"], "linewidth": 2.2},
    )
    means = df.groupby("age_group", observed=True)[TARGET].mean().reindex(AGE_GROUP_ORDER)
    for i, m in enumerate(means.values):
        ax.text(i, m, f"mean ${m:,.0f}", ha="center", va="bottom",
                fontsize=9.5, color="#222222", weight="medium")
    ax.scatter(range(len(means)), means.values, marker="D", s=42,
               color=ACCENT, edgecolor="white", zorder=5, label="mean")
    ax.set_title("Charges distribution by age group")
    ax.set_xlabel("age group (years)")
    ax.set_ylabel("charges (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    save(fig, "cat3_box_charges_by_agegroup", "charges boxplot by age group")


# ===========================================================================
# 7. cat3_bar_charges_by_agegroup_smoker — mean charges by age_group x smoker
# ===========================================================================
def fig_bar_charges_by_agegroup_smoker():
    fig, ax = plt.subplots(figsize=(11.5, 7))
    _smoker_hue_bar(
        ax, df, x="age_group", order=AGE_GROUP_ORDER,
        xlabel="age group (years)",
        title="Mean charges by age group, grouped by smoker status",
    )
    fig.tight_layout()
    save(fig, "cat3_bar_charges_by_agegroup_smoker",
         "grouped mean charges: age_group x smoker")


# ===========================================================================
# 8. cat3_box_charges_by_bmicat — boxplot of charges by bmi_category
# ===========================================================================
def fig_box_charges_by_bmicat():
    fig, ax = plt.subplots(figsize=(11, 7))
    palette = sns.color_palette(SEQ_CMAP, len(BMI_CATEGORY_ORDER))
    sns.boxplot(
        data=df, x="bmi_category", y=TARGET, order=BMI_CATEGORY_ORDER,
        hue="bmi_category", legend=False, palette=palette, ax=ax,
        width=0.6, linewidth=1.2, fliersize=3,
        flierprops={"marker": "o", "markerfacecolor": "#888888",
                    "markeredgecolor": "none", "alpha": 0.4},
        medianprops={"color": WONG["yellow"], "linewidth": 2.2},
    )
    means = df.groupby("bmi_category", observed=True)[TARGET].mean().reindex(BMI_CATEGORY_ORDER)
    for i, m in enumerate(means.values):
        ax.text(i, m, f"mean ${m:,.0f}", ha="center", va="bottom",
                fontsize=9.5, color="#222222", weight="medium")
    ax.scatter(range(len(means)), means.values, marker="D", s=42,
               color=ACCENT, edgecolor="white", zorder=5, label="mean")
    ax.set_title("Charges distribution by BMI category")
    ax.set_xlabel("BMI category")
    ax.set_ylabel("charges (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    save(fig, "cat3_box_charges_by_bmicat", "charges boxplot by BMI category")


# ===========================================================================
# 9. cat3_bar_charges_by_bmicat_smoker — mean charges by bmi_category x smoker
#    (reveals the smoker x obesity interaction)
# ===========================================================================
def fig_bar_charges_by_bmicat_smoker():
    fig, ax = plt.subplots(figsize=(11.5, 7))
    _smoker_hue_bar(
        ax, df, x="bmi_category", order=BMI_CATEGORY_ORDER,
        xlabel="BMI category",
        title="Mean charges by BMI category, grouped by smoker status\n"
              "(smoker x obesity interaction)",
    )
    # Emphasise the interaction: annotate the Obese-smoker segment.
    obese_smoker = df[(df["bmi_category"] == "Obese") & (df["smoker"] == "yes")][TARGET].mean()
    ax.annotate("Obese smokers:\nmost expensive segment",
                xy=(len(BMI_CATEGORY_ORDER) - 1 + 0.2, obese_smoker),
                xytext=(len(BMI_CATEGORY_ORDER) - 2.1, obese_smoker * 0.92),
                fontsize=10.5, color=WONG["vermillion"], weight="semibold",
                ha="left",
                arrowprops={"arrowstyle": "->", "color": WONG["vermillion"],
                            "linewidth": 1.6})
    fig.tight_layout()
    save(fig, "cat3_bar_charges_by_bmicat_smoker",
         "grouped mean charges: bmi_category x smoker (interaction)")


if __name__ == "__main__":
    fig_grouped_region_smoker()
    fig_grouped_sex_smoker()
    fig_stacked_smoker_by_region()
    fig_heatmap_age_bmi()
    fig_treemap_region_smoker()
    fig_box_charges_by_agegroup()
    fig_bar_charges_by_agegroup_smoker()
    fig_box_charges_by_bmicat()
    fig_bar_charges_by_bmicat_smoker()
    print("\nAll Category 3 figures generated.")
