"""
CATEGORY 2 — Relationship / Bivariate visualizations (feature vs. target = charges)
for the Medical Insurance Cost Dataset.

Run (from the dataset directory):
    "....venv\\Scripts\\python.exe" visualizations\\cat2_relationships.py

Produces PNG+PDF figures into visualizations/figures/ (all prefixed cat2_).
"""
import sys
from pathlib import Path

# Make sibling viz_config importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from viz_config import (
    set_style, load_data, save,
    SMOKER_PALETTE, SEX_PALETTE, REGION_PALETTE, SEQ_CMAP, ACCENT, WONG,
    REGION_ORDER,
)

# --------------------------------------------------------------------------
set_style()
df = load_data()

NUM_FEATURES = [
    ("age", "age (years)"),
    ("bmi", "bmi (kg/m$^2$)"),
    ("children", "children (count)"),
]
CHARGES_LABEL = "charges (USD)"
CAT_FEATURES = [
    ("sex", SEX_PALETTE, None),
    ("smoker", SMOKER_PALETTE, ["no", "yes"]),
    ("region", REGION_PALETTE, REGION_ORDER),
]

# Semantic order for smoker hue everywhere
SMOKER_ORDER = ["no", "yes"]


def _usd(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))


def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p


# ==========================================================================
# 1. cat2_scatter_regline — scatter + linear regression + CI (1x3)
# ==========================================================================
def fig_scatter_regline():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, label) in zip(axes, NUM_FEATURES):
        sns.regplot(
            data=df, x=col, y="charges", ax=ax,
            scatter_kws=dict(alpha=0.28, s=22, color=WONG["skyblue"],
                             edgecolor="none"),
            line_kws=dict(color=ACCENT, lw=2.4),
            ci=95,
        )
        r, _ = pearson(df[col], df["charges"])
        ax.set_xlabel(label)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges vs {col}")
        _usd(ax)
        ax.text(0.04, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=WONG["blue"], alpha=0.9))
    fig.suptitle("Numeric features vs charges — linear fit with 95% CI",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_scatter_regline",
         "Linear regression of charges on age, bmi, children")


# ==========================================================================
# 2. cat2_scatter_loess — scatter + LOWESS smooth (1x3), Pearson r annotated
# ==========================================================================
def fig_scatter_loess():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, label) in zip(axes, NUM_FEATURES):
        sns.regplot(
            data=df, x=col, y="charges", ax=ax, lowess=True,
            scatter_kws=dict(alpha=0.25, s=22, color=WONG["green"],
                             edgecolor="none"),
            line_kws=dict(color=WONG["vermillion"], lw=2.6),
        )
        r, p = pearson(df[col], df["charges"])
        ax.set_xlabel(label)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges vs {col} (LOWESS)")
        _usd(ax)
        ax.text(0.04, 0.95, f"Pearson r = {r:.2f}\np = {p:.1e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=WONG["green"], alpha=0.9))
    fig.suptitle("Numeric features vs charges — nonparametric LOWESS trend",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_scatter_loess",
         "LOWESS smoothed trend of charges vs numeric features")


# ==========================================================================
# 3. cat2_hexbin — hexbin density (1x3) with colorbar labeled "count"
# ==========================================================================
def fig_hexbin():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, label) in zip(axes, NUM_FEATURES):
        gridsize = 18 if col == "children" else 30
        hb = ax.hexbin(df[col], df["charges"], gridsize=gridsize,
                       cmap=SEQ_CMAP, mincnt=1, linewidths=0.2)
        ax.set_xlabel(label)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges vs {col} — density")
        _usd(ax)
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("count", fontsize=11)
    fig.suptitle("Numeric features vs charges — hexbin density",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_hexbin", "Hexbin density of charges vs numeric features")


# ==========================================================================
# 4. cat2_jointplot_{age,bmi,children} — jointplot with marginals
# ==========================================================================
def fig_jointplots():
    # age — colored by smoker (parallel bands)
    for col, label, hue in [
        ("age", "age (years)", "smoker"),
        ("bmi", "bmi (kg/m$^2$)", "smoker"),
        ("children", "children (count)", None),
    ]:
        if hue:
            jg = sns.jointplot(
                data=df, x=col, y="charges", hue="smoker",
                hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
                height=7.2, alpha=0.5, s=26,
                marginal_kws=dict(common_norm=False),
            )
        else:
            jg = sns.jointplot(
                data=df, x=col, y="charges", height=7.2,
                color=WONG["blue"],
                joint_kws=dict(alpha=0.4, s=26, edgecolor="none"),
                marginal_kws=dict(color=WONG["blue"]),
            )
        r, _ = pearson(df[col], df["charges"])
        jg.ax_joint.set_xlabel(label)
        jg.ax_joint.set_ylabel(CHARGES_LABEL)
        jg.ax_joint.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        jg.ax_joint.text(
            0.04, 0.96, f"Pearson r = {r:.2f}", transform=jg.ax_joint.transAxes,
            ha="left", va="top", fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#666666", alpha=0.9))
        jg.figure.suptitle(f"charges vs {col} — joint distribution",
                           fontsize=15, weight="bold", y=1.01)
        save(jg.figure, f"cat2_jointplot_{col}",
             f"Jointplot of charges vs {col}")


# ==========================================================================
# 5. cat2_residuals — OLS residuals for charges~age and charges~bmi (1x2)
# ==========================================================================
def fig_residuals():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, col in zip(axes, ["age", "bmi"]):
        X = sm.add_constant(df[col])
        model = sm.OLS(df["charges"], X).fit()
        fitted = model.fittedvalues
        resid = model.resid
        ax.scatter(fitted, resid, alpha=0.3, s=22, color=WONG["skyblue"],
                   edgecolor="none")
        # LOWESS of residuals to reveal structure
        sns.regplot(x=fitted, y=resid, ax=ax, lowess=True, scatter=False,
                    line_kws=dict(color=ACCENT, lw=2.2))
        ax.axhline(0, color="#444444", lw=1.1, ls="--")
        ax.set_xlabel(f"fitted charges (USD) — OLS(charges ~ {col})")
        ax.set_ylabel("residual (USD)")
        ax.set_title(f"Residuals vs fitted — charges ~ {col}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        ax.text(0.04, 0.96,
                f"R$^2$ = {model.rsquared:.3f}\nfunnel shape → heteroscedasticity",
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                weight="medium",
                bbox=dict(boxstyle="round,pad=0.35", fc=WONG["yellow"],
                          ec="#999999", alpha=0.85))
    fig.suptitle("OLS residual diagnostics — non-constant variance (heteroscedastic)",
                 fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "cat2_residuals", "Residual plots for simple OLS models")


# ==========================================================================
# Categorical helpers
# ==========================================================================
def _cat_order(name, order):
    return order if order is not None else sorted(df[name].unique())


# ==========================================================================
# 6. cat2_box — boxplot of charges by category (1x3)
# ==========================================================================
def fig_box():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, pal, order) in zip(axes, CAT_FEATURES):
        order = _cat_order(col, order)
        sns.boxplot(data=df, x=col, y="charges", ax=ax, order=order,
                    hue=col, palette=pal, legend=False,
                    fliersize=2, linewidth=1.2)
        ax.set_xlabel(col)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges by {col}")
        _usd(ax)
        if col == "region":
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Charges distribution by category — boxplots",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_box", "Boxplots of charges by categorical features")


# ==========================================================================
# 7. cat2_violin — violin of charges by category (1x3)
# ==========================================================================
def fig_violin():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, pal, order) in zip(axes, CAT_FEATURES):
        order = _cat_order(col, order)
        sns.violinplot(data=df, x=col, y="charges", ax=ax, order=order,
                       hue=col, palette=pal, legend=False,
                       cut=0, inner="quartile", linewidth=1.1)
        ax.set_xlabel(col)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges by {col}")
        _usd(ax)
        if col == "region":
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Charges distribution shape by category — violin plots",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_violin", "Violin plots of charges by categorical features")


# ==========================================================================
# 8. cat2_strip_swarm — strip/swarm of charges by category (1x3)
# ==========================================================================
def fig_strip_swarm():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, pal, order) in zip(axes, CAT_FEATURES):
        order = _cat_order(col, order)
        if col == "sex":
            sns.swarmplot(data=df, x=col, y="charges", ax=ax, order=order,
                          hue=col, palette=pal, legend=False, size=2.6)
            kind = "swarm"
        else:
            sns.stripplot(data=df, x=col, y="charges", ax=ax, order=order,
                          hue=col, palette=pal, legend=False,
                          jitter=0.28, alpha=0.45, size=3.2)
            kind = "strip"
        ax.set_xlabel(col)
        ax.set_ylabel(CHARGES_LABEL)
        ax.set_title(f"charges by {col} ({kind})")
        _usd(ax)
        if col == "region":
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Individual observations of charges by category",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_strip_swarm", "Strip/swarm plots of charges by category")


# ==========================================================================
# 9. cat2_bar_ci — mean charges per category with 95% CI (1x3), value labels
# ==========================================================================
def fig_bar_ci():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, pal, order) in zip(axes, CAT_FEATURES):
        order = _cat_order(col, order)
        sns.barplot(data=df, x=col, y="charges", ax=ax, order=order,
                    hue=col, palette=pal, legend=False,
                    errorbar=("ci", 95), capsize=0.15,
                    err_kws=dict(color="#333333", linewidth=1.4))
        ax.set_xlabel(col)
        ax.set_ylabel("mean charges (USD)")
        ax.set_title(f"mean charges by {col}")
        # value labels above error caps
        means = df.groupby(col, observed=True)["charges"].mean().reindex(order)
        ymax = ax.get_ylim()[1]
        for i, m in enumerate(means.values):
            ax.text(i, m + ymax * 0.02, f"${m:,.0f}", ha="center", va="bottom",
                    fontsize=10, weight="bold", color="#222222")
        ax.set_ylim(0, ymax * 1.12)
        _usd(ax)
        if col == "region":
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Mean charges by category with 95% confidence intervals",
                 fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat2_bar_ci", "Mean charges per category with 95% CI")


# ==========================================================================
# 10. cat2_catplot_faceted — charges by smoker, faceted by region
# ==========================================================================
def fig_catplot_faceted():
    g = sns.catplot(
        data=df, x="smoker", y="charges", col="region", kind="box",
        order=SMOKER_ORDER, col_order=REGION_ORDER,
        hue="smoker", hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
        legend=False, height=4.6, aspect=0.72, fliersize=2, linewidth=1.1,
    )
    g.set_axis_labels("smoker", CHARGES_LABEL)
    g.set_titles("region = {col_name}")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    g.figure.suptitle("charges by smoker status, faceted by region",
                      fontsize=16, weight="bold", y=1.03)
    save(g.figure, "cat2_catplot_faceted",
         "Faceted boxplot of charges by smoker across regions")


# ==========================================================================
# 11. cat2_smoker_box_points — box + overlaid points, annotate mean gap
# ==========================================================================
def fig_smoker_box_points():
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.boxplot(data=df, x="smoker", y="charges", ax=ax, order=SMOKER_ORDER,
                hue="smoker", palette=SMOKER_PALETTE, legend=False,
                width=0.55, fliersize=0, linewidth=1.5,
                boxprops=dict(alpha=0.55))
    sns.stripplot(data=df, x="smoker", y="charges", ax=ax, order=SMOKER_ORDER,
                  hue="smoker", palette=SMOKER_PALETTE, legend=False,
                  jitter=0.22, alpha=0.5, size=3.4, edgecolor="none")
    ax.set_xlabel("smoker")
    ax.set_ylabel(CHARGES_LABEL)
    ax.set_title("Smoking status is the dominant driver of insurance charges")
    _usd(ax)

    means = df.groupby("smoker", observed=True)["charges"].mean()
    m_no, m_yes = means["no"], means["yes"]
    gap = m_yes - m_no
    # mean markers
    for i, (lvl, m) in enumerate([("no", m_no), ("yes", m_yes)]):
        ax.scatter(i, m, marker="D", s=90, color="white",
                   edgecolor="black", zorder=5)
        ax.text(i + 0.30, m, f"mean\n${m:,.0f}", va="center", ha="left",
                fontsize=10.5, weight="bold")
    # gap annotation
    ax.annotate(
        "", xy=(1.0, m_yes), xytext=(1.0, m_no),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.8))
    ax.text(1.06, (m_no + m_yes) / 2,
            f"mean gap\n≈ ${gap:,.0f}\n(~3.8×)",
            va="center", ha="left", fontsize=11, weight="bold",
            color=WONG["vermillion"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=WONG["vermillion"], alpha=0.95))
    ax.set_xlim(-0.6, 1.9)
    fig.tight_layout()
    save(fig, "cat2_smoker_box_points",
         "Boxplot of charges by smoker with points and mean-gap annotation")


# ==========================================================================
if __name__ == "__main__":
    fig_scatter_regline()
    fig_scatter_loess()
    fig_hexbin()
    fig_jointplots()
    fig_residuals()
    fig_box()
    fig_violin()
    fig_strip_swarm()
    fig_bar_ci()
    fig_catplot_faceted()
    fig_smoker_box_points()
    print("\nAll Category 2 figures generated.")
