"""
CATEGORY 4 — Correlation & Multivariate visualizations for the
Medical Insurance Cost Dataset.

Run (from the dataset directory):
    "....venv\\Scripts\\python.exe" visualizations\\cat4_multivariate.py

Produces PNG+PDF figures into visualizations/figures/ (all prefixed cat4_).
"""
import sys
from pathlib import Path

# Make sibling viz_config importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas.plotting import parallel_coordinates, radviz, andrews_curves

from viz_config import (
    set_style, load_data, save,
    SMOKER_PALETTE, REGION_PALETTE, DIVERGING_CMAP, WONG,
    REGION_ORDER, NUMERIC_COLS,
)

# --------------------------------------------------------------------------
set_style()
df = load_data()

NUM = ["age", "bmi", "children", "charges"]
NUM_LABELS = {
    "age": "age (years)",
    "bmi": "bmi (kg/m$^2$)",
    "children": "children (count)",
    "charges": "charges (USD)",
}
SMOKER_ORDER = ["no", "yes"]
CHARGES_LABEL = "charges (USD)"


def _usd_axis(ax, which="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k")
    if which == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _minmax(frame, cols):
    """Return a copy with the given columns min-max normalized to [0, 1]."""
    out = frame.copy()
    for c in cols:
        lo, hi = out[c].min(), out[c].max()
        out[c] = (out[c] - lo) / (hi - lo)
    return out


# ==========================================================================
# 1. cat4_corr_heatmap — Pearson correlation heatmap of the numerics
#    (+ binary-encoded smoker to expose its dominant ~0.79 correlation)
# ==========================================================================
def fig_corr_heatmap():
    cdf = df[NUM].copy()
    cdf["smoker (yes=1)"] = (df["smoker"] == "yes").astype(int)
    corr = cdf.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.2, 7))
    sns.heatmap(
        corr, ax=ax, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.6,
        linecolor="white", annot_kws=dict(fontsize=11, weight="bold"),
        cbar_kws=dict(label="Pearson r", shrink=0.85),
    )
    ax.set_title("Correlation of numeric features (binary smoker included)",
                 fontsize=15, weight="bold", pad=14)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    save(fig, "cat4_corr_heatmap",
         "Pearson correlation heatmap; binary smoker r~0.79 with charges")


# ==========================================================================
# 2 & 3. Pairplots (scatter matrix) colored by smoker / region
# ==========================================================================
def _pairplot(hue, palette, hue_order, name, caption):
    g = sns.pairplot(
        df, vars=NUM, hue=hue, palette=palette, hue_order=hue_order,
        diag_kind="kde", corner=False,
        plot_kws=dict(alpha=0.45, s=18, edgecolor="none"),
        diag_kws=dict(fill=True, alpha=0.35, common_norm=False),
        height=2.3,
    )
    # Proper axis labels with units.
    for i, rvar in enumerate(NUM):
        for j, cvar in enumerate(NUM):
            ax = g.axes[i][j]
            if ax is None:
                continue
            if i == len(NUM) - 1:
                ax.set_xlabel(NUM_LABELS[cvar])
            if j == 0:
                ax.set_ylabel(NUM_LABELS[rvar])
    if g.legend is not None:
        g.legend.set_title(hue)
    g.figure.suptitle(f"Scatter-matrix of numeric features — colored by {hue}",
                      fontsize=16, weight="bold", y=1.02)
    save(g.fig, name, caption)


def fig_pairplot_smoker():
    _pairplot("smoker", SMOKER_PALETTE, SMOKER_ORDER, "cat4_pairplot_smoker",
              "Pairplot of numeric features colored by smoker status")


def fig_pairplot_region():
    _pairplot("region", REGION_PALETTE, REGION_ORDER, "cat4_pairplot_region",
              "Pairplot of numeric features colored by region")


# ==========================================================================
# 4. cat4_parallel_coords — parallel coordinates (min-max normalized)
# ==========================================================================
def fig_parallel_coords():
    norm = _minmax(df, NUM)
    plot_df = norm[NUM + ["smoker"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6.4))
    parallel_coordinates(
        plot_df, "smoker", ax=ax,
        color=[SMOKER_PALETTE["no"], SMOKER_PALETTE["yes"]],
        alpha=0.28, linewidth=0.9,
    )
    ax.set_title("Parallel coordinates by smoker — axes min-max normalized to [0, 1]",
                 fontsize=15, weight="bold")
    ax.set_ylabel("normalized value (0 = min, 1 = max)")
    ax.set_xlabel("feature (each independently normalized)")
    ax.set_xticklabels([NUM_LABELS[c] for c in NUM])
    ax.grid(True, axis="y", alpha=0.4)
    # Rebuild legend so colors map to smoker levels cleanly.
    handles = [mpatches.Patch(color=SMOKER_PALETTE[k], label=k)
               for k in SMOKER_ORDER]
    ax.legend(handles=handles, title="smoker", loc="upper right")
    fig.tight_layout()
    save(fig, "cat4_parallel_coords",
         "Parallel coordinates of normalized numerics by smoker")


# ==========================================================================
# 5. cat4_radviz — RadViz projection by smoker (normalized inputs)
# ==========================================================================
def fig_radviz():
    norm = _minmax(df, NUM)
    plot_df = norm[NUM + ["smoker"]].copy()
    fig, ax = plt.subplots(figsize=(8.4, 8))
    radviz(plot_df, "smoker", ax=ax,
           color=[SMOKER_PALETTE["no"], SMOKER_PALETTE["yes"]],
           alpha=0.45, s=18)
    ax.set_title("RadViz projection by smoker — normalized age, bmi, children, charges",
                 fontsize=14.5, weight="bold")
    handles = [mpatches.Patch(color=SMOKER_PALETTE[k], label=k)
               for k in SMOKER_ORDER]
    ax.legend(handles=handles, title="smoker", loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "cat4_radviz",
         "RadViz of normalized numeric features colored by smoker")


# ==========================================================================
# 6. cat4_andrews — Andrews curves by smoker (normalized, sampled)
# ==========================================================================
def fig_andrews():
    norm = _minmax(df, NUM)
    plot_df = norm[NUM + ["smoker"]].copy()
    # Sample to reduce overplotting; keep class balance representative.
    parts = [
        g.sample(min(len(g), 150), random_state=42)
        for _, g in plot_df.groupby("smoker", observed=True)
    ]
    sample = pd.concat(parts, ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    andrews_curves(sample, "smoker", ax=ax,
                   color=[SMOKER_PALETTE["no"], SMOKER_PALETTE["yes"]],
                   alpha=0.35, linewidth=0.9)
    ax.set_title("Andrews curves by smoker — normalized features (~300-row sample)",
                 fontsize=14.5, weight="bold")
    ax.set_xlabel("t  (curve parameter, $-\\pi$ to $\\pi$)")
    ax.set_ylabel("f(t) — Fourier encoding of the 4 normalized features")
    handles = [mpatches.Patch(color=SMOKER_PALETTE[k], label=k)
               for k in SMOKER_ORDER]
    ax.legend(handles=handles, title="smoker", loc="upper right")
    fig.tight_layout()
    save(fig, "cat4_andrews",
         "Andrews curves of normalized numerics by smoker (sampled)")


# ==========================================================================
# 7. cat4_facet_age_charges_by_smoker — lmplot age vs charges | smoker
# ==========================================================================
def fig_facet_age_charges_by_smoker():
    g = sns.lmplot(
        data=df, x="age", y="charges", col="smoker", col_order=SMOKER_ORDER,
        hue="smoker", hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
        height=5, aspect=0.95, legend=False,
        scatter_kws=dict(alpha=0.35, s=22, edgecolor="none"),
        line_kws=dict(lw=2.4, color=WONG["black"]),
        ci=95,
    )
    g.set_axis_labels("age (years)", CHARGES_LABEL)
    g.set_titles("smoker = {col_name}")
    for ax in g.axes.flat:
        _usd_axis(ax)
    g.figure.suptitle("age vs charges with linear fit, faceted by smoker",
                      fontsize=16, weight="bold", y=1.03)
    save(g.figure, "cat4_facet_age_charges_by_smoker",
         "Faceted regression of charges on age by smoker")


# ==========================================================================
# 8. cat4_facet_bmi_charges_by_smoker — lmplot bmi vs charges | smoker
# ==========================================================================
def fig_facet_bmi_charges_by_smoker():
    g = sns.lmplot(
        data=df, x="bmi", y="charges", col="smoker", col_order=SMOKER_ORDER,
        hue="smoker", hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
        height=5, aspect=0.95, legend=False,
        scatter_kws=dict(alpha=0.35, s=22, edgecolor="none"),
        line_kws=dict(lw=2.4, color=WONG["black"]),
        ci=95,
    )
    g.set_axis_labels("bmi (kg/m$^2$)", CHARGES_LABEL)
    g.set_titles("smoker = {col_name}")
    for ax in g.axes.flat:
        _usd_axis(ax)
    # Annotate the key contrast: steep vs flat slope.
    axes = g.axes.flat
    for ax, lvl in zip(axes, SMOKER_ORDER):
        sub = df[df["smoker"] == lvl]
        slope = np.polyfit(sub["bmi"], sub["charges"], 1)[0]
        note = ("steep slope:\nbmi drives cost" if lvl == "yes"
                else "flat slope:\nbmi ~ irrelevant")
        ax.text(0.04, 0.95, f"{note}\n(+${slope:,.0f}/bmi unit)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=SMOKER_PALETTE[lvl], alpha=0.95))
    g.figure.suptitle("bmi vs charges with linear fit, faceted by smoker "
                      "— the smoker x bmi interaction",
                      fontsize=15.5, weight="bold", y=1.03)
    save(g.figure, "cat4_facet_bmi_charges_by_smoker",
         "Faceted regression of charges on bmi by smoker (interaction)")


# ==========================================================================
# 9. cat4_facet_age_charges_by_region — lmplot age vs charges | region
# ==========================================================================
def fig_facet_age_charges_by_region():
    g = sns.lmplot(
        data=df, x="age", y="charges", col="region", col_order=REGION_ORDER,
        col_wrap=2, hue="region", hue_order=REGION_ORDER, palette=REGION_PALETTE,
        height=4.4, aspect=1.15, legend=False,
        scatter_kws=dict(alpha=0.35, s=20, edgecolor="none"),
        line_kws=dict(lw=2.3, color=WONG["black"]),
        ci=95,
    )
    g.set_axis_labels("age (years)", CHARGES_LABEL)
    g.set_titles("region = {col_name}")
    for ax in g.axes.flat:
        _usd_axis(ax)
    g.figure.suptitle("age vs charges with linear fit, faceted by region",
                      fontsize=16, weight="bold", y=1.02)
    save(g.figure, "cat4_facet_age_charges_by_region",
         "Faceted regression of charges on age by region")


# ==========================================================================
# 10. cat4_interaction_smoker_region — mean charges, region x smoker
# ==========================================================================
def fig_interaction_smoker_region():
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    sns.pointplot(
        data=df, x="region", y="charges", hue="smoker",
        order=REGION_ORDER, hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
        estimator="mean", errorbar=None, dodge=False, ax=ax,
        markers=["o", "s"], linestyles=["-", "-"], linewidth=2.4,
        markersize=9,
    )
    ax.set_xlabel("region")
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("Interaction: mean charges by region x smoker",
                 fontsize=15, weight="bold")
    _usd_axis(ax)
    # Direct line labels at the right end.
    means = df.groupby(["smoker", "region"], observed=True)["charges"].mean()
    x_last = len(REGION_ORDER) - 1
    for lvl in SMOKER_ORDER:
        y = means[lvl].reindex(REGION_ORDER).iloc[-1]
        ax.text(x_last + 0.08, y, f"smoker = {lvl}", va="center", ha="left",
                fontsize=10.5, weight="bold", color=SMOKER_PALETTE[lvl])
    ax.text(0.03, 0.55,
            "Large, roughly parallel smoker gap across all regions\n"
            "→ big smoker main effect, only modest region interaction",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc=WONG["yellow"],
                      ec="#999999", alpha=0.85))
    ax.set_xlim(-0.4, x_last + 0.9)
    ax.legend(title="smoker", loc="upper left")
    fig.tight_layout()
    save(fig, "cat4_interaction_smoker_region",
         "Interaction plot of mean charges by region and smoker")


# ==========================================================================
# 11. cat4_interaction_smoker_sex — mean charges, sex x smoker
# ==========================================================================
def fig_interaction_smoker_sex():
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    sex_order = sorted(df["sex"].unique())
    sns.pointplot(
        data=df, x="sex", y="charges", hue="smoker",
        order=sex_order, hue_order=SMOKER_ORDER, palette=SMOKER_PALETTE,
        estimator="mean", errorbar=None, dodge=False, ax=ax,
        markers=["o", "s"], linestyles=["-", "-"], linewidth=2.4,
        markersize=9,
    )
    ax.set_xlabel("sex")
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("Interaction: mean charges by sex x smoker",
                 fontsize=15, weight="bold")
    _usd_axis(ax)
    means = df.groupby(["smoker", "sex"], observed=True)["charges"].mean()
    x_last = len(sex_order) - 1
    for lvl in SMOKER_ORDER:
        y = means[lvl].reindex(sex_order).iloc[-1]
        ax.text(x_last + 0.06, y, f"smoker = {lvl}", va="center", ha="left",
                fontsize=10.5, weight="bold", color=SMOKER_PALETTE[lvl])
    ax.text(0.03, 0.55,
            "Near-parallel lines across sex\n→ negligible smoker x sex interaction",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc=WONG["yellow"],
                      ec="#999999", alpha=0.85))
    ax.set_xlim(-0.4, x_last + 0.7)
    ax.legend(title="smoker", loc="upper left")
    fig.tight_layout()
    save(fig, "cat4_interaction_smoker_sex",
         "Interaction plot of mean charges by sex and smoker")


# ==========================================================================
if __name__ == "__main__":
    fig_corr_heatmap()
    fig_pairplot_smoker()
    fig_pairplot_region()
    fig_parallel_coords()
    fig_radviz()
    fig_andrews()
    fig_facet_age_charges_by_smoker()
    fig_facet_bmi_charges_by_smoker()
    fig_facet_age_charges_by_region()
    fig_interaction_smoker_region()
    fig_interaction_smoker_sex()
    print("\nAll Category 4 figures generated.")
