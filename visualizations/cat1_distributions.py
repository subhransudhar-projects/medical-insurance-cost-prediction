"""
CATEGORY 1 — Distribution / Univariate visualizations
Medical Insurance Cost Dataset.

Produces publication-quality univariate distribution figures for the four
numeric variables (age, bmi, children, charges) and the three categorical
variables (sex, smoker, region), plus charges-specific figures.

Run (from the dataset dir):
  "<.venv>\\Scripts\\python.exe" visualizations\\cat1_distributions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Make the script's own folder importable so `from viz_config import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_config import (  # noqa: E402
    set_style, load_data, save, annotate_bars,
    WONG, SMOKER_PALETTE, SEX_PALETTE, REGION_PALETTE, SEQ_CMAP, ACCENT,
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET,
)

set_style()
df = load_data()

# --- Shared metadata -------------------------------------------------------
NUM_LABELS = {
    "age": "age (years)",
    "bmi": "bmi (kg/m^2)",
    "children": "children (count)",
    "charges": "charges (USD)",
}
NUM_TITLES = {
    "age": "Age",
    "bmi": "BMI",
    "children": "Number of children",
    "charges": "Charges",
}
CAT_LABELS = {"sex": "sex", "smoker": "smoker", "region": "region"}
CAT_PALETTES = {"sex": SEX_PALETTE, "smoker": SMOKER_PALETTE, "region": REGION_PALETTE}
# A neutral single-series color for numeric distributions.
NUM_COLOR = WONG["blue"]


def cat_order(col):
    """Deterministic category order (matches palette keys where possible)."""
    if col == "region":
        return ["northeast", "northwest", "southeast", "southwest"]
    if col == "smoker":
        return ["no", "yes"]
    if col == "sex":
        return ["female", "male"]
    return sorted(df[col].unique())


def skew_label(series):
    s = stats.skew(series.dropna())
    tag = "right-skewed" if s > 0.5 else ("left-skewed" if s < -0.5 else "~symmetric")
    return s, tag


# ===========================================================================
# 1. cat1_hist_kde — Histogram WITH KDE overlay (2x2)
# ===========================================================================
def fig_hist_kde():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        bins = df[col].nunique() if col == "children" else 30
        sns.histplot(df[col], bins=bins, kde=True, ax=ax,
                     color=NUM_COLOR, edgecolor="white", alpha=0.75,
                     line_kws={"linewidth": 2.2})
        if ax.lines:
            ax.lines[-1].set_color(ACCENT)
        s, tag = skew_label(df[col])
        ax.set_title(f"{NUM_TITLES[col]}  (skew={s:+.2f}, {tag})")
        ax.set_xlabel(NUM_LABELS[col])
        ax.set_ylabel("count")
    fig.suptitle("Numeric distributions — histogram with KDE overlay",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_hist_kde", "Histograms + KDE for the 4 numeric variables")


# ===========================================================================
# 2. cat1_box_violin — Boxplot with violin overlay (2x2)
# ===========================================================================
def fig_box_violin():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        sns.violinplot(x=df[col], ax=ax, color=WONG["skyblue"],
                       inner=None, cut=0, linewidth=0, alpha=0.45)
        for coll in ax.collections:
            coll.set_alpha(0.45)
        sns.boxplot(x=df[col], ax=ax, width=0.18, showcaps=True,
                    boxprops={"facecolor": WONG["blue"], "alpha": 0.9,
                              "edgecolor": "#222222"},
                    medianprops={"color": WONG["yellow"], "linewidth": 2.4},
                    whiskerprops={"color": "#222222"},
                    capprops={"color": "#222222"},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": ACCENT,
                                "markeredgecolor": "none", "alpha": 0.5})
        ax.set_title(NUM_TITLES[col])
        ax.set_xlabel(NUM_LABELS[col])
        ax.set_ylabel("")
    fig.suptitle("Numeric distributions — violin + box overlay (spread & outliers)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_box_violin", "Violin+box for the 4 numeric variables")


# ===========================================================================
# 3. cat1_density — Smooth KDE density (2x2)
# ===========================================================================
def fig_density():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        sns.kdeplot(df[col], ax=ax, color=NUM_COLOR, fill=True,
                    alpha=0.35, linewidth=2.2, cut=0)
        mean, med = df[col].mean(), df[col].median()
        ax.axvline(mean, color=ACCENT, linestyle="--", linewidth=1.6,
                   label=f"mean = {mean:,.0f}")
        ax.axvline(med, color=WONG["green"], linestyle=":", linewidth=1.8,
                   label=f"median = {med:,.0f}")
        ax.set_title(NUM_TITLES[col])
        ax.set_xlabel(NUM_LABELS[col])
        ax.set_ylabel("density")
        ax.legend(loc="upper right", frameon=True)
    fig.suptitle("Numeric distributions — smooth density (KDE) with mean & median",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_density", "KDE density for the 4 numeric variables")


# ===========================================================================
# 4. cat1_qq — Q-Q plot vs normal (2x2), annotate skew
# ===========================================================================
def fig_qq():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        (osm, osr), (slope, intercept, r) = stats.probplot(
            df[col].dropna(), dist="norm")
        ax.scatter(osm, osr, s=12, color=NUM_COLOR, alpha=0.5,
                   edgecolor="none")
        line_x = np.array([osm.min(), osm.max()])
        ax.plot(line_x, slope * line_x + intercept, color=ACCENT,
                linewidth=2, label="normal reference")
        s, tag = skew_label(df[col])
        ax.set_title(f"{NUM_TITLES[col]}")
        ax.set_xlabel("theoretical quantiles (standard normal)")
        ax.set_ylabel(f"ordered {NUM_LABELS[col]}")
        ax.annotate(f"skew = {s:+.2f}\n({tag})\n$R^2$ = {r**2:.3f}",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=10.5,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                          "edgecolor": "#999999", "alpha": 0.9})
        ax.legend(loc="lower right", frameon=True)
    fig.suptitle("Normality check — Q-Q plots vs normal distribution",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_qq", "Q-Q normality plots for the 4 numeric variables")


# ===========================================================================
# 5. cat1_ecdf — ECDF (2x2)
# ===========================================================================
def fig_ecdf():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        sns.ecdfplot(df[col], ax=ax, color=NUM_COLOR, linewidth=2.2)
        med = df[col].median()
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.2)
        ax.axvline(med, color=ACCENT, linestyle="--", linewidth=1.6,
                   label=f"median = {med:,.0f}")
        ax.set_title(NUM_TITLES[col])
        ax.set_xlabel(NUM_LABELS[col])
        ax.set_ylabel("cumulative proportion")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", frameon=True)
    fig.suptitle("Numeric distributions — empirical cumulative distribution (ECDF)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_ecdf", "ECDF for the 4 numeric variables")


# ===========================================================================
# 6. cat1_bar_counts — Bar chart of counts with value labels (1x3)
# ===========================================================================
def fig_bar_counts():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        order = cat_order(col)
        counts = df[col].value_counts().reindex(order)
        colors = [CAT_PALETTES[col][k] for k in order]
        ax.bar(order, counts.values, color=colors, edgecolor="white",
               alpha=0.9)
        annotate_bars(ax, fmt="{:,.0f}")
        ax.set_title(CAT_LABELS[col].capitalize())
        ax.set_xlabel(CAT_LABELS[col])
        ax.set_ylabel("count")
        ax.set_ylim(0, counts.max() * 1.15)
        if col == "region":
            ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Categorical distributions — counts", fontsize=17,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "cat1_bar_counts", "Count bars for the 3 categoricals")


# ===========================================================================
# 7. cat1_pie — Percentage breakdown pie (1x3)
# ===========================================================================
def fig_pie():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        order = cat_order(col)
        counts = df[col].value_counts().reindex(order)
        colors = [CAT_PALETTES[col][k] for k in order]
        wedges, _texts, autotexts = ax.pie(
            counts.values, labels=order, colors=colors, autopct="%1.1f%%",
            startangle=90, counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 11},
            pctdistance=0.72)
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
            at.set_fontsize(11)
        ax.set_title(CAT_LABELS[col].capitalize())
        ax.axis("equal")
    fig.suptitle("Categorical distributions — percentage breakdown",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "cat1_pie", "Pie % breakdown for the 3 categoricals")


# ===========================================================================
# 8. cat1_waffle — Manual 10x10 waffle grid (1x3)
# ===========================================================================
def fig_waffle():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        order = cat_order(col)
        counts = df[col].value_counts().reindex(order)
        props = counts / counts.sum()
        # Largest-remainder rounding to exactly 100 cells.
        raw = props * 100
        floors = np.floor(raw).astype(int)
        remainder = 100 - floors.sum()
        frac_order = np.argsort(-(raw - floors).values)
        cells = floors.values.copy()
        for i in range(remainder):
            cells[frac_order[i]] += 1

        # Build a 10x10 category index grid, filled column-major bottom-up.
        cat_grid = np.empty((10, 10), dtype=int)
        seq = np.repeat(np.arange(len(order)), cells)
        k = 0
        for c in range(10):
            for r in range(10):
                cat_grid[r, c] = seq[k]
                k += 1
        colors = [CAT_PALETTES[col][kk] for kk in order]
        for r in range(10):
            for c in range(10):
                ax.add_patch(mpatches.Rectangle(
                    (c, r), 0.9, 0.9, facecolor=colors[cat_grid[r, c]],
                    edgecolor="white", linewidth=0.6))
        ax.set_xlim(-0.2, 10)
        ax.set_ylim(-0.2, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(CAT_LABELS[col].capitalize(), pad=10)
        handles = [mpatches.Patch(color=colors[i],
                                  label=f"{order[i]} ({props.iloc[i]*100:.1f}%)")
                   for i in range(len(order))]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), ncol=min(2, len(order)),
                  frameon=False, fontsize=10)
    fig.suptitle("Categorical distributions — waffle charts (1 square = 1%)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "cat1_waffle", "Manual waffle charts for the 3 categoricals")


# ===========================================================================
# 9. cat1_charges_binsensitivity — charges histogram at 15/30/60/100 bins (2x2)
# ===========================================================================
def fig_charges_binsensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, nbins in zip(axes.ravel(), [15, 30, 60, 100]):
        sns.histplot(df[TARGET], bins=nbins, ax=ax, color=NUM_COLOR,
                     edgecolor="white", alpha=0.8)
        ax.set_title(f"{nbins} bins")
        ax.set_xlabel("charges (USD)")
        ax.set_ylabel("count")
    fig.suptitle("Charges histogram — bin-size sensitivity",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "cat1_charges_binsensitivity", "charges histogram at 4 bin counts")


# ===========================================================================
# 10. cat1_charges_cdf — cumulative distribution of charges, overlay by smoker
# ===========================================================================
def fig_charges_cdf():
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.ecdfplot(df[TARGET], ax=ax, color="#444444", linewidth=2.6,
                 label="all")
    for level in ["no", "yes"]:
        sub = df[df["smoker"] == level][TARGET]
        sns.ecdfplot(sub, ax=ax, color=SMOKER_PALETTE[level], linewidth=2.4,
                     linestyle="--", label=f"smoker = {level}")
    med_all = df[TARGET].median()
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.2)
    ax.axvline(med_all, color=ACCENT, linestyle="--", linewidth=1.4)
    ax.annotate(f"overall median\n= ${med_all:,.0f}",
                xy=(med_all, 0.5), xytext=(med_all + 6000, 0.32),
                fontsize=10.5,
                arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_title("Charges — cumulative distribution, overall vs by smoker status",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("charges (USD)")
    ax.set_ylabel("cumulative proportion")
    ax.set_ylim(0, 1.02)
    ax.legend(title="group", loc="lower right", frameon=True)
    fig.tight_layout()
    save(fig, "cat1_charges_cdf", "charges CDF overall + by smoker")


if __name__ == "__main__":
    fig_hist_kde()
    fig_box_violin()
    fig_density()
    fig_qq()
    fig_ecdf()
    fig_bar_counts()
    fig_pie()
    fig_waffle()
    fig_charges_binsensitivity()
    fig_charges_cdf()
    print("\nAll Category 1 figures generated.")
