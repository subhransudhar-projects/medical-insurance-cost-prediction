"""
Shared visualization configuration for the Medical Insurance Cost Dataset.

Every category script imports from this module so that all charts use an
identical, colorblind-safe palette, consistent styling, the exact schema
column names, and the same derived categories (age_group, bmi_category).

Exact column names (do not rename): age, sex, bmi, children, smoker, region, charges
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths -----------------------------------------------------------------
BASE = Path(__file__).resolve().parent            # .../Medical Insurance Cost Dataset/visualizations
DATASET_DIR = BASE.parent                          # .../Medical Insurance Cost Dataset
FIGDIR = BASE / "figures"
SECTIONS = BASE / "sections"
FIGDIR.mkdir(exist_ok=True)
SECTIONS.mkdir(exist_ok=True)

CLEANED = DATASET_DIR / "insurance_cleaned.csv"
RAW = DATASET_DIR / "insurance.csv"

# --- Exact schema ----------------------------------------------------------
NUMERIC_COLS = ["age", "bmi", "children", "charges"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]
TARGET = "charges"

# --- Colorblind-safe palette (Wong 2011, Nature Methods) -------------------
# Safe for deuteranopia/protanopia/tritanopia; avoids red-green confusion.
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermillion": "#D55E00",
    "purple":  "#CC79A7",
}

# Fixed semantic color assignments — identical everywhere for consistency.
SMOKER_PALETTE = {"yes": WONG["vermillion"], "no": WONG["blue"]}
SEX_PALETTE = {"male": WONG["blue"], "female": WONG["orange"]}
REGION_PALETTE = {
    "northeast": WONG["blue"],
    "northwest": WONG["green"],
    "southeast": WONG["vermillion"],
    "southwest": WONG["orange"],
}
SEQ_CMAP = "viridis"          # colorblind-safe sequential
DIVERGING_CMAP = "RdBu_r"     # correlation heatmaps only (paired with vmin/vmax/center)
ACCENT = WONG["vermillion"]   # single-series emphasis

# Ordered category levels
AGE_GROUP_ORDER = ["18-30", "31-40", "41-50", "51-64"]
BMI_CATEGORY_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
REGION_ORDER = ["northeast", "northwest", "southeast", "southwest"]


def set_style():
    """Apply the global publication style. Call once at the top of each script."""
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 15,      # titles >= 14pt
        "axes.titleweight": "semibold",
        "axes.labelsize": 12.5,    # axis labels >= 12pt
        "xtick.labelsize": 10.5,   # tick labels >= 10pt
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "legend.title_fontsize": 11,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_data():
    """Load the cleaned dataset (falls back to raw) and attach derived categories.

    Returns a DataFrame with the exact schema columns plus:
      - age_group     : ordered Categorical (18-30, 31-40, 41-50, 51-64)
      - bmi_category  : ordered Categorical (Underweight, Normal, Overweight, Obese)
    """
    src = CLEANED if CLEANED.exists() else RAW
    df = pd.read_csv(src)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    # Guard: enforce clean categoricals
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df["age_group"] = pd.cut(
        df["age"], bins=[17, 30, 40, 50, 64],
        labels=AGE_GROUP_ORDER, right=True,
    ).astype(pd.CategoricalDtype(AGE_GROUP_ORDER, ordered=True))

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, np.inf],
        labels=BMI_CATEGORY_ORDER, right=False,
    ).astype(pd.CategoricalDtype(BMI_CATEGORY_ORDER, ordered=True))

    return df


def save(fig, name, caption=None):
    """Save a matplotlib figure as PNG (+PDF) into figures/. Returns the png path."""
    png = FIGDIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    if caption:
        print(f"[saved] {png.name} — {caption}")
    else:
        print(f"[saved] {png.name}")
    return png


def annotate_bars(ax, fmt="{:,.0f}", fontsize=10, color="#222222", offset=0.01, prefix=""):
    """Add value labels above vertical bars."""
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(p.get_x() + p.get_width() / 2, h + ymax * offset,
                prefix + fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color, weight="medium")


if __name__ == "__main__":
    set_style()
    df = load_data()
    print("Rows:", len(df), "| Columns:", list(df.columns))
    print("\nage_group counts:\n", df["age_group"].value_counts().reindex(AGE_GROUP_ORDER))
    print("\nbmi_category counts:\n", df["bmi_category"].value_counts().reindex(BMI_CATEGORY_ORDER))
    print("\nSchema dtypes:\n", df.dtypes)
