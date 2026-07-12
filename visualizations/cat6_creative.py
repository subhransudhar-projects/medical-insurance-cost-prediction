"""
CATEGORY 6 — Creative / Original visualizations
Medical Insurance Cost Dataset.

Five original, high-impact charts that go beyond the standard chart types used
in other categories, each engineered to reveal a real driver of `charges`:

  1. cat6_ridge_charges_by_bmicat_smoker  — manual ridge / joyplot (no joypy)
  2. cat6_waterfall_charges_decomposition — additive cost-premium waterfall
  3. cat6_slope_region_smoker             — slope chart (non-smoker -> smoker)
  4. cat6_diverging_deviation             — diverging deviation-from-mean bars
  5. cat6_sankey_region_smoker_tier       — Sankey flow (plotly; HTML + PNG)

Run (from the dataset dir):
  "<.venv>\\Scripts\\python.exe" visualizations\\cat6_creative.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

# Make the script's own folder importable so `from viz_config import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_config import (  # noqa: E402
    set_style, load_data, save,
    WONG, SMOKER_PALETTE, REGION_PALETTE, ACCENT,
    BMI_CATEGORY_ORDER, REGION_ORDER, TARGET, FIGDIR,
)

set_style()
df = load_data()

SMOKER_ORDER = ["no", "yes"]
SMOKER_LABELS = {"no": "non-smoker", "yes": "smoker"}
OVERALL_MEAN = df[TARGET].mean()


def _usd(x):
    return f"${x:,.0f}"


def _usd_axis(ax, which="y"):
    fmt = mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(fmt)


# ===========================================================================
# 1. cat6_ridge_charges_by_bmicat_smoker — manual ridge / joyplot
#    Overlapping, vertically-offset KDE curves of charges, one row per
#    bmi_category, smoker vs non-smoker densities within each row.
# ===========================================================================
def fig_ridge_bmicat_smoker():
    rows = BMI_CATEGORY_ORDER                 # Underweight -> Obese
    x_grid = np.linspace(0, df[TARGET].quantile(0.995), 512)
    row_gap = 1.0                             # vertical spacing between rows
    height_scale = 1.7                        # curve amplitude (in row units)

    fig, ax = plt.subplots(figsize=(12, 8.5))

    # Draw from top row downward so lower rows overlap those above (ridge look).
    # Row i baseline sits at y = (n_rows - 1 - i) * row_gap.
    n = len(rows)
    for i, cat in enumerate(rows):
        baseline = (n - 1 - i) * row_gap
        sub = df[df["bmi_category"] == cat]
        for sm in SMOKER_ORDER:
            vals = sub[sub["smoker"] == sm][TARGET].dropna().values
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals)
            dens = kde(x_grid)
            dens = dens / dens.max() * height_scale
            ax.fill_between(x_grid, baseline, baseline + dens,
                            color=SMOKER_PALETTE[sm], alpha=0.55,
                            linewidth=0, zorder=i * 2)
            ax.plot(x_grid, baseline + dens, color=SMOKER_PALETTE[sm],
                    linewidth=1.4, alpha=0.95, zorder=i * 2 + 1)
        # Row label (inside plot, left) + baseline guide
        ax.axhline(baseline, color="#CCCCCC", linewidth=0.7, zorder=0)
        ax.text(0.010 * x_grid.max(), baseline + 0.12, cat,
                ha="left", va="bottom", fontsize=13, weight="semibold",
                color="#222222", zorder=50)

    # Legend (proxy patches)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SMOKER_PALETTE[s], alpha=0.7,
                     label=SMOKER_LABELS[s]) for s in SMOKER_ORDER]
    ax.legend(handles=handles, title="smoker status",
              loc="upper right", frameon=True)

    ax.annotate("Smoker densities shift sharply right\n"
                "as BMI rises — Obese smokers peak near $40k+",
                xy=(0.62, 0.30), xycoords="axes fraction",
                fontsize=11, color=WONG["vermillion"], weight="semibold",
                ha="left")

    ax.set_yticks([])
    ax.set_ylim(-0.15, (n - 1) * row_gap + height_scale + 0.4)
    ax.set_xlim(0, x_grid.max())
    _usd_axis(ax, "x")
    ax.set_xlabel("charges (USD)")
    ax.set_ylabel("BMI category  (rows: Underweight → Obese)")
    ax.set_title("Charges density ridgeline by BMI category and smoker status")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "cat6_ridge_charges_by_bmicat_smoker",
         "ridge/joyplot: charges KDE by bmi_category x smoker")


# ===========================================================================
# 2. cat6_waterfall_charges_decomposition — additive cost-premium waterfall
#    Start at overall non-smoker mean; add smoker effect, age effect, obesity
#    effect; arrive at obese-older-smoker mean.
# ===========================================================================
def fig_waterfall_decomposition():
    # Baseline: mean charge for non-smokers (all ages/bmi)
    base = df[df["smoker"] == "no"][TARGET].mean()

    # Target group: older (51-64) obese smokers
    older = df["age_group"] == "51-64"
    obese = df["bmi_category"] == "Obese"
    smk = df["smoker"] == "yes"
    target = df[older & obese & smk][TARGET].mean()

    # Additive attribution of the total gap (base -> target) across 3 factors,
    # measured as marginal main-effects, then rescaled to close exactly.
    smoker_eff = df[smk][TARGET].mean() - df[df["smoker"] == "no"][TARGET].mean()
    age_eff = df[older][TARGET].mean() - df[~older][TARGET].mean()
    bmi_eff = df[obese][TARGET].mean() - df[~obese][TARGET].mean()

    raw = np.array([smoker_eff, age_eff, bmi_eff])
    total_gap = target - base
    scaled = raw / raw.sum() * total_gap        # rescale so bars close to target

    labels = ["Non-smoker\nbaseline", "+ Smoker", "+ Older\n(51–64)",
              "+ Obese\n(BMI≥30)", "Obese older\nsmoker"]
    steps = [base, scaled[0], scaled[1], scaled[2]]  # last is summary

    fig, ax = plt.subplots(figsize=(12, 7.5))
    x = np.arange(len(labels))
    width = 0.62

    # Baseline bar
    ax.bar(x[0], base, width, color=WONG["blue"], edgecolor="white",
           zorder=3)
    ax.text(x[0], base / 2, _usd(base), ha="center", va="center",
            color="white", fontsize=11, weight="bold")

    running = base
    for i in range(1, 4):
        inc = scaled[i - 1]
        ax.bar(x[i], inc, width, bottom=running, color=WONG["orange"],
               edgecolor="white", zorder=3)
        # connector line from previous top
        ax.plot([x[i - 1] + width / 2, x[i] - width / 2],
                [running, running], color="#888888", linewidth=1.1,
                linestyle="--", zorder=2)
        ax.text(x[i], running + inc / 2, f"+{_usd(inc)}", ha="center",
                va="center", color="#222222", fontsize=10.5, weight="bold")
        running += inc

    # Summary bar (full height)
    ax.bar(x[4], running, width, color=WONG["black"], alpha=0.82,
           edgecolor="white", zorder=3)
    ax.plot([x[3] + width / 2, x[4] - width / 2], [running, running],
            color="#888888", linewidth=1.1, linestyle="--", zorder=2)
    ax.text(x[4], running / 2, _usd(running), ha="center", va="center",
            color="white", fontsize=11, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _usd_axis(ax, "y")
    ax.set_ylabel("mean charges (USD)")
    ax.set_ylim(0, running * 1.14)
    ax.set_title("Decomposing the smoker cost premium\n"
                 "from the non-smoker baseline to the obese older smoker")
    ax.annotate(f"Total gap: {_usd(running - base)}",
                xy=(0.5, 0.94), xycoords="axes fraction", ha="center",
                fontsize=11.5, color=WONG["vermillion"], weight="semibold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=WONG["blue"], label="baseline"),
        Patch(facecolor=WONG["orange"], label="incremental effect"),
        Patch(facecolor=WONG["black"], alpha=0.82, label="final segment mean"),
    ], loc="upper left", frameon=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "cat6_waterfall_charges_decomposition",
         "waterfall: additive decomposition of smoker cost premium")


# ===========================================================================
# 3. cat6_slope_region_smoker — slope chart
#    non-smoker | smoker mean charge, one line per region.
# ===========================================================================
def fig_slope_region_smoker():
    means = (df.groupby(["region", "smoker"], observed=True)[TARGET]
               .mean().unstack().reindex(index=REGION_ORDER,
                                         columns=SMOKER_ORDER))
    fig, ax = plt.subplots(figsize=(9.5, 8))
    x0, x1 = 0, 1

    # Declutter the tightly-clustered left (non-smoker) labels: spread their
    # text y-positions apart while leaders point to the true values.
    yrange = means.values.max() - means.values.min()
    min_gap = 0.028 * yrange
    left_order = means["no"].sort_values().index.tolist()   # low -> high
    label_y = {}
    prev = -np.inf
    for region in left_order:
        y = means.loc[region, "no"]
        y = max(y, prev + min_gap)
        label_y[region] = y
        prev = y

    for region in REGION_ORDER:
        y0 = means.loc[region, "no"]
        y1 = means.loc[region, "yes"]
        col = REGION_PALETTE[region]
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=2.6,
                marker="o", markersize=9, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
        ly = label_y[region]
        if abs(ly - y0) > 1e-6:      # leader line to displaced label
            ax.plot([x0 - 0.055, x0 - 0.02], [ly, y0], color=col,
                    linewidth=0.9, alpha=0.8, zorder=2)
        ax.text(x0 - 0.06, ly, f"{region}  {_usd(y0)}", ha="right",
                va="center", fontsize=10.5, color=col, weight="semibold")
        ax.text(x1 + 0.03, y1, f"{_usd(y1)}  {region}", ha="left",
                va="center", fontsize=10.5, color=col, weight="semibold")

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["non-smoker", "smoker"], fontsize=12.5,
                       weight="semibold")
    _usd_axis(ax, "y")
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("Smoker penalty by region (slope chart)")
    ax.annotate("Every region roughly quadruples;\n"
                "rank order is broadly preserved",
                xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=10.5, color="#444444", style="italic")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "cat6_slope_region_smoker",
         "slope chart: region mean charge non-smoker -> smoker")


# ===========================================================================
# 4. cat6_diverging_deviation — diverging deviation-from-mean bars
#    Each smoker x bmi_category segment as deviation from overall mean charge.
# ===========================================================================
def fig_diverging_deviation():
    seg = (df.groupby(["bmi_category", "smoker"], observed=True)[TARGET]
             .mean().reset_index())
    seg["dev"] = seg[TARGET] - OVERALL_MEAN
    seg["label"] = (seg["bmi_category"].astype(str) + " · "
                    + seg["smoker"].map(SMOKER_LABELS))
    seg = seg.sort_values("dev")

    colors = [WONG["orange"] if d > 0 else WONG["blue"] for d in seg["dev"]]

    fig, ax = plt.subplots(figsize=(11.5, 8))
    y = np.arange(len(seg))
    ax.barh(y, seg["dev"], color=colors, edgecolor="white", height=0.72,
            zorder=3)
    ax.axvline(0, color="#333333", linewidth=1.4, zorder=4)

    for yi, d in zip(y, seg["dev"]):
        ha = "left" if d > 0 else "right"
        off = (abs(d) * 0.02 + 400) * (1 if d > 0 else -1)
        ax.text(d + off, yi, f"{'+' if d > 0 else '−'}{_usd(abs(d))}",
                ha=ha, va="center", fontsize=9.8, color="#222222",
                weight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(seg["label"], fontsize=11)
    _usd_axis(ax, "x")
    ax.set_xlabel(f"deviation from overall mean charge  (mean = {_usd(OVERALL_MEAN)})")
    ax.set_title("How each BMI × smoker segment deviates from the average charge")
    # widen x for labels
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 1.18, xmax * 1.18)
    ax.annotate(f"overall mean\n{_usd(OVERALL_MEAN)}",
                xy=(0, len(seg) - 0.6), xytext=(0, len(seg) - 0.6),
                ha="center", va="center", fontsize=10, color="#333333",
                weight="semibold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#333333", lw=1))
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=WONG["orange"], label="above average"),
        Patch(facecolor=WONG["blue"], label="below average"),
    ], loc="lower right", frameon=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "cat6_diverging_deviation",
         "diverging deviation-from-mean bars: bmi_category x smoker")


# ===========================================================================
# 5. cat6_sankey_region_smoker_tier — Sankey flow (plotly)
#    region -> smoker -> charge tier (Low <$10k, Mid $10-30k, High >=$30k)
# ===========================================================================
def fig_sankey_region_smoker_tier():
    import plotly.graph_objects as go

    d = df.copy()
    tier_bins = [-np.inf, 10000, 30000, np.inf]
    tier_labels = ["Low (<$10k)", "Mid ($10–30k)", "High (≥$30k)"]
    d["tier"] = pd.cut(d[TARGET], bins=tier_bins, labels=tier_labels)
    d["smoker_label"] = d["smoker"].map(SMOKER_LABELS)

    regions = REGION_ORDER
    smokers = ["non-smoker", "smoker"]
    tiers = tier_labels

    nodes = regions + smokers + tiers
    idx = {name: i for i, name in enumerate(nodes)}

    node_colors = (
        [REGION_PALETTE[r] for r in regions]
        + [SMOKER_PALETTE["no"], SMOKER_PALETTE["yes"]]
        + [WONG["skyblue"], WONG["yellow"], WONG["vermillion"]]
    )

    def _rgba(hexc, a=0.45):
        h = hexc.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    src, tgt, val, lcol = [], [], [], []

    # region -> smoker
    g1 = d.groupby(["region", "smoker_label"], observed=True).size()
    for (reg, sm), c in g1.items():
        src.append(idx[reg]); tgt.append(idx[sm]); val.append(int(c))
        lcol.append(_rgba(REGION_PALETTE[reg]))
    # smoker -> tier
    g2 = d.groupby(["smoker_label", "tier"], observed=True).size()
    for (sm, ti), c in g2.items():
        if c == 0:
            continue
        smhex = SMOKER_PALETTE["no"] if sm == "non-smoker" else SMOKER_PALETTE["yes"]
        src.append(idx[sm]); tgt.append(idx[ti]); val.append(int(c))
        lcol.append(_rgba(smhex))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=nodes, color=node_colors,
            pad=18, thickness=22,
            line=dict(color="white", width=1.2),
        ),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    fig.update_layout(
        title=dict(
            text="Beneficiary flow: region → smoker status → charge tier<br>"
                 "<sup>The High-cost tier is fed almost entirely by smokers</sup>",
            font=dict(size=20)),
        font=dict(family="DejaVu Sans, Arial", size=13, color="#222222"),
        margin=dict(t=90, l=15, r=15, b=20),
        width=1200, height=760, paper_bgcolor="white",
    )
    html_path = FIGDIR / "cat6_sankey_region_smoker_tier.html"
    png_path = FIGDIR / "cat6_sankey_region_smoker_tier.png"
    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)
    print(f"[saved] {html_path.name} (interactive)")
    print(f"[saved] {png_path.name}")


if __name__ == "__main__":
    fig_ridge_bmicat_smoker()
    fig_waterfall_decomposition()
    fig_slope_region_smoker()
    fig_diverging_deviation()
    fig_sankey_region_smoker_tier()
    print("\nAll Category 6 figures generated.")
