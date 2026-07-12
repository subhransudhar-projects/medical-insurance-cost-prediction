"""
CATEGORY 5 — Advanced & Specialized visualizations
Medical Insurance Cost Dataset.

Produces:
  3D scatter / surface  (Plotly HTML + PNG, plus matplotlib fallback)
  Geographic schematic maps (Plotly choropleth + bubble map -> HTML + PNG)
  Age-progression trend  (matplotlib PNG+PDF)
  Static multi-panel summary dashboard (matplotlib PNG+PDF)
  Interactive Plotly dashboard with dropdown filters (HTML + PNG)

Run (from the dataset dir):
  "<.venv>\\Scripts\\python.exe" visualizations\\cat5_advanced.py

Every Plotly figure writes an interactive HTML AND a static PNG preview into
visualizations/figures/ (all prefixed cat5_). Matplotlib figures use save().
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

import plotly.express as px
import plotly.graph_objects as go

# Make the script's own folder importable so `from viz_config import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_config import (  # noqa: E402
    set_style, load_data, save,
    WONG, SMOKER_PALETTE, SEX_PALETTE, REGION_PALETTE,
    SEQ_CMAP, DIVERGING_CMAP, ACCENT,
    AGE_GROUP_ORDER, BMI_CATEGORY_ORDER, REGION_ORDER,
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET,
)

set_style()
df = load_data()

FIGDIR = Path(__file__).resolve().parent / "figures"

# Axis labels with units (reused throughout)
LBL_AGE = "age (years)"
LBL_BMI = "bmi (kg/m^2)"
LBL_CHG = "charges (USD)"

# Plotly semantic color maps (hex strings keyed by category value)
PLOTLY_SMOKER = dict(SMOKER_PALETTE)
PLOTLY_REGION = dict(REGION_PALETTE)

# viridis as a plotly-compatible continuous scale name
PLOTLY_SEQ = "Viridis"

PLOTLY_FONT = dict(family="DejaVu Sans, Arial", size=13)


def _write_plotly(fig, name, width=1100, height=720, scale=2):
    """Write an interactive HTML + a static PNG preview for a plotly figure."""
    html_path = FIGDIR / f"{name}.html"
    png_path = FIGDIR / f"{name}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(png_path), width=width, height=height, scale=scale)
    print(f"[saved] {html_path.name} + {png_path.name}")
    return png_path


# ==========================================================================
# 1. cat5_3d_scatter_smoker — 3D scatter age/bmi/charges colored by smoker
# ==========================================================================
def fig_3d_scatter_smoker():
    fig = px.scatter_3d(
        df, x="age", y="bmi", z="charges",
        color="smoker", color_discrete_map=PLOTLY_SMOKER,
        category_orders={"smoker": ["no", "yes"]},
        opacity=0.55,
        title="3D structure of charges by age, bmi and smoking status",
    )
    fig.update_traces(marker=dict(size=3.2, line=dict(width=0)))
    fig.update_layout(
        template="plotly_white", font=PLOTLY_FONT,
        title_font=dict(size=20),
        legend=dict(title="smoker", font=dict(size=13)),
        scene=dict(
            xaxis_title=LBL_AGE, yaxis_title=LBL_BMI, zaxis_title=LBL_CHG,
            xaxis=dict(title_font=dict(size=14)),
            yaxis=dict(title_font=dict(size=14)),
            zaxis=dict(title_font=dict(size=14)),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    _write_plotly(fig, "cat5_3d_scatter_smoker")


# ==========================================================================
# 2. cat5_3d_scatter_region — 3D scatter age/bmi/charges colored by region
# ==========================================================================
def fig_3d_scatter_region():
    fig = px.scatter_3d(
        df, x="age", y="bmi", z="charges",
        color="region", color_discrete_map=PLOTLY_REGION,
        category_orders={"region": REGION_ORDER},
        opacity=0.55,
        title="3D structure of charges by age, bmi and region",
    )
    fig.update_traces(marker=dict(size=3.2, line=dict(width=0)))
    fig.update_layout(
        template="plotly_white", font=PLOTLY_FONT,
        title_font=dict(size=20),
        legend=dict(title="region", font=dict(size=13)),
        scene=dict(
            xaxis_title=LBL_AGE, yaxis_title=LBL_BMI, zaxis_title=LBL_CHG,
            xaxis=dict(title_font=dict(size=14)),
            yaxis=dict(title_font=dict(size=14)),
            zaxis=dict(title_font=dict(size=14)),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    _write_plotly(fig, "cat5_3d_scatter_region")


# ==========================================================================
# 3. cat5_3d_surface_age_bmi — OLS-predicted charges surface over age x bmi
# ==========================================================================
def fig_3d_surface():
    # Fit OLS with interaction on the full data
    model = smf.ols("charges ~ age + bmi + age:bmi", data=df).fit()

    ages = np.linspace(18, 64, 50)
    bmis = np.linspace(16, 53, 50)
    AA, BB = np.meshgrid(ages, bmis)
    grid = pd.DataFrame({"age": AA.ravel(), "bmi": BB.ravel()})
    ZZ = model.predict(grid).values.reshape(AA.shape)

    # matplotlib mplot3d surface via save()
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        AA, BB, ZZ, cmap=SEQ_CMAP, edgecolor="none",
        antialiased=True, alpha=0.95, rstride=1, cstride=1,
    )
    ax.set_xlabel("\n" + LBL_AGE, fontsize=12.5, labelpad=10)
    ax.set_ylabel("\n" + LBL_BMI, fontsize=12.5, labelpad=10)
    ax.set_zlabel("\npredicted charges (USD)", fontsize=12.5, labelpad=12)
    ax.set_title(
        "OLS-predicted charges surface: charges ~ age + bmi + age:bmi\n"
        f"(R² = {model.rsquared:.3f}; charges rise with both age and bmi)",
        fontsize=15, weight="semibold", pad=18,
    )
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.view_init(elev=24, azim=-128)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.10)
    cb.set_label("predicted charges (USD)", fontsize=11)
    cb.formatter = plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k")
    cb.update_ticks()
    fig.tight_layout()
    save(fig, "cat5_3d_surface_age_bmi",
         "OLS predicted charges surface over age x bmi")


# ==========================================================================
# State -> census region mapping (standard 4-region grouping).
# Dataset regions: northeast / northwest / southeast / southwest.
# west == northwest+southwest (western states); east == northeast+southeast.
# We split western states into north/south and eastern states into north/south
# to honestly cover all four dataset labels. This is an explicit schematic.
# ==========================================================================
STATE_REGION = {
    # ---- northeast (NE census + eastern-north) ----
    "ME": "northeast", "NH": "northeast", "VT": "northeast", "MA": "northeast",
    "RI": "northeast", "CT": "northeast", "NY": "northeast", "NJ": "northeast",
    "PA": "northeast", "OH": "northeast", "MI": "northeast", "IN": "northeast",
    "IL": "northeast", "WI": "northeast", "MN": "northeast",
    # ---- southeast ----
    "DE": "southeast", "MD": "southeast", "DC": "southeast", "VA": "southeast",
    "WV": "southeast", "NC": "southeast", "SC": "southeast", "GA": "southeast",
    "FL": "southeast", "KY": "southeast", "TN": "southeast", "AL": "southeast",
    "MS": "southeast", "AR": "southeast", "LA": "southeast",
    # ---- northwest (western-north) ----
    "WA": "northwest", "OR": "northwest", "ID": "northwest", "MT": "northwest",
    "WY": "northwest", "ND": "northwest", "SD": "northwest", "NE": "northwest",
    "IA": "northwest", "AK": "northwest",
    # ---- southwest (western-south) ----
    "CA": "southwest", "NV": "southwest", "UT": "southwest", "CO": "southwest",
    "AZ": "southwest", "NM": "southwest", "TX": "southwest", "OK": "southwest",
    "KS": "southwest", "MO": "southwest", "HI": "southwest",
}


# ==========================================================================
# 4. cat5_region_choropleth — schematic US-state choropleth of mean charges
# ==========================================================================
def fig_region_choropleth():
    region_mean = df.groupby("region", observed=True)["charges"].mean()
    rows = []
    for state, region in STATE_REGION.items():
        rows.append({
            "state": state, "region": region,
            "mean_charges": region_mean[region],
        })
    sdf = pd.DataFrame(rows)

    fig = px.choropleth(
        sdf, locations="state", locationmode="USA-states",
        color="mean_charges", scope="usa",
        color_continuous_scale=PLOTLY_SEQ,
        hover_name="region",
        hover_data={"state": True, "mean_charges": ":$,.0f", "region": True},
        labels={"mean_charges": "mean charges (USD)"},
        title=("Mean insurance charges by dataset region "
               "(SCHEMATIC: 4 regions mapped onto US states)"),
    )
    fig.update_layout(
        template="plotly_white", font=PLOTLY_FONT, title_font=dict(size=18),
        coloraxis_colorbar=dict(title="mean<br>charges<br>(USD)", tickprefix="$"),
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(lakecolor="white"),
    )
    _write_plotly(fig, "cat5_region_choropleth", width=1100, height=680)


# ==========================================================================
# 5. cat5_region_bubble_map — schematic centroid bubbles sized by mean charges
# ==========================================================================
def fig_region_bubble_map():
    region_mean = df.groupby("region", observed=True)["charges"].mean()
    region_n = df.groupby("region", observed=True)["charges"].size()
    # Approximate schematic centroids (lon, lat) for the 4 quadrant regions
    centroids = {
        "northeast": (-75.0, 42.5),
        "southeast": (-82.0, 32.5),
        "northwest": (-112.0, 45.5),
        "southwest": (-112.0, 34.0),
    }
    rows = []
    for region in REGION_ORDER:
        lon, lat = centroids[region]
        rows.append({
            "region": region, "lon": lon, "lat": lat,
            "mean_charges": region_mean[region], "n": int(region_n[region]),
        })
    bdf = pd.DataFrame(rows)

    fig = px.scatter_geo(
        bdf, lon="lon", lat="lat", scope="usa",
        size="mean_charges", color="region",
        color_discrete_map=PLOTLY_REGION,
        category_orders={"region": REGION_ORDER},
        size_max=55,
        hover_name="region",
        hover_data={"mean_charges": ":$,.0f", "n": True,
                    "lon": False, "lat": False},
        text="region",
        title=("Mean insurance charges by region — bubble size = mean charges "
               "(SCHEMATIC centroids)"),
    )
    fig.update_traces(textposition="top center",
                      textfont=dict(size=12, color="#222222"))
    fig.update_layout(
        template="plotly_white", font=PLOTLY_FONT, title_font=dict(size=18),
        legend=dict(title="region"),
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(lakecolor="white", landcolor="#f2f2f2"),
    )
    _write_plotly(fig, "cat5_region_bubble_map", width=1100, height=680)


# ==========================================================================
# 6. cat5_age_progression — smoothed charges vs age, one line per smoker
# ==========================================================================
def fig_age_progression():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Per-age mean + 95% CI band, per smoker group
    for smk in ["no", "yes"]:
        sub = df[df["smoker"] == smk]
        grp = sub.groupby("age")["charges"].agg(["mean", "std", "count"])
        grp = grp.sort_index()
        se = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
        ci = 1.96 * se
        # light rolling smoothing on the mean to tame per-age noise
        sm_mean = grp["mean"].rolling(3, center=True, min_periods=1).mean()
        color = SMOKER_PALETTE[smk]
        ax.plot(grp.index, sm_mean, color=color, lw=2.8,
                label=f"smoker = {smk}", zorder=3)
        ax.fill_between(grp.index, (sm_mean - ci), (sm_mean + ci),
                        color=color, alpha=0.18, zorder=1)
        # raw per-age means as light points
        ax.scatter(grp.index, grp["mean"], color=color, s=14,
                   alpha=0.35, zorder=2)

    ax.set_xlabel(LBL_AGE)
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("Charges rise steadily with age — the smoker gap stays wide "
                 "across the age span")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(title="", loc="upper left", frameon=True)
    ax.set_xlim(18, 64)

    # annotate the persistent gap at the oldest ages
    old_no = df[(df["smoker"] == "no") & (df["age"] >= 60)]["charges"].mean()
    old_yes = df[(df["smoker"] == "yes") & (df["age"] >= 60)]["charges"].mean()
    ax.annotate(
        f"gap at 60+\n≈ ${old_yes - old_no:,.0f}",
        xy=(62, (old_no + old_yes) / 2), xytext=(50, (old_no + old_yes) / 2),
        fontsize=10.5, weight="bold", color=WONG["vermillion"], va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=WONG["vermillion"], alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4),
    )
    fig.tight_layout()
    save(fig, "cat5_age_progression",
         "Smoothed charges-vs-age progression per smoker status")


# ==========================================================================
# 7. cat5_summary_multipanel — static 2x3 dashboard of key insights
# ==========================================================================
def fig_summary_multipanel():
    fig = plt.figure(figsize=(19, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.26)

    def _usd_y(ax):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

    # (a) charges histogram
    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(df["charges"], bins=40, color=WONG["blue"], ax=ax,
                 edgecolor="white", alpha=0.9)
    ax.axvline(df["charges"].mean(), color=ACCENT, ls="--", lw=2,
               label=f"mean ${df['charges'].mean():,.0f}")
    ax.set_xlabel(LBL_CHG)
    ax.set_ylabel("count")
    ax.set_title("(a) Distribution of charges")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(frameon=True)

    # (b) charges by smoker box
    ax = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df, x="smoker", y="charges", order=["no", "yes"],
                hue="smoker", palette=SMOKER_PALETTE, legend=False,
                ax=ax, fliersize=2, linewidth=1.2)
    ax.set_xlabel("smoker")
    ax.set_ylabel(LBL_CHG)
    ax.set_title("(b) Charges by smoking status")
    _usd_y(ax)

    # (c) bmi vs charges colored by smoker
    ax = fig.add_subplot(gs[0, 2])
    for smk in ["no", "yes"]:
        sub = df[df["smoker"] == smk]
        ax.scatter(sub["bmi"], sub["charges"], s=16, alpha=0.45,
                   color=SMOKER_PALETTE[smk], edgecolor="none",
                   label=f"smoker = {smk}")
    ax.axvline(30, color="#777777", ls=":", lw=1.4)
    ax.set_xlabel(LBL_BMI)
    ax.set_ylabel(LBL_CHG)
    ax.set_title("(c) bmi vs charges by smoker")
    _usd_y(ax)
    ax.legend(frameon=True, loc="upper left", fontsize=9)

    # (d) mean charges by age_group x smoker bars
    ax = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df, x="age_group", y="charges", hue="smoker",
                order=AGE_GROUP_ORDER, hue_order=["no", "yes"],
                palette=SMOKER_PALETTE, ax=ax, errorbar=None)
    ax.set_xlabel("age group (years)")
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("(d) Mean charges by age group & smoker")
    _usd_y(ax)
    ax.legend(title="smoker", frameon=True, fontsize=9)

    # (e) region mean charges bars
    ax = fig.add_subplot(gs[1, 1])
    sns.barplot(data=df, x="region", y="charges", order=REGION_ORDER,
                hue="region", palette=REGION_PALETTE, legend=False,
                ax=ax, errorbar=None)
    means = df.groupby("region", observed=True)["charges"].mean().reindex(REGION_ORDER)
    ymax = ax.get_ylim()[1]
    for i, m in enumerate(means.values):
        ax.text(i, m + ymax * 0.02, f"${m:,.0f}", ha="center", va="bottom",
                fontsize=9.5, weight="bold")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("region")
    ax.set_ylabel("mean charges (USD)")
    ax.set_title("(e) Mean charges by region")
    _usd_y(ax)
    ax.tick_params(axis="x", rotation=18)

    # (f) correlation heatmap of numeric cols
    ax = fig.add_subplot(gs[1, 2])
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP,
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws=dict(label="Pearson r", shrink=0.8),
                linewidths=0.5, linecolor="white")
    ax.set_title("(f) Numeric correlations")

    fig.suptitle("Medical Insurance Charges — Key Insights Dashboard",
                 fontsize=21, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "cat5_summary_multipanel",
         "Static multi-panel summary dashboard of key insights")


# ==========================================================================
# 8. cat5_interactive_dashboard — Plotly dashboard w/ dropdown filters
# ==========================================================================
def fig_interactive_dashboard():
    # Build one scatter trace per (smoker) group so color encodes smoker,
    # and use dropdown updatemenus to filter by smoker / sex / region.
    fig = go.Figure()

    # Base traces: one per smoker level (colored). We add all points but will
    # filter via marker opacity / visibility using restyle over x/y arrays.
    # Simpler & robust: store the full arrays and use buttons that pass a
    # boolean mask by rewriting x/y/customdata for a single trace per smoker.
    smoker_levels = ["no", "yes"]

    def masked(mask, col):
        return df.loc[mask, col].tolist()

    base_mask = pd.Series(True, index=df.index)
    for smk in smoker_levels:
        m = base_mask & (df["smoker"] == smk)
        fig.add_trace(go.Scatter(
            x=df.loc[m, "bmi"], y=df.loc[m, "charges"],
            mode="markers", name=f"smoker = {smk}",
            marker=dict(size=7, opacity=0.55, color=SMOKER_PALETTE[smk],
                        line=dict(width=0)),
            customdata=df.loc[m, ["age", "region", "sex"]].values,
            hovertemplate=("bmi=%{x:.1f}<br>charges=$%{y:,.0f}"
                           "<br>age=%{customdata[0]}"
                           "<br>region=%{customdata[1]}"
                           "<br>sex=%{customdata[2]}<extra>"
                           f"smoker={smk}</extra>"),
        ))

    def build_button(label, mask):
        xs, ys, cds = [], [], []
        for smk in smoker_levels:
            m = mask & (df["smoker"] == smk)
            xs.append(df.loc[m, "bmi"].tolist())
            ys.append(df.loc[m, "charges"].tolist())
            cds.append(df.loc[m, ["age", "region", "sex"]].values)
        return dict(
            label=label, method="restyle",
            args=[{"x": xs, "y": ys, "customdata": cds}],
        )

    # Smoker filter buttons
    smoker_buttons = [build_button("All", base_mask)]
    for smk in smoker_levels:
        smoker_buttons.append(build_button(f"smoker: {smk}",
                                           df["smoker"] == smk))

    # Sex filter buttons
    sex_buttons = [build_button("All sexes", base_mask)]
    for sx in ["female", "male"]:
        sex_buttons.append(build_button(f"sex: {sx}", df["sex"] == sx))

    # Region filter buttons
    region_buttons = [build_button("All regions", base_mask)]
    for rg in REGION_ORDER:
        region_buttons.append(build_button(f"region: {rg}", df["region"] == rg))

    fig.update_layout(
        template="plotly_white", font=PLOTLY_FONT,
        title=dict(text=("Interactive explorer — bmi vs charges "
                         "(filter by smoker, sex, region)"),
                   font=dict(size=19)),
        xaxis=dict(title=LBL_BMI, title_font=dict(size=14)),
        yaxis=dict(title=LBL_CHG, title_font=dict(size=14), tickprefix="$"),
        legend=dict(title="colored by smoker", y=0.98, x=0.01,
                    bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=60, r=40, t=110, b=60),
        updatemenus=[
            dict(buttons=smoker_buttons, direction="down", showactive=True,
                 x=0.02, xanchor="left", y=1.14, yanchor="top",
                 pad=dict(t=2, b=2)),
            dict(buttons=sex_buttons, direction="down", showactive=True,
                 x=0.24, xanchor="left", y=1.14, yanchor="top",
                 pad=dict(t=2, b=2)),
            dict(buttons=region_buttons, direction="down", showactive=True,
                 x=0.46, xanchor="left", y=1.14, yanchor="top",
                 pad=dict(t=2, b=2)),
        ],
        annotations=[
            dict(text="smoker:", x=0.02, xref="paper", y=1.185, yref="paper",
                 showarrow=False, font=dict(size=12), xanchor="left"),
            dict(text="sex:", x=0.24, xref="paper", y=1.185, yref="paper",
                 showarrow=False, font=dict(size=12), xanchor="left"),
            dict(text="region:", x=0.46, xref="paper", y=1.185, yref="paper",
                 showarrow=False, font=dict(size=12), xanchor="left"),
        ],
    )
    _write_plotly(fig, "cat5_interactive_dashboard", width=1150, height=760)


# ==========================================================================
if __name__ == "__main__":
    fig_3d_scatter_smoker()
    fig_3d_scatter_region()
    fig_3d_surface()
    fig_region_choropleth()
    fig_region_bubble_map()
    fig_age_progression()
    fig_summary_multipanel()
    fig_interactive_dashboard()
    print("\nAll Category 5 figures generated.")
