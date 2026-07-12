"""
Stage 4 · Chunk 28 — SHAP dependence plots (how each driver's effect varies).

Dependence plots show, for one feature, how its SHAP contribution changes
across its range — and color reveals interaction with a second feature.
"""
import numpy as np
import matplotlib.pyplot as plt
import shap
from model_utils import load_splits, load_model, set_style, FIGDIR

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
gb = load_model("gradient_boosting")
explainer = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)

print("=" * 66); print("CHUNK 28 — SHAP DEPENDENCE PLOTS"); print("=" * 66)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
panels = [
    ("smoker_bmi", "auto", "(a) smoker_bmi (top driver)"),
    ("age", "bmi", "(b) age  (colored by bmi)"),
    ("bmi", "smoker", "(c) bmi  (colored by smoker)"),
    ("smoker", "age", "(d) smoker  (colored by age)"),
]
for ax, (feat, inter, title) in zip(axes.ravel(), panels):
    shap.dependence_plot(feat, shap_values, X_test, interaction_index=inter,
                         ax=ax, show=False)
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold")

fig.suptitle("SHAP Dependence — Gradient Boosting", fontsize=16, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(FIGDIR / "chunk28_shap_dependence.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGDIR / "chunk28_shap_dependence.pdf", bbox_inches="tight")
plt.close(fig)
print("[saved] figures/chunk28_shap_dependence.png")

# Quantify the smoker split for interpretation
sm = X_test["smoker"].values == 1
sv_smoker = shap_values[:, list(X_test.columns).index("smoker_bmi")]
print(f"\nmean SHAP(smoker_bmi) for smokers    : ${sv_smoker[sm].mean():,.0f}")
print(f"mean SHAP(smoker_bmi) for non-smokers: ${sv_smoker[~sm].mean():,.0f}")
sv_age = shap_values[:, list(X_test.columns).index("age")]
young = X_test["age"].values <= 30
print(f"mean SHAP(age) for age<=30 : ${sv_age[young].mean():,.0f}")
print(f"mean SHAP(age) for age>50  : ${sv_age[X_test['age'].values > 50].mean():,.0f}")
