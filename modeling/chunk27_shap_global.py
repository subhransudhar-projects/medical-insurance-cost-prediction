"""
Stage 4 · Chunk 27 — SHAP global feature importance (Gradient Boosting).

SHAP assigns each feature a signed dollar contribution to every prediction.
Averaging |SHAP| gives a global importance in interpretable USD units, and the
beeswarm shows direction (does a high value push cost up or down?).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from model_utils import load_splits, load_model, set_style, save_fig, FIGDIR, RESULTS, NAVY

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
gb = load_model("gradient_boosting")

explainer = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)          # (n, n_features), USD units
expected = float(np.ravel(explainer.expected_value)[0])

print("=" * 66); print("CHUNK 27 — SHAP GLOBAL FEATURE IMPORTANCE"); print("=" * 66)
print(f"SHAP base value (expected prediction): ${expected:,.0f}")
print(f"SHAP values shape: {shap_values.shape}")

mean_abs = pd.Series(np.abs(shap_values).mean(0), index=X_test.columns).sort_values(ascending=False)
print("\n--- Mean |SHAP| (avg USD impact on a prediction) — top 10 ---")
for f, v in mean_abs.head(10).items():
    print(f"  {f:22} ${v:,.0f}")

# --- Beeswarm summary ---
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, max_display=12)
fig = plt.gcf(); fig.set_size_inches(11, 8)
fig.suptitle("SHAP Summary (beeswarm) — Gradient Boosting", fontsize=14, fontweight="bold")
fig.savefig(FIGDIR / "chunk27_shap_beeswarm.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGDIR / "chunk27_shap_beeswarm.pdf", bbox_inches="tight")
plt.close(fig)
print("[saved] figures/chunk27_shap_beeswarm.png")

# --- SHAP vs built-in importance comparison ---
builtin = pd.Series(gb.feature_importances_, index=X_test.columns)
cmp = pd.DataFrame({
    "shap_mean_abs_usd": mean_abs,
    "shap_rank": mean_abs.rank(ascending=False).astype(int),
    "builtin_importance": builtin,
    "builtin_rank": builtin.rank(ascending=False).astype(int),
}).sort_values("shap_mean_abs_usd", ascending=False)
cmp.to_csv(RESULTS / "shap_vs_builtin_importance.csv")
print("\n--- SHAP vs built-in importance (top 8) ---")
print(cmp.head(8).round(4).to_string())

top = mean_abs.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top.index, top.values, color="#6a51a3")
for y_i, v in enumerate(top.values):
    ax.text(v + top.values.max() * 0.01, y_i, f"${v:,.0f}", va="center", ha="left",
            fontsize=9.5, color=NAVY, fontweight="medium")
ax.set_title("SHAP Feature Importance (mean |SHAP|, USD)", loc="left")
ax.set_xlabel("mean |SHAP| value (USD impact per prediction)")
ax.margins(x=0.14)
save_fig(fig, "chunk27_shap_importance_bar")

print("\n--- TOP 5 DRIVERS (SHAP) ---")
for i, (f, v) in enumerate(mean_abs.head(5).items(), 1):
    print(f"  {i}. {f:22} ${v:,.0f} average impact")
