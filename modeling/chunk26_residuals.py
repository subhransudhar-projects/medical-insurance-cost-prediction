"""
Stage 4 · Chunk 26 — Residual analysis of the selected model (Gradient Boosting).

Diagnoses WHERE the model errs: residuals vs fitted (bias/heteroscedasticity),
residual distribution, Q-Q vs normal, and a Shapiro-Wilk normality test.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from model_utils import load_splits, load_model, set_style, save_fig, BLUE, VERMILLION, NAVY

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
gb = load_model("gradient_boosting")

y_pred = gb.predict(X_test)
resid = y_test.values - y_pred

print("=" * 66); print("CHUNK 26 — RESIDUAL ANALYSIS (Gradient Boosting, test set)"); print("=" * 66)
print(f"n test            : {len(resid)}")
print(f"Mean residual     : {resid.mean():,.1f}  (near 0 => unbiased on average)")
print(f"Median residual   : {np.median(resid):,.1f}")
print(f"Residual std      : {resid.std():,.1f}")
print(f"Min / Max residual: {resid.min():,.0f} / {resid.max():,.0f}")

# Shapiro-Wilk normality
sw_stat, sw_p = stats.shapiro(resid)
print(f"\nShapiro-Wilk: W = {sw_stat:.4f}, p = {sw_p:.2e}  "
      f"=> residuals are {'NOT ' if sw_p < 0.05 else ''}normally distributed")

# Heteroscedasticity: correlation of |resid| with fitted
het_r, het_p = stats.pearsonr(np.abs(resid), y_pred)
print(f"Heteroscedasticity: corr(|resid|, fitted) = {het_r:.3f}, p = {het_p:.2e}  "
      f"=> {'error grows with predicted cost' if het_p < 0.05 and het_r > 0 else 'roughly constant spread'}")

# --- 2x2 diagnostic figure ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.scatter(y_pred, resid, s=18, alpha=0.5, color=BLUE, edgecolors="none")
ax.axhline(0, color=VERMILLION, linewidth=1.5)
ax.set_title("(a) Residuals vs Fitted", loc="left")
ax.set_xlabel("predicted charges (USD)"); ax.set_ylabel("residual (actual - predicted)")

ax = axes[0, 1]
ax.hist(resid, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
ax.axvline(0, color=VERMILLION, linewidth=1.5)
ax.set_title("(b) Residual Distribution", loc="left")
ax.set_xlabel("residual (USD)"); ax.set_ylabel("count")

ax = axes[1, 0]
stats.probplot(resid, dist="norm", plot=ax)
ax.get_lines()[0].set(marker="o", markersize=4, alpha=0.5, color=BLUE)
ax.get_lines()[1].set(color=VERMILLION, linewidth=1.5)
ax.set_title("")  # clear scipy's centered "Probability Plot" title
ax.set_title("(c) Q-Q Plot (residuals vs normal)", loc="left")

ax = axes[1, 1]
ax.scatter(y_pred, np.abs(resid), s=18, alpha=0.5, color=BLUE, edgecolors="none")
ax.set_title(f"(d) |Residual| vs Fitted  (corr={het_r:.2f})", loc="left")
ax.set_xlabel("predicted charges (USD)"); ax.set_ylabel("|residual| (USD)")

fig.suptitle("Gradient Boosting — Residual Diagnostics (test set)",
             fontsize=16, fontweight="bold", y=0.995)
fig.tight_layout()
save_fig(fig, "chunk26_residual_diagnostics")

# Where are the biggest misses?
big = pd.DataFrame({"actual": y_test.values, "predicted": y_pred, "residual": resid,
                    "smoker": X_test["smoker"].values, "bmi": X_test["bmi"].values,
                    "age": X_test["age"].values})
print("\n--- 5 largest under-predictions (model too low) ---")
print(big.nlargest(5, "residual")[["actual", "predicted", "residual", "smoker", "bmi", "age"]].round(0).to_string(index=False))
print("\n--- 5 largest over-predictions (model too high) ---")
print(big.nsmallest(5, "residual")[["actual", "predicted", "residual", "smoker", "bmi", "age"]].round(0).to_string(index=False))
