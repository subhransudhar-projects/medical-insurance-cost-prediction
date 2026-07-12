"""
Stage 4 · Chunk 8 — Baseline: Linear Regression.

The simplest, most interpretable model — our yardstick for everything after.
Uses the SCALED features so coefficients are on a comparable (standardized) scale.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from model_utils import (load_splits, evaluate_and_log, set_style, save_fig,
                         VERMILLION, BLUE, NAVY)

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=True)

lr = LinearRegression().fit(X_train, y_train)
rec = evaluate_and_log("Linear Regression", lr, X_train, y_train, X_test, y_test,
                       note="OLS on scaled features; baseline", scaled=True)

# --- Coefficients (standardized: comparable because inputs are scaled) ---
coef = pd.Series(lr.coef_, index=X_train.columns).sort_values(key=np.abs, ascending=False)
print("\n--- Coefficients (standardized; sorted by |value|) ---")
print(f"intercept: {lr.intercept_:,.0f}")
print(coef.round(1).to_string())

# --- Multicollinearity diagnostic (flagged back in Chunk 5) ---
cond = np.linalg.cond(X_train.values)
print(f"\nDesign-matrix condition number: {cond:,.0f}")
print("Large condition number confirms the multicollinearity we flagged in Chunk 5")
print("(smoker, smoker_bmi, smoker_age, smoker_mean_charges all encode smoking).")
print("=> individual OLS coefficients are unstable; regularization (Ch 9-11) fixes this.")

# --- Top-10 coefficient bar chart ---
top = coef.head(10).iloc[::-1]
colors = [VERMILLION if v > 0 else BLUE for v in top.values]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top.index, top.values, color=colors)
for y_i, v in enumerate(top.values):
    ax.text(v + (np.sign(v) * max(abs(top.values)) * 0.01), y_i, f"{v:,.0f}",
            va="center", ha="left" if v > 0 else "right",
            fontsize=9.5, color=NAVY, fontweight="medium")
ax.axvline(0, color="#888888", linewidth=0.8)
ax.set_title("Linear Regression — Top 10 Standardized Coefficients", loc="left")
ax.set_xlabel("coefficient (USD per 1 SD of feature)")
ax.margins(x=0.15)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=VERMILLION, label="increases charges"),
                   Patch(color=BLUE, label="decreases charges")],
          loc="lower right", framealpha=0.9)
save_fig(fig, "chunk8_linear_coefficients")

print(f"\nBaseline set: Test R2 = {rec['test_r2']:.4f}, Test RMSE = ${rec['test_rmse']:,.0f}")
