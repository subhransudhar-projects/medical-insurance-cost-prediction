"""
Stage 4 · Chunk 35 — One-page modeling dashboard (6 curated panels).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from model_utils import (load_splits, load_model, load_registry, set_style, save_fig,
                         VERMILLION, BLUE, NAVY)

set_style()
BASE = Path(__file__).resolve().parent
feat = pd.read_csv(BASE / "insurance_features.csv")
X_train, X_test, y_train, y_test = load_splits(scaled=False)
gb = load_model("gradient_boosting")
reg = load_registry()
SMK = {"yes": VERMILLION, "no": BLUE}

explainer = shap.TreeExplainer(gb)
sv = explainer.shap_values(X_test)
mean_abs = pd.Series(np.abs(sv).mean(0), index=X_test.columns).sort_values(ascending=False)
y_pred = gb.predict(X_test)
resid = y_test.values - y_pred

fig, axes = plt.subplots(2, 3, figsize=(19, 11))

# (a) SHAP importance
ax = axes[0, 0]
top = mean_abs.head(8).iloc[::-1]
ax.barh(top.index, top.values, color="#6a51a3")
ax.set_title("(a) SHAP Feature Importance", loc="left", fontweight="semibold")
ax.set_xlabel("mean |SHAP| (USD)")

# (b) Model comparison
ax = axes[0, 1]
comp = reg.sort_values("test_r2", ascending=False).head(8).sort_values("test_r2")
colors = [VERMILLION if m in comp["model"].tail(3).values else "#9aa7bd" for m in comp["model"]]
ax.barh(comp["model"], comp["test_r2"], color=colors)
ax.set_xlim(0.85, 0.93)
ax.set_title("(b) Model Comparison (test R2, top 8)", loc="left", fontweight="semibold")
ax.set_xlabel("test R2")

# (c) age vs charges by smoker
ax = axes[0, 2]
for g in ["no", "yes"]:
    s = feat[feat["smoker"] == g]
    ax.scatter(s["age"], s["charges"], s=12, alpha=0.4, color=SMK[g], edgecolors="none",
               label=f"smoker={g}")
ax.set_title("(c) Age vs Charges by Smoker", loc="left", fontweight="semibold")
ax.set_xlabel("age (years)"); ax.set_ylabel("charges (USD)")
ax.legend(framealpha=0.9)

# (d) bmi vs charges by smoker
ax = axes[1, 0]
for g in ["no", "yes"]:
    s = feat[feat["smoker"] == g]
    ax.scatter(s["bmi"], s["charges"], s=12, alpha=0.4, color=SMK[g], edgecolors="none",
               label=f"smoker={g}")
ax.axvline(30, color="#888", linestyle=":", linewidth=1.2)
ax.set_title("(d) BMI vs Charges by Smoker", loc="left", fontweight="semibold")
ax.set_xlabel("bmi (kg/m^2)"); ax.set_ylabel("charges (USD)")
ax.legend(framealpha=0.9)

# (e) residuals vs fitted
ax = axes[1, 1]
ax.scatter(y_pred, resid, s=14, alpha=0.5, color=BLUE, edgecolors="none")
ax.axhline(0, color=VERMILLION, linewidth=1.5)
ax.set_title("(e) Residuals vs Fitted (GB)", loc="left", fontweight="semibold")
ax.set_xlabel("predicted charges (USD)"); ax.set_ylabel("residual (USD)")

# (f) predicted vs actual
ax = axes[1, 2]
ax.scatter(y_test, y_pred, s=14, alpha=0.5, color="#009E73", edgecolors="none")
lim = [0, max(y_test.max(), y_pred.max()) * 1.02]
ax.plot(lim, lim, color=NAVY, linewidth=1.3, linestyle="--")
r2 = reg.loc[reg["model"] == "Gradient Boosting", "test_r2"].iloc[0]
ax.text(0.05, 0.92, f"test R2 = {r2:.3f}", transform=ax.transAxes, fontsize=11,
        color=NAVY, fontweight="semibold")
ax.set_title("(f) Predicted vs Actual (GB)", loc="left", fontweight="semibold")
ax.set_xlabel("actual charges (USD)"); ax.set_ylabel("predicted charges (USD)")

fig.suptitle("Medical Insurance Cost Modeling — Results Dashboard",
             fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "chunk35_modeling_dashboard")
print("[done] modeling dashboard")
