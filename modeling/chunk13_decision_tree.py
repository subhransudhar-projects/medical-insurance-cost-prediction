"""
Stage 4 · Chunk 13 — Non-linear: Decision Tree Regressor.

First rule-based model. Trees capture 'if smoker AND high bmi then ...' splits
natively, with no scaling and no manual interaction terms required.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

from model_utils import (load_splits, evaluate_and_log, load_registry,
                         set_style, save_fig, GREEN, NAVY)

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)  # trees: no scaling

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    {"max_depth": [3, 5, 7, 10, 15],
     "min_samples_split": [2, 5, 10],
     "min_samples_leaf": [1, 2, 4]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66)
print("CHUNK 13 — DECISION TREE REGRESSOR")
print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")

rec = evaluate_and_log(
    "Decision Tree", best, X_train, y_train, X_test, y_test,
    best_params=grid.best_params_, note="GridSearchCV 5-fold; unscaled", scaled=False,
)

# --- Feature importances ---
imp = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n--- Feature importances (top 10) ---")
print(imp.head(10).round(4).to_string())

top = imp.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top.index, top.values, color=GREEN)
for y_i, v in enumerate(top.values):
    ax.text(v + top.values.max() * 0.01, y_i, f"{v:.3f}", va="center", ha="left",
            fontsize=9.5, color=NAVY, fontweight="medium")
ax.set_title(f"Decision Tree — Feature Importances (depth={best.get_params()['max_depth']})", loc="left")
ax.set_xlabel("importance (variance reduction share)")
ax.margins(x=0.12)
save_fig(fig, "chunk13_decision_tree_importances")

reg = load_registry()
best_linear = reg[reg["model"].str.contains("Linear|Ridge|Lasso|Elastic|Poly")]["test_r2"].max()
print(f"\nBest linear-family test R2 so far: {best_linear:.4f}")
print(f"Decision Tree test R2           : {rec['test_r2']:.4f}  "
      f"({'+' if rec['test_r2']>best_linear else ''}{rec['test_r2']-best_linear:+.4f} vs best linear)")
