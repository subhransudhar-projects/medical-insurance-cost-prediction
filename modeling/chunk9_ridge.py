"""
Stage 4 · Chunk 9 — Baseline: Ridge Regression (L2).

GridSearchCV over alpha with a leak-free Pipeline(StandardScaler -> Ridge):
the scaler is refit inside every CV fold, so tuning never sees held-out stats.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=42))])
grid = GridSearchCV(
    pipe, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
best_alpha = grid.best_params_["ridge__alpha"]

print("=" * 66)
print("CHUNK 9 — RIDGE REGRESSION")
print("=" * 66)
print(f"Best alpha           : {best_alpha}")
print(f"Best CV R2 (5-fold)  : {grid.best_score_:.4f}")
print("\nCV R2 by alpha:")
for a, s in zip(grid.cv_results_["param_ridge__alpha"], grid.cv_results_["mean_test_score"]):
    print(f"  alpha={a:<6} CV R2={s:.4f}")

rec = evaluate_and_log(
    "Ridge Regression", best, X_train, y_train, X_test, y_test,
    best_params={"alpha": best_alpha}, note="Pipeline(StandardScaler+Ridge), 5-fold GridSearchCV",
    scaled="pipeline",
)

# --- Coefficient stability vs Linear baseline (focus on smoker family) ---
coef = pd.Series(best.named_steps["ridge"].coef_, index=X_train.columns)
smoker_family = ["smoker", "smoker_bmi", "smoker_age", "smoker_children", "smoker_mean_charges"]
print("\n--- Ridge coefficients: smoker family (standardized) ---")
print(coef[smoker_family].round(1).to_string())
print(f"Sum of smoker-family coefficients: {coef[smoker_family].sum():,.0f}")
print("(Ridge shrinks the wild offsetting values from Chunk 8 toward a coherent net effect.)")

print("\n--- Top 8 Ridge coefficients by |value| ---")
print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).round(1).to_string())

# --- Comparison with Linear baseline ---
reg = load_registry()
comp = reg[reg["model"].isin(["Linear Regression", "Ridge Regression"])][
    ["model", "test_r2", "test_rmse", "test_mae", "overfit_gap_r2"]
].round(4)
print("\n--- Linear vs Ridge (test set) ---")
print(comp.to_string(index=False))
