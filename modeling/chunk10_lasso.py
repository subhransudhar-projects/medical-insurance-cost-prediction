"""
Stage 4 · Chunk 10 — Baseline: Lasso Regression (L1).

L1 regularization can drive coefficients to EXACTLY zero -> built-in feature
selection. We inspect which drivers survive and which get dropped.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("lasso", Lasso(max_iter=100000, random_state=42))])
grid = GridSearchCV(
    pipe, {"lasso__alpha": [0.001, 0.01, 0.1, 1, 10]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
best_alpha = grid.best_params_["lasso__alpha"]

print("=" * 66)
print("CHUNK 10 — LASSO REGRESSION")
print("=" * 66)
print(f"Best alpha          : {best_alpha}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
print("\nCV R2 by alpha:")
for a, s in zip(grid.cv_results_["param_lasso__alpha"], grid.cv_results_["mean_test_score"]):
    print(f"  alpha={a:<7} CV R2={s:.4f}")

rec = evaluate_and_log(
    "Lasso Regression", best, X_train, y_train, X_test, y_test,
    best_params={"alpha": best_alpha}, note="Pipeline(StandardScaler+Lasso), 5-fold GridSearchCV",
    scaled="pipeline",
)

# --- Feature selection: which survived vs dropped ---
coef = pd.Series(best.named_steps["lasso"].coef_, index=X_train.columns)
kept = coef[coef != 0].reindex(coef[coef != 0].abs().sort_values(ascending=False).index)
dropped = coef[coef == 0].index.tolist()

print(f"\n--- Features SELECTED ({len(kept)}/{len(coef)}) — nonzero coefficients ---")
print(kept.round(1).to_string())
print(f"\n--- Features DROPPED ({len(dropped)}) — coefficient forced to 0 ---")
print(dropped if dropped else "(none)")

# --- Comparison across linear family ---
reg = load_registry()
comp = reg[reg["model"].isin(["Linear Regression", "Ridge Regression", "Lasso Regression"])][
    ["model", "test_r2", "test_rmse", "test_mae"]
].round(4)
print("\n--- Linear vs Ridge vs Lasso (test set) ---")
print(comp.to_string(index=False))
