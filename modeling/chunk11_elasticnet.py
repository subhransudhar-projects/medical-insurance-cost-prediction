"""
Stage 4 · Chunk 11 — Baseline: Elastic-Net (L1 + L2 hybrid).

Tunes alpha (overall strength) and l1_ratio (L1 vs L2 mix). Useful with
correlated features — spreads weight across a correlated group (like Ridge)
while still able to zero out dead features (like Lasso).
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("enet", ElasticNet(max_iter=100000, random_state=42))])
param_grid = {
    "enet__alpha": [0.001, 0.01, 0.1, 1, 10],
    "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}
grid = GridSearchCV(pipe, param_grid, scoring="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
best_params = {k.replace("enet__", ""): v for k, v in grid.best_params_.items()}

print("=" * 66)
print("CHUNK 11 — ELASTIC-NET")
print("=" * 66)
print(f"Best params         : {best_params}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")

rec = evaluate_and_log(
    "Elastic-Net", best, X_train, y_train, X_test, y_test,
    best_params=best_params, note="Pipeline(StandardScaler+ElasticNet), 5-fold GridSearchCV",
    scaled="pipeline",
)

coef = pd.Series(best.named_steps["enet"].coef_, index=X_train.columns)
n_zero = int((coef == 0).sum())
print(f"\nFeatures dropped to 0: {n_zero}/{len(coef)}  -> {list(coef[coef==0].index) if n_zero else '(none)'}")
print("\n--- Top 8 Elastic-Net coefficients by |value| ---")
print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).round(1).to_string())

reg = load_registry()
comp = reg[reg["model"].isin(
    ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic-Net"])][
    ["model", "test_r2", "test_rmse", "test_mae"]].round(4)
print("\n--- Linear family comparison (test set) ---")
print(comp.to_string(index=False))
