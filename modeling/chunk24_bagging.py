"""
Stage 4 · Chunk 24 — Ensemble: Bagging Regressor.

Bootstrap Aggregating: train many trees on random resamples of the data and
average them. Reduces variance (like Random Forest, minus the feature subsampling).
"""
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import load_splits, evaluate_and_log, load_registry, save_model

X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), random_state=42, n_jobs=-1),
    {"n_estimators": [10, 50, 100], "max_samples": [0.5, 0.8, 1.0]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66); print("CHUNK 24 — BAGGING REGRESSOR"); print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("Bagging", best, X_train, y_train, X_test, y_test,
                       best_params=grid.best_params_, note="Bagging + DT base, GridSearchCV 5-fold", scaled=False)
save_model(best, "bagging")

reg = load_registry()
print("\n--- FULL leaderboard so far (test R2) ---")
print(reg[["model", "test_r2", "test_rmse", "test_mae"]]
      .sort_values("test_r2", ascending=False).round(4).to_string(index=False))
