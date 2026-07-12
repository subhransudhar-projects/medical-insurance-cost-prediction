"""Stage 4 · Chunk 14 — Non-linear: Random Forest Regressor."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import (load_splits, evaluate_and_log, load_registry, set_style,
                         plot_importances, save_model, GREEN)

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {"n_estimators": [50, 100, 200],
     "max_depth": [5, 10, 15, None],
     "min_samples_split": [2, 5, 10]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66); print("CHUNK 14 — RANDOM FOREST"); print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("Random Forest", best, X_train, y_train, X_test, y_test,
                       best_params=grid.best_params_, note="GridSearchCV 5-fold; unscaled", scaled=False)
save_model(best, "random_forest")

imp = plot_importances(best.feature_importances_, X_train.columns,
                       "Random Forest — Feature Importances", "chunk14_rf_importances", color=GREEN)
print("\n--- Feature importances (top 8) ---"); print(imp.head(8).round(4).to_string())

reg = load_registry()
dt = reg.loc[reg["model"] == "Decision Tree", "test_r2"].iloc[0]
print(f"\nDecision Tree test R2 : {dt:.4f}")
print(f"Random Forest test R2 : {rec['test_r2']:.4f}  ({rec['test_r2']-dt:+.4f} vs single tree)")
