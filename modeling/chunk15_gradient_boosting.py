"""Stage 4 · Chunk 15 — Non-linear: Gradient Boosting Regressor."""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import (load_splits, evaluate_and_log, load_registry, set_style,
                         plot_importances, save_model, VERMILLION)

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {"n_estimators": [50, 100, 200],
     "learning_rate": [0.01, 0.05, 0.1],
     "max_depth": [3, 5, 7]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66); print("CHUNK 15 — GRADIENT BOOSTING"); print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("Gradient Boosting", best, X_train, y_train, X_test, y_test,
                       best_params=grid.best_params_, note="GridSearchCV 5-fold; unscaled", scaled=False)
save_model(best, "gradient_boosting")

imp = plot_importances(best.feature_importances_, X_train.columns,
                       "Gradient Boosting — Feature Importances", "chunk15_gb_importances", color=VERMILLION)
print("\n--- Feature importances (top 8) ---"); print(imp.head(8).round(4).to_string())

reg = load_registry()
rf = reg.loc[reg["model"] == "Random Forest", "test_r2"].iloc[0]
print(f"\nRandom Forest test R2     : {rf:.4f}")
print(f"Gradient Boosting test R2 : {rec['test_r2']:.4f}  ({rec['test_r2']-rf:+.4f} vs RF)")
