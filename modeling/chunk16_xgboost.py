"""Stage 4 · Chunk 16 — Non-linear: XGBoost Regressor."""
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import (load_splits, evaluate_and_log, load_registry, set_style,
                         plot_importances, save_model, BLUE)

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", verbosity=0),
    {"n_estimators": [50, 100, 200, 500],
     "max_depth": [3, 5, 7],
     "learning_rate": [0.01, 0.05, 0.1],
     "subsample": [0.8, 1.0],
     "colsample_bytree": [0.8, 1.0]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66); print("CHUNK 16 — XGBOOST"); print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("XGBoost", best, X_train, y_train, X_test, y_test,
                       best_params=grid.best_params_, note="GridSearchCV 5-fold; unscaled", scaled=False)
save_model(best, "xgboost")

imp = plot_importances(best.feature_importances_, X_train.columns,
                       "XGBoost — Feature Importances", "chunk16_xgb_importances", color=BLUE)
print("\n--- Feature importances (top 8) ---"); print(imp.head(8).round(4).to_string())

reg = load_registry()
gb = reg.loc[reg["model"] == "Gradient Boosting", "test_r2"].iloc[0]
print(f"\nGradient Boosting test R2 : {gb:.4f}")
print(f"XGBoost test R2           : {rec['test_r2']:.4f}  ({rec['test_r2']-gb:+.4f} vs GB)")
