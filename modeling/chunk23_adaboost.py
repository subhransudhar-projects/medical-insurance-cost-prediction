"""
Stage 4 · Chunk 23 — Ensemble: MedCost-AdaBoost (research-inspired).

AdaBoost with Decision Tree base learners: fits trees sequentially, each
re-weighting the cases the previous trees got most wrong. A base depth of 4 is
used so each learner is moderately expressive (pure stumps underfit this data).
"""
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import load_splits, evaluate_and_log, load_registry, save_model

X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

base = DecisionTreeRegressor(max_depth=4, random_state=42)
grid = GridSearchCV(
    AdaBoostRegressor(estimator=base, random_state=42),
    {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("=" * 66); print("CHUNK 23 — MEDCOST-ADABOOST"); print("=" * 66)
print(f"Base learner        : DecisionTree(max_depth=4)")
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("MedCost-AdaBoost", best, X_train, y_train, X_test, y_test,
                       best_params={**grid.best_params_, "base": "DecisionTree(max_depth=4)"},
                       note="AdaBoost + DT base, GridSearchCV 5-fold", scaled=False)
save_model(best, "adaboost")

reg = load_registry()
print("\n--- AdaBoost vs top tree models (test set) ---")
names = ["MedCost-AdaBoost", "Gradient Boosting", "XGBoost", "Random Forest"]
print(reg[reg["model"].isin(names)][["model", "test_r2", "test_rmse", "test_mae"]]
      .sort_values("test_r2", ascending=False).round(4).to_string(index=False))
