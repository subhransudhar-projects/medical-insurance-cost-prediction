"""
Stage 4 · Chunk 22 — Ensemble: Stacking Regressor (XGBoost meta-learner).

Same top-4 base models, but the meta-learner is now an XGBoost regressor
(a small one) instead of LinearRegression — can it find non-linear structure
in how the base predictions relate to the truth?
"""
import pandas as pd
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from model_utils import (load_splits, evaluate_and_log, load_registry,
                         build_estimator, save_model)

X_train, X_test, y_train, y_test = load_splits(scaled=False)
reg = load_registry()
individual = reg[~reg["model"].str.contains("Voting|Stacking|Bagging|AdaBoost|MedCost")]
top4 = individual.sort_values("test_r2", ascending=False).head(4)[["model", "best_params"]].to_dict("records")

print("=" * 66); print("CHUNK 22 — STACKING (XGBoost meta-learner)"); print("=" * 66)
print("Base models:", [r["model"] for r in top4])
print("Meta-learner: XGBoost (small: 100 trees, depth 3, lr 0.05)")

estimators = [(r["model"], build_estimator(r["model"], r["best_params"])) for r in top4]
meta = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                    random_state=42, n_jobs=1, objective="reg:squarederror", verbosity=0)
stack = StackingRegressor(estimators=estimators, final_estimator=meta, cv=5, n_jobs=-1)
stack.fit(X_train, y_train)

rec = evaluate_and_log("Stacking (XGB meta)", stack, X_train, y_train, X_test, y_test,
                       best_params={"base": [r["model"] for r in top4], "meta": "XGBoost"},
                       note="StackingRegressor, XGBoost meta, 5-fold internal CV", scaled="mixed")
save_model(stack, "stacking_xgb")

reg2 = load_registry()
print("\n--- All ensembles vs best individual (test set) ---")
names = ["Voting (top 3)", "Stacking (Linear meta)", "Stacking (XGB meta)", "SVR (target-scaled)"]
print(reg2[reg2["model"].isin(names)][["model", "test_r2", "test_rmse", "test_mae"]]
      .sort_values("test_r2", ascending=False).round(4).to_string(index=False))
