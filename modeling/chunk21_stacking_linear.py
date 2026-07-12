"""
Stage 4 · Chunk 21 — Ensemble: Stacking Regressor (Linear meta-learner).

Base models: top 4 individual models. A meta-model (LinearRegression) learns how
to best weight their predictions. Stacking uses internal cross-validated
predictions as meta-features, so it is leak-safe by construction.
"""
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from model_utils import (load_splits, evaluate_and_log, load_registry,
                         build_estimator, save_model)

X_train, X_test, y_train, y_test = load_splits(scaled=False)
reg = load_registry()
# top 4 individual (exclude any ensemble already built)
individual = reg[~reg["model"].str.contains("Voting|Stacking|Bagging|AdaBoost|MedCost")]
top4 = individual.sort_values("test_r2", ascending=False).head(4)[["model", "best_params"]].to_dict("records")

print("=" * 66); print("CHUNK 21 — STACKING (Linear meta-learner)"); print("=" * 66)
print("Base models:", [r["model"] for r in top4])
print("Meta-learner: LinearRegression")

estimators = [(r["model"], build_estimator(r["model"], r["best_params"])) for r in top4]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(),
                          cv=5, n_jobs=-1)
stack.fit(X_train, y_train)

rec = evaluate_and_log("Stacking (Linear meta)", stack, X_train, y_train, X_test, y_test,
                       best_params={"base": [r["model"] for r in top4], "meta": "LinearRegression"},
                       note="StackingRegressor, 5-fold internal CV", scaled="mixed")
save_model(stack, "stacking_linear")

# meta-learner weights (how it combined base models)
meta = stack.final_estimator_
print("\n--- Meta-learner weights on base predictions ---")
for name, w in zip([r["model"] for r in top4], meta.coef_):
    print(f"  {name:24} weight = {w:+.3f}")
print(f"  intercept = {meta.intercept_:,.0f}")

reg2 = load_registry()
best_ind = individual["test_r2"].max()
print(f"\nBest individual test R2 : {best_ind:.4f}")
print(f"Stacking test R2        : {rec['test_r2']:.4f}  ({rec['test_r2']-best_ind:+.4f})")
