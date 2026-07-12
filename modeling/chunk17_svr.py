"""
Stage 4 · Chunk 17 — Non-linear: Support Vector Regressor (SVR).

SVR is scale-sensitive on BOTH inputs and target. We scale inputs via a
Pipeline (leak-free per CV fold). We ALSO report a target-scaled variant
(TransformedTargetRegressor) because plain SVR on a USD target in the tens of
thousands tends to underfit badly — an honest, fair comparison.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"svr__C": [0.1, 1, 10, 100], "svr__gamma": [0.01, 0.1, 1],
              "svr__epsilon": [0.01, 0.1, 0.5]}

print("=" * 66); print("CHUNK 17 — SUPPORT VECTOR REGRESSOR (SVR)"); print("=" * 66)

# --- (a) As-specified: scale inputs only ---
pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
grid = GridSearchCV(pipe, param_grid, scoring="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"[inputs scaled only] best params: {grid.best_params_}  CV R2: {grid.best_score_:.4f}")
rec = evaluate_and_log("SVR", grid.best_estimator_, X_train, y_train, X_test, y_test,
                       best_params={k.replace('svr__',''): v for k, v in grid.best_params_.items()},
                       note="Pipeline(StandardScaler+SVR rbf); target NOT scaled", scaled="pipeline")

# --- (b) Fair variant: also scale the target ---
ttr = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
grid2 = GridSearchCV(ttr, {f"regressor__{k}": v for k, v in param_grid.items()},
                     scoring="r2", cv=cv, n_jobs=-1)
grid2.fit(X_train, y_train)
print(f"\n[target also scaled] best params: {grid2.best_params_}  CV R2: {grid2.best_score_:.4f}")
rec2 = evaluate_and_log("SVR (target-scaled)", grid2.best_estimator_, X_train, y_train, X_test, y_test,
                        best_params={k.replace('regressor__svr__',''): v for k, v in grid2.best_params_.items()},
                        note="TransformedTargetRegressor; inputs+target scaled", scaled="pipeline")

print("\nTakeaway: plain SVR on a USD target underfits; scaling the target is")
print("essential for SVR to be competitive on this dataset.")

reg = load_registry()
print("\n--- All non-linear so far (test R2) ---")
print(reg[reg["model"].isin(["Decision Tree","Random Forest","Gradient Boosting","XGBoost","SVR","SVR (target-scaled)"])]
      [["model","test_r2","test_rmse"]].sort_values("test_r2", ascending=False).round(4).to_string(index=False))
